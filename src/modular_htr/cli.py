import logging
import math
from dataclasses import asdict

import torch
import tyro
from torchinfo import summary

import modular_htr.config
from modular_htr.data.analysis import run_analysis
from modular_htr.data.dataloaders import make_dataloaders
from modular_htr.data.hf_dataset import build_hf_dataset
from modular_htr.data.tokenization import apply_ctc_tokenizer, build_char_tokenizer
from modular_htr.logging_config import add_file_handler, log_memory_usage, setup_logging
from modular_htr.model.HTRModel import HTRModel
from modular_htr.training.ctc import CTCLossWrapper
from modular_htr.training.evaluate import run_evaluate
from modular_htr.training.finetune import (
    freeze_cnn_stages,
    make_unfreeze_callback,
    prepare_finetune_model,
)
from modular_htr.training.loop import Trainer
from modular_htr.training.utils import (
    CheckpointManager,
    configure_torch,
    create_run_dir,
    dump_config,
    get_dataset,
    get_device,
    log_model_info,
    set_all_seeds,
)
from modular_htr.types import MonitorMetric

logger = logging.getLogger(__name__)

type Scheduler = (
    torch.optim.lr_scheduler.OneCycleLR | torch.optim.lr_scheduler.CosineAnnealingLR
)


def build_model(cfg: modular_htr.config.ModelConfig, num_classes: int) -> HTRModel:
    backbone_cfg = asdict(cfg.backbone)
    model_dict = {
        "num_classes": num_classes,
        "backbone_config": backbone_cfg,
        "img_channels": cfg.img_channels,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "seq_encoder": cfg.seq_encoder,
        "height_collapse": cfg.height_collapse,
        "temporal_convolution": cfg.temporal_convolution,
        "fixed_height": cfg.fixed_height,
        "shortcut_ctc": cfg.shortcut_ctc,
        "temporal_dropout": cfg.temporal_dropout,
        "height_attention_dropout": cfg.height_attention_dropout,
        "pos_encoding_dropout": cfg.pos_encoding_dropout,
        "encoder_dropout": cfg.encoder_dropout,
        "classifier_dropout": cfg.classifier_dropout,
        "shortcut_dropout": cfg.shortcut_dropout,
    }
    return HTRModel(**model_dict)


def build_optimizer(
    cfg: modular_htr.config.OptimAdamwConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=device.type == "cuda",
    )
    return optimizer, cfg.lr


def build_scheduler(
    cfg: modular_htr.config.Scheduler,
    optimizer: torch.optim.Optimizer,
    lr: float,
    epochs: int,
    steps_per_epoch: int | None,
) -> Scheduler:
    if isinstance(cfg, modular_htr.config.Cosine):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs, eta_min=cfg.eta_min
        )
        return scheduler
    if isinstance(cfg, modular_htr.config.Onecycle):
        if steps_per_epoch is None:
            raise ValueError(
                "Can't build a OneCycleLR without computing steps_per_epoch."
            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.pct_start,
            anneal_strategy=cfg.anneal_strategy.value,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
        )
        return scheduler
    raise NotImplementedError("Unknown scheduler config")


def run_training(cfg: modular_htr.config.Train) -> None:
    set_all_seeds(cfg.seed)
    configure_torch(benchmark=cfg.trainer.torch_benchmark)
    device = get_device()

    fixed_height = cfg.model.fixed_height

    ds = get_dataset(cfg.data.dataset)
    text_col = cfg.data.dataset.text_col
    tokenizer = build_char_tokenizer(
        ds,
        text_col,
        cfg.data.unicode_normalize,
        cfg.data.strip_space_before_punctuation,
    )

    logger.info(
        "Built the tokenizer with %d chars: %s",
        len(tokenizer),
        tokenizer.to_dict()["alphabet"],
    )

    train_loader, val_loader = make_dataloaders(
        ds,  # type: ignore (should be duck-type compatible)
        tokenizer=tokenizer,
        text_col=text_col,
        image_col=cfg.data.dataset.img_col,
        filter_config=cfg.data.width_filters,
        fixed_height=fixed_height,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        use_bucketing=cfg.data.use_bucketing,
        bin_bucket_widths=False,
        pin_memory=device.type != "cpu",
        augmentation=cfg.data.augmentation,
        invert_image=cfg.data.invert_image,
        fixed_width=cfg.data.fixed_width,
    )

    model = build_model(cfg.model, num_classes=len(tokenizer))
    log_model_info(model)
    model_summary = summary(
        model,
        input_size=(cfg.data.batch_size, 1, fixed_height, 1000),
        col_names=("output_size", "num_params", "kernel_size", "mult_adds"),
        depth=5,
    )
    model.to(device)
    if cfg.trainer.channels_last:
        model = model.to(memory_format=torch.channels_last)  # type: ignore

    # torch.compile is auto-enabled for the backbone in fixed-width mode (see below).
    if cfg.data.fixed_width is not None:
        model.backbone = torch.compile(model.backbone)  # type: ignore[assignment]
        logger.info("Fixed-width mode: compiled backbone with torch.compile.")
        if cfg.trainer.torch_benchmark:
            logger.warning(
                "cudNN benchmark can cause issues with torch.compile. Disable if training slows down a regular intervals."
            )

    epochs = cfg.trainer.epochs
    loss_fn = CTCLossWrapper(blank=tokenizer.blank_index)
    accum_steps = cfg.trainer.accum_steps
    optimizer, lr = build_optimizer(cfg.optim, model, device)
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    scheduler = build_scheduler(
        cfg.scheduler, optimizer, lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    step_per_batch = cfg.scheduler.step_per_batch

    run_dir = create_run_dir(cfg.base_dir, cfg.run_name)
    add_file_handler(run_dir / "train.log")
    log_memory_usage(logger, "After dataset loading")
    with open(f"{run_dir}/model_summary.txt", "w") as f:
        f.write(str(model_summary))
    dump_config(run_dir, cfg)

    if cfg.trainer.checkpoint.compute_error_rates:
        monitor_metric = cfg.trainer.checkpoint.monitor
    else:
        monitor_metric = MonitorMetric.VAL_LOSS

    checkpoint_manager = CheckpointManager(
        save_dir=run_dir,
        monitor=monitor_metric,
        mode=cfg.trainer.checkpoint.metric_mode,
        top_k=cfg.trainer.checkpoint.top_k,
        model_config=model.to_config(),
        tokenizer=tokenizer,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        tokenizer=tokenizer,
        cfg=cfg.trainer,
        scheduler=scheduler,
        step_per_batch=step_per_batch,
        decoder_cfg=cfg.decoder,
        checkpoint_manager=checkpoint_manager,
        augmentation=cfg.data.augmentation,
    )

    model, _metrics_history = trainer.fit(train_loader, val_loader)


def run_finetune(cfg: modular_htr.config.Finetune) -> None:
    set_all_seeds(cfg.seed)
    configure_torch(benchmark=cfg.trainer.torch_benchmark)
    device = get_device()

    finetune_ds = get_dataset(cfg.data.dataset)

    model, merged_tokenizer, _charset_diff = prepare_finetune_model(
        checkpoint_path=cfg.from_checkpoint,
        finetune_ds=finetune_ds,
        drop_unused=cfg.strategy.drop_unused_symbols,
        device=device,
        head_init=cfg.strategy.head_init_mode,
        new_class_init=cfg.strategy.new_symbols_init,
    )
    if cfg.trainer.channels_last:
        model = model.to(memory_format=torch.channels_last)  # type: ignore

    if cfg.data.fixed_width is not None:
        model.backbone = torch.compile(model.backbone)  # type: ignore[assignment]
        logger.info("Fixed-width mode: compiled backbone with torch.compile.")
        if cfg.trainer.torch_benchmark:
            logger.warning(
                "cudNN benchmark can cause issues with torch.compile. Disable if training slows down a regular intervals."
            )

    tokenizer = merged_tokenizer
    # Allow CLI override of strip_space_before_punctuation (default: inherit from checkpoint).
    if cfg.data.strip_space_before_punctuation:
        tokenizer.strip_space_before_punctuation = True
    finetune_ds = apply_ctc_tokenizer(finetune_ds, tokenizer, cfg.data.dataset.text_col)

    # Get fixed_height from the loaded model's config (set during training).
    fixed_height = model.to_config().get("fixed_height")
    if fixed_height is None:
        raise ValueError(
            "Could not determine fixed_height from checkpoint model config."
        )

    train_loader, val_loader = make_dataloaders(
        finetune_ds,  # type: ignore (should be duck-type compatible)
        filter_config=cfg.data.width_filters,
        tokenizer=tokenizer,
        text_col=cfg.data.dataset.text_col,
        image_col=cfg.data.dataset.img_col,
        fixed_height=fixed_height,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        use_bucketing=cfg.data.use_bucketing,
        pin_memory=device.type != "cpu",
        invert_image=cfg.data.invert_image,
        fixed_width=cfg.data.fixed_width,
    )

    epochs = cfg.trainer.epochs
    loss_fn = CTCLossWrapper(blank=tokenizer.blank_index)
    accum_steps = cfg.trainer.accum_steps
    optimizer, lr = build_optimizer(cfg.optim, model, device)
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    scheduler = build_scheduler(
        cfg.scheduler, optimizer, lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    step_per_batch = cfg.scheduler.step_per_batch

    run_dir = create_run_dir(cfg.base_dir, cfg.run_name)
    add_file_handler(run_dir / "train.log")
    log_memory_usage(logger, "After dataset loading")
    dump_config(run_dir, cfg)

    checkpoint_manager = CheckpointManager(
        save_dir=run_dir,
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.metric_mode,
        top_k=cfg.trainer.checkpoint.top_k,
        model_config=model.to_config(),
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        tokenizer=tokenizer,
        cfg=cfg.trainer,
        scheduler=scheduler,
        step_per_batch=step_per_batch,
        decoder_cfg=cfg.decoder,
        checkpoint_manager=checkpoint_manager,
    )

    unfreeze_callback = None
    if cfg.strategy.freeze_n_stages > 0:
        freeze_cnn_stages(model, cfg.strategy.freeze_n_stages)
        if cfg.strategy.unfreeze_epoch > 0:
            unfreeze_callback = make_unfreeze_callback(
                unfreeze_epoch=cfg.strategy.unfreeze_epoch, num_stages=None
            )

    model, _metrics_history = trainer.fit(
        train_loader, val_loader, on_epoch_start=unfreeze_callback
    )


def run_ds_compile(cfg: modular_htr.config.CompileDataset):
    build_hf_dataset(cfg.xml_path, cfg.img_path, cfg.out_path)


def main():
    setup_logging()
    cfg = tyro.cli(
        modular_htr.config.Config,  # type: ignore (tyro doesn't get the type alias)
        config=(
            # tyro.conf.CascadeSubcommandArgs,
            tyro.conf.FlagConversionOff,
            tyro.conf.EnumChoicesFromValues,
            tyro.conf.SuppressFixed,
        ),
    )
    if isinstance(cfg, modular_htr.config.Train):
        run_training(cfg)
    elif isinstance(cfg, modular_htr.config.Finetune):
        run_finetune(cfg)
    elif isinstance(cfg, modular_htr.config.Evaluate):
        run_evaluate(cfg)
    elif isinstance(cfg, modular_htr.config.CompileDataset):
        run_ds_compile(cfg)
    elif isinstance(cfg, modular_htr.config.AnalyseDataset):
        run_analysis(cfg)
    elif isinstance(cfg, modular_htr.config.BuildLM):
        from modular_htr.lm.build import run_build_lm

        run_build_lm(cfg)
    elif isinstance(cfg, modular_htr.config.InterpolateLM):
        from modular_htr.lm.build import run_interpolate_lm

        run_interpolate_lm(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
