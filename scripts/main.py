import math
import torch
import tyro
from datasets import load_from_disk

import src.config
from src.data.dataloaders import make_dataloaders
from src.data.tokenization import build_char_tokenizer, apply_ctc_tokenizer
from src.model.CRNN import CRNN
from src.training.ctc import CTCLossWrapper
from src.training.loop import fit
from src.training.utils import (
    CheckpointManager,
    configure_torch,
    create_run_dir,
    dump_config,
    get_device,
    set_all_seeds,
)
from src.training.finetune import (
    prepare_finetune_model,
    freeze_cnn_stages,
    make_unfreeze_callback,
)


type Scheduler = (
    torch.optim.lr_scheduler.OneCycleLR | torch.optim.lr_scheduler.CosineAnnealingLR
)


def build_model(cfg: src.config.CRNNConfig, num_classes: int) -> CRNN:
    model = CRNN(
        img_channels=cfg.img_channels,
        num_classes=num_classes,
        rnn_layers=cfg.rnn_layers,
        rnn_hidden=cfg.rnn_hidden,
        conv_channels=cfg.conv_channels,
        dropout=cfg.dropout,
    )
    return model


def build_optimizer(
    cfg: src.config.OptimAdamwConfig, model: torch.nn.Module, device: torch.device
) -> tuple[torch.optim.Optimizer, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=device.type == "cuda",
    )
    return optimizer, cfg.lr


def build_scheduler(
    cfg: src.config.Scheduler,
    optimizer: torch.optim.Optimizer,
    lr: float,
    epochs: int,
    steps_per_epoch: int | None,
) -> Scheduler:
    if isinstance(cfg, src.config.Cosine):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs, eta_min=cfg.eta_min
        )
        return scheduler
    if isinstance(cfg, src.config.Onecycle):
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
            anneal_strategy=cfg.anneal_strategy,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
        )
        return scheduler
    raise ValueError("Unkown scheduler config")


def run_training(cfg: src.config.Train) -> None:
    set_all_seeds(cfg.seed)
    configure_torch(benchmark=cfg.trainer.torch_benchmark)
    device = get_device()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None  # type: ignore (stubs out of date)

    ds = load_from_disk(cfg.data.dataset.path)
    text_col = cfg.data.dataset.text_col
    tokenizer = build_char_tokenizer(ds, text_col)
    ds = apply_ctc_tokenizer(ds, tokenizer, text_col)

    train_loader, val_loader = make_dataloaders(
        ds,  # type: ignore (should be duck-type compatible)
        fixed_height=cfg.data.fixed_height,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        use_bucketing=cfg.data.use_bucketing,
        pin_memory=device.type != "cpu",
    )

    model = build_model(cfg.model, num_classes=len(tokenizer))
    model.to(device)

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
    dump_config(run_dir, cfg)

    checkpoint_manager = CheckpointManager(
        save_dir=run_dir,
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.metric_mode,
        top_k=cfg.trainer.checkpoint.top_k,
        model_config=model.to_config(),
    )

    model, _metrics_history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        accum_steps=accum_steps,
        loss_fn=loss_fn,
        device=device,
        scaler=scaler,
        scheduler=scheduler,
        step_per_batch=step_per_batch,
        epochs=epochs,
        tokenizer=tokenizer,
        print_samples=cfg.trainer.checkpoint.print_samples,
        full_eval_interval=cfg.trainer.checkpoint.full_eval_interval,
        checkpoint_manager=checkpoint_manager,
        ctc_decoder_type=cfg.decoder.mode,
    )


def run_finetune(cfg: src.config.Finetune) -> None:
    set_all_seeds(cfg.seed)
    configure_torch(benchmark=cfg.trainer.torch_benchmark)
    device = get_device()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None  # type: ignore (stubs out of date)

    finetune_ds = load_from_disk(cfg.data.dataset.path)

    model, merged_tokenizer, _charset_diff = prepare_finetune_model(
        checkpoint_path=cfg.from_checkpoint,
        finetune_ds=finetune_ds,
        drop_unused=cfg.strategy.drop_unused_symbols,
        device=device,
        head_init=cfg.strategy.head_init_mode,
        new_class_init=cfg.strategy.new_symbols_init,
    )

    tokenizer = merged_tokenizer
    finetune_ds = apply_ctc_tokenizer(finetune_ds, tokenizer, cfg.data.dataset.text_col)

    train_loader, val_loader = make_dataloaders(
        finetune_ds,  # type: ignore (should be duck-type compatible)
        fixed_height=cfg.data.fixed_height,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        use_bucketing=cfg.data.use_bucketing,
        pin_memory=device.type != "cpu",
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
    dump_config(run_dir, cfg)

    checkpoint_manager = CheckpointManager(
        save_dir=run_dir,
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.metric_mode,
        top_k=cfg.trainer.checkpoint.top_k,
        model_config=model.to_config(),
    )

    unfreeze_callback = None
    if cfg.strategy.freeze_n_stages > 0:
        freeze_cnn_stages(model, cfg.strategy.freeze_n_stages)
        if cfg.strategy.unfreeze_epoch > 0:
            unfreeze_callback = make_unfreeze_callback(
                unfreeze_epoch=cfg.strategy.unfreeze_epoch, num_stages=None
            )

    model, _metrics_history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        accum_steps=accum_steps,
        loss_fn=loss_fn,
        device=device,
        scaler=scaler,
        scheduler=scheduler,
        step_per_batch=step_per_batch,
        epochs=epochs,
        tokenizer=tokenizer,
        print_samples=cfg.trainer.checkpoint.print_samples,
        full_eval_interval=cfg.trainer.checkpoint.full_eval_interval,
        checkpoint_manager=checkpoint_manager,
        ctc_decoder_type=cfg.decoder.mode,
        on_epoch_start=unfreeze_callback,
    )


def main():
    cfg = tyro.cli(
        src.config.Config,  # type: ignore (tyro doesn't get the type alias)
        config=(tyro.conf.CascadeSubcommandArgs, tyro.conf.FlagConversionOff),
    )
    if isinstance(cfg, src.config.Train):
        run_training(cfg)
    elif isinstance(cfg, src.config.Finetune):
        run_finetune(cfg)
    else:
        print("Don't know what to do.")


if __name__ == "__main__":
    main()
