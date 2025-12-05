import math
import torch
from datasets import load_from_disk
from torch import optim

from src.data.dataloaders import make_dataloaders
from src.data.hf_dataset import build_hf_dataset
from src.data.tokenization import apply_ctc_tokenizer, build_char_tokenizer
from src.model.CRNN import CRNN
from src.training.ctc import CTCLossWrapper
from src.training.loop import fit
from src.training.utils import (
    CheckpointManager,
    configure_torch,
    create_run_dir,
    serialize_optimizer_config,
    dump_config,
    get_device,
    set_all_seeds,
)


def main():
    set_all_seeds(42)
    configure_torch(benchmark=False)

    device = get_device()
    print(f"Using device: {device}")

    # ds = load_from_disk("data/hfds")
    # build_hf_dataset(
    #     "/home/ludovic/Documents/proj/datasets/HOME-alcar/preproc_data/page",
    #     "/home/ludovic/Documents/proj/datasets/HOME-alcar/preproc_data",
    #     "data/home",
    # )
    ds = load_from_disk("data/home")

    print("Building tokenizer...")
    tokenizer = build_char_tokenizer(ds, text_col="text")
    print(f"Done. Tokenizer size: {len(tokenizer)}")

    print("Applying tokenizer to dataset...")
    ds = apply_ctc_tokenizer(ds, tokenizer, text_col="text")
    print("Done.")

    dataset_name = "HOME-alcar"

    fixed_height = 96
    batch_size = 16
    accum_steps = 2

    pin_memory = device.type != "cpu"

    print("Building dataloaders...")

    train_loader, val_loader = make_dataloaders(
        ds,
        fixed_height=fixed_height,
        batch_size=batch_size,
        num_workers=16,
        use_bucketing=True,
        pin_memory=pin_memory,
    )

    print("Done.")

    num_classes = len(tokenizer)
    img_channels = 1
    rnn_layers = 3
    rnn_hidden = 384
    dropout = 0.3
    conv_channels = [64, 128, 256, 256, 512, 512]

    model = CRNN(
        img_channels=img_channels,
        num_classes=num_classes,
        rnn_layers=rnn_layers,
        rnn_hidden=rnn_hidden,
        conv_channels=conv_channels,
        dropout=dropout,
    )
    model.to(device)

    use_compile = device.type == "cuda"
    # override to avoid compiler issues with adaptive_avg_pool2d:
    use_compile = False
    if use_compile:
        model = torch.compile(model)

    loss_fn = CTCLossWrapper(blank=tokenizer.blank_index)
    epochs = 20
    lr = 6e-4
    weight_decay = 1e-4
    
   


    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=device.type == "cuda"
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #   optimizer, T_max=epochs, eta_min=eta_min
    # )
    steps_per_epoch = math.ceil(len(train_loader) /accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=20.0,
    )
    scheduler.step_per_batch = True

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

     # pull the live config back from the created objects for logging
    optimizer_config = serialize_optimizer_config(optimizer)
    run_config = {
        "dataset_name": dataset_name,
        "train_samples": len(ds["train"]),
        "val_samples": len(ds["test"]),
        "fixed_height": fixed_height,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "lr": lr,
        "epochs": epochs,
        "optimizer": optimizer_config,
        "scheduler": scheduler.__class__.__name__,
        "model": model.to_config()
    }


    run_dir = create_run_dir(base_dir="runs", run_name="pretrain-large-v2")
    print(f"Run directory: {run_dir}")

    # for convenience (it goes in checkpoint anyway)
    dump_config(run_dir, run_config)

    checkpoint_manager = CheckpointManager(
        save_dir=run_dir,
        monitor="cer",
        mode="min",
        top_k=3,
        tokenizer=tokenizer,
        config=run_config,
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
        epochs=epochs,
        tokenizer=tokenizer,
        print_samples=3,
        full_eval_interval=1,
        checkpoint_manager=checkpoint_manager,
    )


if __name__ == "__main__":
    main()
