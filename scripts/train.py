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
    dump_config,
    get_device,
    set_all_seeds,
)


def main():
    set_all_seeds(42)
    configure_torch()

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

    fixed_height = 64
    batch_size = 32
    accum_steps = 1

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
    rnn_layers = 2

    model = CRNN(
        img_channels=img_channels, num_classes=num_classes, rnn_layers=rnn_layers
    )
    model.to(device)

    use_compile = device.type == "cuda"
    # override to avoid compiler issues with adaptive_avg_pool2d:
    use_compile = False
    if use_compile:
        model = torch.compile(model)

    loss_fn = CTCLossWrapper(blank=tokenizer.blank_index)
    epochs = 80
    lr = 3e-4
    weight_decay = 1e-5
    eta_min = 2e-5
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    run_config = {
        "dataset_name": dataset_name,
        "train_samples": len(ds["train"]),
        "val_samples": len(ds["test"]),
        "fixed_height": fixed_height,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "eta_min": eta_min,
        "epochs": epochs,
        "model": {
            "img_channels": img_channels,
            "num_classes": num_classes,
            "rnn_layers": rnn_layers,
            "time_reduction": model.time_reduction,
        },
    }

    run_dir = create_run_dir(base_dir="runs", run_name="test-pretrain-optim")
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
