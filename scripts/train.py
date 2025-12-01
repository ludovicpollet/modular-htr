import torch
from datasets import load_from_disk
from torch import optim

from src.data.dataloaders import make_dataloaders
from src.data.tokenization import apply_ctc_tokenizer, build_char_tokenizer
from src.model.CRNN import CRNN
from src.training.ctc import CTCLossWrapper
from src.training.loop import fit
from src.training.utils import (
    CheckpointManager,
    configure_torch,
    create_run_dir,
    get_device,
    set_all_seeds,
    to_device,
)


def main():
    set_all_seeds(42)
    configure_torch()

    device = get_device()
    print(f"Using device: {device}")

    ds = load_from_disk("data/hfds")

    tokenizer = build_char_tokenizer(ds, text_col="text")

    ds = apply_ctc_tokenizer(ds, tokenizer, text_col="text")

    train_loader, val_loader = make_dataloaders(
        ds,
        tokenizer=tokenizer,
        fixed_height=128,
        batch_size=32,
        num_workers=8,
        use_bucketing=True,
    )
    num_classes = len(tokenizer)

    model = CRNN(img_channels=1, num_classes=num_classes, rnn_layers=2)
    model.to(device)

    use_compile = device.type == "cuda"
    # override to avoid compiler issues with adaptive_avg_pool2d:
    use_compile = False
    if use_compile:
        model = torch.compile(model)

    loss_fn = CTCLossWrapper(blank=tokenizer.blank_index)
    epochs = 150
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=3e-5
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    batch = next(iter(train_loader))
    batch = to_device(batch, device)

    images = batch["images"]
    widths = batch["widths"]
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]

    with torch.inference_mode():
        logits = model(images)

    T, B, C = logits.shape

    print(
        "Sanity shapes:",
        "images",
        images.shape,
        "logits",
        logits.shape,
        "time_reduction",
        model.time_reduction,
        "computed_input_lengths",
        model.output_lengths(widths),
    )

    assert B == images.size(0)
    assert targets.numel() == int(target_lengths.sum())
    assert model.output_lengths(widths).max() <= T

    run_dir = create_run_dir(base_dir="runs", run_name="test-checkpointing")

    print(f"Run directory: {run_dir}")

    checkpoint_manager = CheckpointManager(
        save_dir=run_dir,
        monitor="cer",
        mode="min",
        top_k=3,
    )

    model, _metrics_history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
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
