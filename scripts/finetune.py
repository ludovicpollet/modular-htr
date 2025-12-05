from datasets import load_from_disk
import torch
import math

from src.data.tokenization import apply_ctc_tokenizer
from src.training.finetune import (
    prepare_finetune_model, 
    freeze_cnn_stages, 
    make_unfreeze_callback,
)
from src.training.loop import fit
from src.training.utils import (
    configure_torch,
    get_device,
    set_all_seeds,
    CheckpointManager,
    dump_config,
    create_run_dir,
    serialize_optimizer_config
)
from src.data.dataloaders import make_dataloaders
from src.training.ctc import CTCLossWrapper


def main():

    # configuration

    set_all_seeds(42)
    configure_torch(benchmark=False)

    device = get_device()

    finetune_ds_path = "data/hfds"
    checkpoint_path = "runs/2025-12-05_11-32-34_pretrain-large-v2/best_020_cer_0.0962.pt"
    
    dataset_name = "Dg31"

    fixed_height = 96
    batch_size = 32
    accum_steps = 1
    pin_memory = device.type != "cpu"

    epochs = 50
    lr = 1e-4
    weight_decay = 1e-4
    fused = device.type == "cuda"

    pct_start = 0.1
    div_factor = 10.0
    final_div_factor = 10.0

    finetune_ds = load_from_disk(finetune_ds_path)

    model, merged_tokenizer, _charset_diff = prepare_finetune_model(
        checkpoint_path=checkpoint_path,
        finetune_ds=finetune_ds,
        drop_unused=True,
        device=device,
        head_init="copy",
        new_class_init="kaiming",
    )
    tokenizer = merged_tokenizer
    
    finetune_ds = apply_ctc_tokenizer(finetune_ds, tokenizer, text_col="text")
    
    train_loader, val_loader = make_dataloaders(
        finetune_ds,
        fixed_height=fixed_height,
        batch_size=batch_size,
        num_workers=16,
        use_bucketing=True,
        pin_memory=pin_memory,
    )
    

    loss_fn = CTCLossWrapper(blank=tokenizer.blank_index)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=fused
    )

    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
    scheduler.step_per_batch = True
    
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # pull the live config back from the created objects for logging
    optimizer_config = serialize_optimizer_config(optimizer)
    run_config = {
        "dataset_name": dataset_name,
        "finetuned_from": checkpoint_path,
        "train_samples": len(finetune_ds["train"]),
        "val_samples": len(finetune_ds["test"]),
        "fixed_height": fixed_height,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "lr": lr,
        "epochs": epochs,
        "optimizer": optimizer_config,
        "scheduler": scheduler.__class__.__name__,
        "model": model.to_config()
    }


    run_dir = create_run_dir(base_dir="runs", run_name="finetune-new-config-2")
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

    # freezer settings
    freeze_cnn_stages(model, num_stages=5)
    unfreeze_callback = make_unfreeze_callback(unfreeze_epoch=20, num_stages=None)

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
        on_epoch_start=unfreeze_callback,
    )


if __name__ == "__main__":
    main()
