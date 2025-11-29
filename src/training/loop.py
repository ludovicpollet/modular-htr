import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.tokenization import CharTokenizer

from .ctc import greedy_ctc_decode
from .metrics import cer, wer
from .utils import format_metrics, to_device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler=None,
    scheduler=None,
    grad_clip_norm: float | None = 1.0,
    accum_steps: int = 1,
):
    model.train()
    running_loss = 0.0
    num_batches = 0

    use_autocast = (scaler is not None) and (device.type in ("cuda", "xpu", "hpu"))
    autocast_context = torch.autocast(device_type=device.type, enabled=use_autocast)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        batch = to_device(batch, device)

        images = batch["images"]
        targets = batch["targets"]
        widths = batch["widths"]
        target_lengths = batch["target_lengths"]

        input_lengths = model.output_lengths(widths)

        with autocast_context:
            logits = model(images)
            loss = loss_fn(logits, targets, input_lengths, target_lengths)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if (
                scheduler is not None
                and hasattr(scheduler, "step_per_batch")
                and scheduler.step_per_batch
            ):
                scheduler.step()

        running_loss += loss.item() * accum_steps
        num_batches += 1

    if scheduler is not None and not getattr(scheduler, "step_per_batch", False):
        scheduler.step()

    avg_loss = running_loss / max(1, num_batches)
    return {"loss": avg_loss}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    tokenizer: CharTokenizer | None = None,
    compute_metrics: bool = False,
    print_samples: int = 0,
    max_batches: int | None = None,
):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    refs = []
    hyps = []

    with torch.inference_mode():
        for batch_id, batch in enumerate(dataloader):
            if max_batches is not None and batch_id >= max_batches:
                break
            batch = to_device(batch, device)

            images = batch["images"]
            targets = batch["targets"]
            widths = batch["widths"]
            target_lengths = batch["target_lengths"]
            texts = batch["texts"]

            input_lengths = model.output_lengths(widths)

            logits = model(images)
            loss = loss_fn(logits, targets, input_lengths, target_lengths)

            total_loss += float(loss.item())
            total_batches += 1

            if compute_metrics and tokenizer is not None:
                # decode one batch
                decoded = greedy_ctc_decode(logits, input_lengths, tokenizer)
                refs.extend(texts)
                hyps.extend(decoded)

                if print_samples > 0 and batch_id == 0:
                    for i in range(min(print_samples, len(decoded))):
                        print(
                            f"[val sample {i}] pred: {decoded[i]!r} | gt: {texts[i]!r}"
                        )

    avg_loss = total_loss / max(1, total_batches)
    out = {"loss": avg_loss}
    if compute_metrics and tokenizer:
        out["cer"] = cer(refs, hyps)
        out["wer"] = wer(refs, hyps)
    return out


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    scaler=None,
    scheduler=None,
    epochs: int = 40,
    grad_clip_norm: float | None = 1.0,
    accum_steps: int = 1,
    log_fn=print,
    tokenizer=None,
    print_samples=3,
    max_batches=None,
    full_eval_interval: int = 5,
):
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            accum_steps=accum_steps,
        )
        compute_metrics = epoch % full_eval_interval == 0
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            print_samples=print_samples,
            max_batches=max_batches,
        )
        log_fn(format_metrics(epoch, train_metrics, val_metrics, optimizer))
    return model
