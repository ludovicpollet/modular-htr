from typing import Any, Callable, Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.tokenization import CharTokenizer
from src.model.CRNN import CRNN

from .ctc import greedy_ctc_decode, build_beam_decoder, beam_ctc_decode
from .metrics import cer, wer
from .utils import CheckpointManager, format_metrics, make_log_fn


def train_one_epoch(
    model: CRNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler=None,
    scheduler=None,
    step_per_batch: bool = False,
    grad_clip_norm: float | None = 1.0,
    accum_steps: int = 1,
    use_pbar: bool = True,
):
    model.train()
    running_loss = 0.0
    num_batches = 0

    use_autocast = (scaler is not None) and (device.type in ("cuda", "xpu", "hpu"))
    autocast_context = torch.autocast(device_type=device.type, enabled=use_autocast)

    optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(dataloader, desc="Train", leave=False) if use_pbar else dataloader

    for step, batch in enumerate(iterator):
        batch = batch.to(device)

        input_lengths = model.output_lengths(batch.widths)

        with autocast_context:
            logits = model(batch.images)
            loss = loss_fn(logits, batch.targets, input_lengths, batch.target_lengths)
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

            if scheduler is not None and step_per_batch:
                scheduler.step()

        running_loss += loss.item() * accum_steps
        num_batches += 1

    if scheduler is not None and not step_per_batch:
        scheduler.step()

    avg_loss = running_loss / max(1, num_batches)
    return {"loss": avg_loss}


def evaluate(
    model: CRNN,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    tokenizer: CharTokenizer | None = None,
    compute_metrics: bool = False,
    print_samples: int = 0,
    max_batches: int | None = None,
    use_pbar: bool = True,
    log_fn: Callable | None = None,
    beam_decoder=None,
):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    if log_fn is None:
        log_fn = make_log_fn(use_pbar)

    iterator = tqdm(dataloader, desc="Eval", leave=False) if use_pbar else dataloader

    refs = []
    hyps = []

    with torch.inference_mode():
        for batch_id, batch in enumerate(iterator):
            if max_batches is not None and batch_id >= max_batches:
                break
            batch = batch.to(device)

            input_lengths = model.output_lengths(batch.widths)

            logits = model(batch.images)
            loss = loss_fn(logits, batch.targets, input_lengths, batch.target_lengths)

            total_loss += float(loss.item())
            total_batches += 1

            if compute_metrics and tokenizer is not None:
                # decode one batch
                if beam_decoder:
                    decoded = beam_ctc_decode(
                        logits, input_lengths, beam_decoder, tokenizer
                    )
                else:
                    decoded = greedy_ctc_decode(logits, input_lengths, tokenizer)
                refs.extend(batch.texts)
                hyps.extend(decoded)

                if print_samples > 0 and batch_id == 0:
                    for i in range(min(print_samples, len(decoded))):
                        log_fn(
                            f"[val sample {i}] pred: {decoded[i]!r} | gt: {batch.texts[i]!r}"
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
    step_per_batch: bool = False,
    epochs: int = 40,
    grad_clip_norm: float | None = 1.0,
    accum_steps: int = 1,
    log_fn: Callable[[str], None] | None = None,
    tokenizer=None,
    print_samples=3,
    max_batches=None,
    full_eval_interval: int = 5,
    checkpoint_manager: CheckpointManager | None = None,
    metrics_history: list[dict] | None = None,
    use_pbar=True,
    on_epoch_start: Callable | None = None,
    ctc_decoder_type: Literal["greedy", "beam"] = "greedy",
) -> tuple[torch.nn.Module, list[dict]]:
    if metrics_history is None:
        metrics_history = []

    beam_decoder = None
    if ctc_decoder_type == "beam":
        if tokenizer is None:
            raise ValueError("Cannot build a beam search decoder without a tokenizer")
        beam_decoder = build_beam_decoder(tokenizer)

    log = log_fn or make_log_fn(use_pbar)

    epoch_iter = (
        tqdm(range(1, epochs + 1), desc="Epochs") if use_pbar else range(1, epochs + 1)
    )

    for epoch in epoch_iter:
        if on_epoch_start:
            on_epoch_start(epoch, model)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            step_per_batch=step_per_batch,
            grad_clip_norm=grad_clip_norm,
            accum_steps=accum_steps,
            use_pbar=use_pbar,
        )
        compute_metrics = epoch % full_eval_interval == 0
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            tokenizer=tokenizer,
            beam_decoder=beam_decoder,
            compute_metrics=compute_metrics,
            print_samples=print_samples,
            max_batches=max_batches,
            use_pbar=use_pbar,
            log_fn=log,
        )
        log(format_metrics(epoch, train_metrics, val_metrics, optimizer))

        if checkpoint_manager is not None:
            checkpoint_manager.maybe_save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                val_metrics=val_metrics,
            )
            checkpoint_manager.save_last(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                val_metrics=val_metrics,
            )

        history_record: dict[str, Any] = {"epoch": epoch}
        for k, v in train_metrics.items():
            history_record[f"train_{k}"] = float(v)
        for k, v in val_metrics.items():
            history_record[f"val_{k}"] = float(v)
        metrics_history.append(history_record)

    return model, metrics_history
