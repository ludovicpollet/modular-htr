import csv
import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modular_htr.config import CTCDecoder as CTCDecoderConfig
from modular_htr.config import Trainer as TrainerConfig
from modular_htr.data.tokenization import CharTokenizer
from modular_htr.data.transforms import GPUAugmentation
from modular_htr.model.HTRModel import HTRModel
from modular_htr.types import Augmentation, CTCDecoderMode, DebugMode

from .ctc import CTCLossWrapper, beam_ctc_decode, build_beam_decoder, greedy_ctc_decode
from .metrics import cer, wer
from .utils import CheckpointManager, ScalarMeter, format_metrics, linear_scale


class Trainer:
    def __init__(
        self,
        model: HTRModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: CTCLossWrapper,
        device: torch.device,
        tokenizer: CharTokenizer,
        cfg: TrainerConfig,
        scheduler=None,
        step_per_batch: bool = False,
        decoder_cfg: CTCDecoderConfig | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        augmentation: Augmentation = Augmentation.NONE,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.step_per_batch = step_per_batch

        if decoder_cfg is None:
            decoder_cfg = CTCDecoderConfig()
        self.ctc_decoder_mode = CTCDecoderMode(decoder_cfg.mode)

        self.use_pbars = not self.cfg.disable_pbars
        self.debug_enabled = cfg.debug is not None

        self.use_autocast = cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_autocast else None  # type: ignore (stubs out of date)

        if self.tokenizer.blank_index != self.loss_fn.ctc.blank:
            raise ValueError(
                f"tokenizer blank: {self.tokenizer.blank_index}; ctc_loss blank: {self.loss_fn.ctc.blank}"
            )
        self.beam_decoder = None
        self.beam_temperature = decoder_cfg.temperature
        if self.ctc_decoder_mode is CTCDecoderMode.BEAM:
            if self.tokenizer.blank_index != 0:
                raise ValueError(
                    "Blank index must be 0 to use the Flashlight beam search decoder."
                )
            self.beam_decoder = build_beam_decoder(
                self.tokenizer,
                beam_size=decoder_cfg.beam_size,
                lm_path=decoder_cfg.lm_path,
                lm_weight=decoder_cfg.lm_alpha,
                word_score=decoder_cfg.word_score,
                sil_score=decoder_cfg.sil_score,
            )

        self.augment = augmentation is Augmentation.GPU
        if self.augment:
            self.aug_module = GPUAugmentation(self.device)

    def train_one_epoch(self, current_epoch: int, dataloader: DataLoader):
        running_loss = ScalarMeter(self.device)
        running_loss_main = ScalarMeter(self.device)
        running_loss_shortcut = ScalarMeter(self.device)
        debug_records: list[dict[str, Any]] = []
        shortcut_ctc = False
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        iterator = (
            tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)
            if self.use_pbars
            else dataloader
        )
        for step, batch in enumerate(iterator):
            # --- data ---
            batch = batch.to(self.device)
            input_lengths = self.model.output_lengths(batch.widths)

            images = batch.images
            images = images.contiguous()
            # augment on gpu for the more expensive operations
            if self.augment:
                images = self.aug_module(images)

            if self.cfg.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            # --- forward pass ---
            blank_rate = None
            with self._autocast():
                logits, shortcut_logits = self.model(images, batch.widths)
                T = int(logits.size(0))
                B = int(logits.size(1))

                if self.debug_enabled:
                    blank_rate = self._compute_blank_rate(logits, input_lengths)

                # --- compute loss---
                loss_main = self.loss_fn(
                    logits, batch.targets, input_lengths, batch.target_lengths
                )
                loss_total = loss_main
                if shortcut_logits is not None:
                    shortcut_ctc = True
                    shortcut_loss = self.loss_fn(
                        shortcut_logits,
                        batch.targets,
                        input_lengths,
                        batch.target_lengths,
                    )
                    loss_total = loss_main + (
                        linear_scale(self.cfg.epochs, current_epoch) * shortcut_loss
                    )
                    running_loss_main.update(loss_main)
                    running_loss_shortcut.update(shortcut_loss)
                running_loss.update(loss_total)

                # scale total loss only for backprop
                loss = loss_total / self.cfg.accum_steps

            # --- backward pass ---
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = (step + 1) % self.cfg.accum_steps == 0

            if do_step:
                dbg: dict[str, Any] = {}
                if self.debug_enabled:
                    dbg["epoch"] = current_epoch
                    dbg["step"] = step
                    dbg["T"] = T
                    dbg["B"] = B
                    dbg["blank_rate"] = blank_rate
                    dbg["pm_before"] = self._param_checksum()

                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                if self.debug_enabled:
                    dbg["grad_norm_before"] = self._global_grad_norm()

                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                    if self.debug_enabled:
                        dbg["grad_norm_after"] = self._global_grad_norm()
                else:
                    if self.debug_enabled:
                        dbg["grad_norm_after"] = None

                # --- optimizer/scaler step ---
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.debug_enabled:
                    pm_after = self._param_checksum()
                    dbg["pm_delta"] = pm_after - dbg["pm_before"]

                    grad_before = (
                        f"{dbg['grad_norm_before']:.3e}"
                        if dbg["grad_norm_before"] is not None
                        else "n/a"
                    )
                    if self.cfg.grad_clip_norm is None:
                        grad_after = "skip"
                    else:
                        grad_after = (
                            f"{dbg['grad_norm_after']:.3e}"
                            if dbg["grad_norm_after"] is not None
                            else "n/a"
                        )
                    blank_rate_str = (
                        f"{blank_rate:.3f}" if blank_rate is not None else "n/a"
                    )

                    if self.cfg.debug is DebugMode.PRINT:
                        tqdm.write(
                            f"[dbg] step={step} T={T} B={B} "
                            f"blank_rate={blank_rate_str} "
                            f"grad_norm_before_clip={grad_before} "
                            f"grad_norm_after_clip={grad_after} "
                            f"pm_delta={dbg['pm_delta']:+.2e}"
                        )

                    debug_records.append(
                        {
                            "epoch": dbg["epoch"],
                            "step": dbg["step"],
                            "T": dbg["T"],
                            "blank_rate": dbg["blank_rate"],
                            "grad_norm_before": dbg["grad_norm_before"],
                            "grad_norm_after": dbg["grad_norm_after"],
                            "pm_delta": dbg["pm_delta"],
                        }
                    )

                self.optimizer.zero_grad(set_to_none=True)

                # --- scheduler step
                if self.scheduler is not None and self.step_per_batch:
                    self.scheduler.step()

        if self.scheduler is not None and not self.step_per_batch:
            self.scheduler.step()

        metrics = {"loss": running_loss.mean_float()}
        if shortcut_ctc:
            metrics["loss_main"] = running_loss_main.mean_float()
            metrics["loss_shortcut"] = running_loss_shortcut.mean_float()

        return metrics, debug_records

    def _autocast(self) -> AbstractContextManager[Any]:
        if not self.use_autocast:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, enabled=True)

    @torch.no_grad()
    def _compute_blank_rate(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> float:
        pred = logits.argmax(dim=-1)  # [T,B]
        T, B = pred.shape
        t = torch.arange(T, device=pred.device).unsqueeze(1)  # [T,1]
        mask = t < input_lengths.unsqueeze(0)  # [T,B]
        blank = self.tokenizer.blank_index
        return (pred[mask] == blank).float().mean().item()

    @torch.no_grad()
    def _param_checksum(self) -> float:
        # cheap scalar fingerprint to detect sudden jumps/NaNs
        s = 0.0
        n = 0
        for p in self.model.parameters():
            if p.requires_grad:
                x = p.detach()
                # ignore tiny scalars; focus on larger tensors
                if x.numel() >= 1024:
                    s += x.float().mean().item()
                    n += 1
        return s / max(n, 1)

    @torch.no_grad()
    def _global_grad_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if g.is_sparse:
                g = g.coalesce().values()
            g = g.float()
            sq += g.pow(2).sum().item()
        return math.sqrt(sq)

    def evaluate(
        self, dataloader: DataLoader, compute_error_rates: bool = False
    ) -> dict[str, float]:
        self.model.eval()
        loss_meter = ScalarMeter(self.device)
        refs = []
        hyps = []

        iterator = (
            tqdm(dataloader, desc="Eval", leave=False, dynamic_ncols=True)
            if self.use_pbars
            else dataloader
        )

        with torch.inference_mode():
            for batch_id, batch in enumerate(iterator):
                batch = batch.to(self.device)
                input_lengths = self.model.output_lengths(batch.widths)

                logits, _ = self.model(batch.images)
                loss = self.loss_fn(
                    logits, batch.targets, input_lengths, batch.target_lengths
                )
                loss_meter.update(loss)

                if compute_error_rates:
                    if self.ctc_decoder_mode is CTCDecoderMode.BEAM:
                        decoded = beam_ctc_decode(
                            logits,
                            input_lengths,
                            self.beam_decoder,
                            self.tokenizer,
                            temperature=self.beam_temperature,
                        )
                    else:
                        decoded = greedy_ctc_decode(
                            logits, input_lengths, self.tokenizer
                        )
                    refs.extend(batch.texts)
                    hyps.extend(decoded)

                    if self.cfg.checkpoint.print_samples > 0 and batch_id == 0:
                        for i in range(
                            min(self.cfg.checkpoint.print_samples, len(decoded))
                        ):
                            tqdm.write(
                                f"[val sample {i}] pred: {decoded[i]!r} | gt: {batch.texts[i]!r}"
                            )

        out = {"loss": loss_meter.mean_float()}
        if compute_error_rates:
            out["cer"] = cer(refs, hyps)
            out["wer"] = wer(refs, hyps)
        return out

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metrics_history: list[dict] | None = None,
        on_epoch_start: Callable | None = None,
    ) -> tuple[torch.nn.Module, list[dict]]:
        if metrics_history is None:
            metrics_history = []

        epoch_iter = (
            tqdm(range(1, self.cfg.epochs + 1), desc="Epochs", dynamic_ncols=True)
            if self.use_pbars
            else range(1, self.cfg.epochs + 1)
        )

        for epoch in epoch_iter:
            if on_epoch_start:
                on_epoch_start(epoch, self.model)

            train_metrics, debug_records = self.train_one_epoch(epoch, train_loader)

            compute_error_rates = (
                self.cfg.checkpoint.compute_error_rates
                and epoch % self.cfg.checkpoint.full_eval_interval == 0
            )
            val_metrics = self.evaluate(val_loader, compute_error_rates)
            tqdm.write(
                format_metrics(epoch, train_metrics, val_metrics, self.optimizer)
            )

            if self.checkpoint_manager is not None:
                save_kwargs = {
                    "epoch": epoch,
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "scheduler": self.scheduler,
                    "scaler": self.scaler,
                    "val_metrics": val_metrics,
                }
                self.checkpoint_manager.maybe_save(**save_kwargs)
                self.checkpoint_manager.save_last(**save_kwargs)

            history_record: dict[str, Any] = {"epoch": epoch}
            history_record["lr"] = float(self.optimizer.param_groups[0].get("lr", 0.0))
            for k, v in train_metrics.items():
                history_record[f"train_{k}"] = float(v)
            for k, v in val_metrics.items():
                history_record[f"val_{k}"] = float(v)

            metrics_history.append(history_record)

            if self.checkpoint_manager is not None:
                save_dir = Path(self.checkpoint_manager.save_dir)
                _append_rows_to_csv(
                    save_dir / "metrics.csv",
                    list(history_record.keys()),
                    [history_record],
                )

                if self.cfg.debug is DebugMode.LOG and len(debug_records) > 0:
                    _append_rows_to_csv(
                        save_dir / "debug.csv",
                        list(debug_records[0].keys()),
                        debug_records,
                    )

        return self.model, metrics_history


def _append_rows_to_csv(
    path: str | Path, fieldnames: list[str], rows: list[dict]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
