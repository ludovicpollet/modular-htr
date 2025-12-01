import datetime
import json
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(x, device):
    """Recursively move tensors or tuples/dicts of tensors to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def set_all_seeds(seed=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    torch.set_float32_matmul_precision("high")


def format_metrics(epoch, train_metrics, val_metrics, optimizer):
    parts = [f"Epoch {epoch:03d}"]
    parts.append(f"train_loss={train_metrics['loss']:.4f}")
    parts.append(f"val_loss={val_metrics['loss']:.4f}")

    if "cer" in val_metrics:
        parts.append(f"val_cer={val_metrics['cer']:.3f}")
    if "wer" in val_metrics:
        parts.append(f"val_wer={val_metrics['wer']:.3f}")

    parts.append(f"lr={optimizer.param_groups[0]['lr']:.3e}")

    return " | ".join(parts)


def create_run_dir(
    base_dir: str | pathlib.Path = "runs", run_name: str = "experiment"
) -> pathlib.Path:
    base_dir = pathlib.Path(base_dir)
    base_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / pathlib.Path(f"{timestamp}_{run_name}")
    run_dir.mkdir(exist_ok=False)

    return run_dir


def dump_config(run_dir: pathlib.Path, run_config: dict) -> None:
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)


@dataclass(slots=True)
class CheckpointInfo:
    path: pathlib.Path
    score: float
    epoch: int


@dataclass(slots=True)
class CheckpointManager:
    save_dir: str | pathlib.Path
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    top_k: int = 3
    best_checkpoints: list[CheckpointInfo] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.save_dir, str):
            self.save_dir = pathlib.Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _is_better(self, score: float, ref: float) -> bool:
        return score < ref if self.mode == "min" else score > ref

    def _build_state(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler,
        scaler,
        val_metrics,
        config,
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": dict(val_metrics),
            "config": config,
        }

    def maybe_save(
        self, epoch, model, optimizer, scheduler, scaler, val_metrics, config
    ) -> pathlib.Path | None:
        """Save checkpoint if epoch in top_k, return a path only if checkpoint saved"""
        if self.monitor not in val_metrics:
            return None

        score = float(val_metrics[self.monitor])

        should_save = False
        if len(self.best_checkpoints) < self.top_k:
            should_save = True
        else:
            if self.mode == "min":
                worst = max(self.best_checkpoints, key=lambda c: c.score)
            else:
                worst = min(self.best_checkpoints, key=lambda c: c.score)
            should_save = self._is_better(score, worst.score)
        if not should_save:
            return None

        path = self.save_dir / pathlib.Path(
            f"best_{epoch:03d}_{self.monitor}_{score:.4f}.pt"
        )
        state = self._build_state(
            epoch, model, optimizer, scheduler, scaler, val_metrics, config
        )
        torch.save(state, path)

        self.best_checkpoints.append(
            CheckpointInfo(path=path, score=score, epoch=epoch)
        )
        self.best_checkpoints.sort(key=lambda c: c.score, reverse=self.mode == "max")
        if len(self.best_checkpoints) > self.top_k:
            to_remove = self.best_checkpoints.pop(-1)
            to_remove.path.unlink(missing_ok=True)

        return path

    def save_last(
        self, epoch, model, optimizer, scheduler, scaler, val_metrics, config
    ) -> pathlib.Path:
        """Save last.pt checkpoint anyway"""
        path = self.save_dir / pathlib.Path("last.pt")
        state = self._build_state(
            epoch, model, optimizer, scheduler, scaler, val_metrics, config
        )
        torch.save(state, path)
        return path
