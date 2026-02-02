import datetime
import json
import pathlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src import types
from src import config
from src.data.tokenization import CharTokenizer


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


def configure_torch(benchmark: bool = True) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_dataset(cfg: config.Dataset) -> Dataset | DatasetDict:
    """Utility to load a dataset from disk or hub."""
    if isinstance(cfg, config.LocalDataset):
        return load_from_disk(cfg.path)
    if isinstance(cfg, config.HubDataset):
        # casting to avoid the iterable return types variant
        # to do later, maybe support streaming
        return cast(Dataset | DatasetDict, load_dataset(cfg.name, streaming=False))


def make_log_fn(use_pbar: bool = True) -> Callable[[str], None]:
    return tqdm.write if use_pbar else print


def format_metrics(epoch, train_metrics, val_metrics, optimizer):
    parts = [f"Epoch {epoch:03d}"]
    parts.append(f"train_loss={train_metrics['loss']:.4f}")
    if "loss_main" in train_metrics:
        parts.append(f"train_loss_main={train_metrics['loss_main']:.4f}")
    if "loss_shortcut" in train_metrics:
        parts.append(f"train_loss_shortcut={train_metrics['loss_shortcut']:.4f}")

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


def serialize_optimizer_config(optimizer) -> dict[str, Any]:
    """
    Legacy helper. Should not be used anymore.
    Serialize the live optimizer configuration only for the human readable run config dump: it is not used to restore states.
    """
    raise NotImplementedError(
        "Deprecated function serialize_optimizer_config() is only kept for reference"
    )
    opt_config = {
        "name": optimizer.__class__.__name__,
        "param_groups": [
            {
                "lr": pg.get("lr", optimizer.defaults.get("lr")),
                "weight_decay": pg.get(
                    "weight_decay", optimizer.defaults.get("weight_decay")
                ),
                "betas": list(
                    pg.get("betas", optimizer.defaults.get("betas", (None, None)))
                ),
                "eps": pg.get("eps", optimizer.defaults.get("eps")),
                "fused": pg.get("fused", optimizer.defaults.get("fused")),
            }
            for pg in optimizer.param_groups
        ],
    }
    return opt_config


def dump_config(run_dir: pathlib.Path, run_config: config.Config) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot save config file. Run directory does not exist: {run_dir}"
        )
    path = run_dir / "config.json"
    path.write_text(json.dumps(asdict(run_config), indent=2, default=str))


def linear_scale(total_epochs: int, current_epoch: int) -> float:
    """A simple linear scale for the shortcut loss"""
    start = 1
    end = 0.1
    return start + (end - start) * (current_epoch / total_epochs)


class ScalarMeter:
    """Helper to accumulate tensors on GPU and avoid per-step .item() sync issues."""

    def __init__(self, device: torch.device):
        self.device = device
        self.sum = torch.zeros((), device=device)
        self.count = 0

    def update(self, value: torch.Tensor, n: int = 1) -> None:
        """Updates value (scalar Tensor) with .detach() to avoid autograd and sync"""
        self.sum += value.detach()
        self.count += n

    def mean_tensor(self) -> torch.Tensor:
        return self.sum / max(1, self.count)

    def mean_float(self) -> float:
        """Only to be called rarely."""
        return float(self.mean_tensor().item())


@dataclass(slots=True)
class CheckpointInfo:
    path: pathlib.Path
    score: float
    epoch: int


@dataclass(slots=True)
class CheckpointManager:
    save_dir: str | pathlib.Path
    monitor: types.MonitorMetric | str = types.MonitorMetric.VAL_LOSS
    mode: types.MetricMode | str = types.MetricMode.MIN
    top_k: int = 3
    best_checkpoints: list[CheckpointInfo] = field(default_factory=list)
    config: dict[str, Any] | None = None  # for backwards compatibility
    model_config: dict[str, Any] | None = None
    tokenizer: CharTokenizer | None = None

    def __post_init__(self):
        if isinstance(self.save_dir, str):
            self.save_dir = pathlib.Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = types.MonitorMetric(self.monitor).value
        self.mode = types.MetricMode(self.mode).value

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
    ) -> dict[str, Any]:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": dict(val_metrics),
            "model_config": self.model_config or (self.config or {}).get("model"),
        }
        if self.tokenizer is not None:
            state["tokenizer"] = self.tokenizer.to_dict()
        return state

    def maybe_save(
        self, epoch, model, optimizer, scheduler, scaler, val_metrics
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
            epoch, model, optimizer, scheduler, scaler, val_metrics
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
        self, epoch, model, optimizer, scheduler, scaler, val_metrics
    ) -> pathlib.Path:
        """Save last.pt checkpoint anyway"""
        path = self.save_dir / pathlib.Path("last.pt")
        state = self._build_state(
            epoch, model, optimizer, scheduler, scaler, val_metrics
        )
        torch.save(state, path)
        return path


def log_model_info(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = (
        f"{model.__class__.__name__}: "
        f"trainable={trainable / 1e6:.2f}M / total={total / 1e6:.2f}M"
    )
    tqdm.write(msg)
