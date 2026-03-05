import datetime
import json
import logging
import pathlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import torch
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from modular_htr import types
from modular_htr import config
from modular_htr.data.tokenization import CharTokenizer

logger = logging.getLogger(__name__)


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


def normalize_splits(ds: DatasetDict) -> DatasetDict:
    """Map HF split names to canonical 'train' / 'val'.

    For validation, prefers 'validation' over 'test' when both exist.
    Returns only recognized splits; unknown names (e.g. 'train_clean') are dropped.
    The result may contain only 'train' if no val-like split is found.
    """
    if "train" not in ds:
        raise ValueError(f"No 'train' split found. Available: {list(ds.keys())}")

    result = {"train": ds["train"]}

    # Priority: validation > val > test
    for candidate in ("validation", "val", "test"):
        if candidate in ds:
            result["val"] = ds[candidate]
            break

    return DatasetDict(result)  # type: ignore[call-overload]


def get_single_dataset(
    cfg: config.LocalDataset | config.HubDataset,
) -> DatasetDict:
    """Load a single local or hub dataset without split normalization."""
    if isinstance(cfg, config.LocalDataset):
        ds = load_from_disk(cfg.path)
    elif isinstance(cfg, config.HubDataset):
        # casting to avoid the iterable return types variant
        # to do later, maybe support streaming
        ds = cast(DatasetDict, load_dataset(cfg.name, streaming=False))
    else:
        raise ValueError(f"Unknown dataset config type: {type(cfg)}")

    if isinstance(ds, Dataset):
        # Bare dataset without splits — treat as train-only.
        ds = DatasetDict({"train": ds})

    return ds


def _parse_sources(path: pathlib.Path) -> list[config.DatasetSource]:
    """Read a JSON file and return a list of DatasetSource objects."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw).__name__}")
    return [config.DatasetSource(**item) for item in raw]


def _resolve_val_split(
    ds: DatasetDict, source_name: str, val_split: str | None
) -> str | None:
    """Return the concrete validation split name, or None if unavailable.

    "auto" tries validation > val > test in order. None means explicitly no
    validation. An explicit name must exist or a ValueError is raised.
    """
    if val_split is None:
        return None
    if val_split == "auto":
        for candidate in ("validation", "val", "test"):
            if candidate in ds:
                return candidate
        return None
    if val_split not in ds:
        raise ValueError(
            f"Source '{source_name}': requested val_split '{val_split}' "
            f"not found. Available: {list(ds.keys())}"
        )
    return val_split


def _normalize_columns(
    ds: Dataset, img_col: str, text_col: str, source_name: str
) -> Dataset:
    """Rename columns to canonical names, add source provenance, drop extras."""
    # Drop non-essential columns first to avoid rename conflicts.
    keep = {img_col, text_col}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    if img_col != "image":
        ds = ds.rename_column(img_col, "image")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    ds = ds.add_column("source", [source_name] * len(ds))
    return ds


def _get_multi_dataset(cfg: config.MultiDataset) -> DatasetDict:
    """Load and concatenate multiple HF Hub datasets into a single DatasetDict."""
    sources = _parse_sources(cfg.sources_file)

    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []

    for src in sources:
        logger.info("Loading %s ...", src.name)
        ds = cast(DatasetDict, load_dataset(src.name, streaming=False))
        if isinstance(ds, Dataset):
            ds = DatasetDict({"train": ds})

        # Resolve splits.
        val_split_name = _resolve_val_split(ds, src.name, src.val_split)

        if src.train_split is not None:
            if src.train_split not in ds:
                raise ValueError(
                    f"Source '{src.name}': train_split '{src.train_split}' "
                    f"not found. Available: {list(ds.keys())}"
                )
            part = _normalize_columns(
                ds[src.train_split], src.img_col, src.text_col, src.name
            )
            train_parts.append(part)
            logger.info("  train: %d samples", len(part))

        if val_split_name is not None:
            part = _normalize_columns(
                ds[val_split_name], src.img_col, src.text_col, src.name
            )
            val_parts.append(part)
            logger.info("  val (%s): %d samples", val_split_name, len(part))

    if not train_parts:
        raise ValueError(
            "No training data: none of the sources contributed a train split."
        )

    result: dict[str, Dataset] = {"train": concatenate_datasets(train_parts)}
    logger.info(
        "Combined train: %d samples from %d sources",
        len(result["train"]),
        len(train_parts),
    )

    if val_parts:
        result["val"] = concatenate_datasets(val_parts)
        logger.info(
            "Combined val: %d samples from %d sources",
            len(result["val"]),
            len(val_parts),
        )

    return DatasetDict(result)  # type: ignore[call-overload]


def get_dataset(cfg: config.Dataset) -> DatasetDict:
    """Load a dataset from disk or hub and normalize split names."""
    if isinstance(cfg, config.MultiDataset):
        return _get_multi_dataset(cfg)

    if isinstance(cfg, (config.LocalDataset, config.HubDataset)):
        return normalize_splits(get_single_dataset(cfg))

    raise ValueError(f"Unknown dataset config type: {type(cfg)}")


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
        # Strip torch.compile "_orig_mod." prefix so checkpoints are portable
        raw_sd = model.state_dict()
        clean_sd = {k.replace("._orig_mod.", "."): v for k, v in raw_sd.items()}
        state = {
            "epoch": epoch,
            "model_state_dict": clean_sd,
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
    logger.info(
        "%s: trainable=%.2fM / total=%.2fM",
        model.__class__.__name__,
        trainable / 1e6,
        total / 1e6,
    )
