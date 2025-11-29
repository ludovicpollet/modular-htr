import random

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
