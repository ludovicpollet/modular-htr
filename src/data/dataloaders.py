import math
import random
import os

import datasets
import torch

from .transforms import make_basic_image_transform


def ctc_collate(batch):
    if not batch:
        raise ValueError("Empty batch passed to collate")

    batch = sorted(batch, key=lambda x: x["width"], reverse=True)

    images = [b["image"] for b in batch]
    labels_raw = [b["labels"] for b in batch]
    widths = [img.shape[-1] for img in images]

    B = len(batch)
    C, H = images[0].shape[0], images[0].shape[1]
    max_W = max(img.shape[-1] for img in images)

    images_padded = images[0].new_zeros((B, C, H, max_W))
    for i, img in enumerate(images):
        W_i = img.shape[-1]
        images_padded[i, :, :, :W_i] = img

    target_lengths = []
    pieces = []
    for lab in labels_raw:
        lab_tensor = torch.as_tensor(lab, dtype=torch.long)
        T_i = lab_tensor.shape[0]
        target_lengths.append(T_i)
        if lab_tensor.numel() > 0:
            pieces.append(lab_tensor)
    if pieces:
        targets = torch.cat(pieces, dim=0)
    else:
        targets = torch.empty((0,), dtype=torch.long)

    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    widths_tensor = torch.as_tensor(widths, dtype=torch.long)

    batch_out = {
        "images": images_padded,  # [B, 1, H, W_max]
        "targets": targets,  # [sum(target_lengths)]
        "target_lengths": target_lengths,
        "widths": widths_tensor,
        "ids": [b.get("id") for b in batch],
        "texts": [b.get("text", "") for b in batch],
    }
    return batch_out


class BucketByWidthSampler(torch.utils.data.Sampler):
    def __init__(self, widths, batch_size, shuffle=True):
        self.widths = list(widths)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = list(range(len(self.widths)))
        self.indices.sort(key=lambda i: self.widths[i])

    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            chunk_size = self.batch_size * 20
            chunks = [
                indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
            ]
            random.shuffle(chunks)
            indices = [i for chunk in chunks for i in chunk]

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)


def _compute_resized_width(batch, fixed_height):
    widths = []
    for img in batch["image"]:
        w, h = img.size
        widths.append(int(round(w * fixed_height / h)))
    return {"width": widths}


def _ensure_widths(ds, fixed_height, num_proc=None):
    if "width" in ds["train"].column_names and "width" in ds["test"].column_names:
        return ds
    num_proc = num_proc or min(os.cpu_count() or 1, 16)
    return datasets.DatasetDict(
        {
            "train": ds["train"].map(
                _compute_resized_width,
                fn_kwargs={"fixed_height": fixed_height},
                batched=True,
                batch_size=1024,
                num_proc=num_proc,
            ),
            "test": ds["test"].map(
                _compute_resized_width,
                fn_kwargs={"fixed_height": fixed_height},
                batched=True,
                batch_size=1024,
                num_proc=num_proc,
            ),
        }
    )


def _bucket_widths(widths, bin_size: int | None):
    """
    Helper to round up bucketing widths to nearest bin_size multiple.
    Used to reduce number of unique widths when using cuDNN benchmarking.
    """
    if not bin_size:
        return list(widths)
    return [math.ceil(w / bin_size) * bin_size for w in widths]


def make_dataloaders(
    ds: datasets.DatasetDict,
    fixed_height: int = 96,
    batch_size: int = 32,
    num_workers: int = 16,
    use_bucketing: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """High level convenience to build DataLoaders from a HF DatasetDict"""
    if not isinstance(ds, datasets.DatasetDict):
        raise TypeError(
            "make_dataloaders expects a datasets.DatasetDict with train/test splits."
        )
    if "train" not in ds or "test" not in ds:
        raise KeyError("DatasetDict must contain at least 'train' and 'test' splits.")

    ds = _ensure_widths(ds, fixed_height=fixed_height, num_proc=num_workers)
    bucket_bin = 16 if torch.backends.cudnn.benchmark else None
    widths_train = _bucket_widths(ds["train"]["width"], bin_size=bucket_bin)

    train_transform = make_basic_image_transform(
        fixed_height=fixed_height, augment=True
    )
    test_transform = make_basic_image_transform(
        fixed_height=fixed_height, augment=False
    )
    ds = datasets.DatasetDict(
        {
            "train": ds["train"].with_transform(train_transform),
            "test": ds["test"].with_transform(test_transform),
        }
    )

    persistent_workers = num_workers > 0

    loader_kwargs = dict(
        collate_fn=ctc_collate,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    if persistent_workers and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    if use_bucketing:
        train_batch_sampler = BucketByWidthSampler(
            widths=widths_train,
            batch_size=batch_size,
            shuffle=True,
        )
        train_loader = torch.utils.data.DataLoader(
            ds["train"],
            batch_sampler=train_batch_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            ds["train"],
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    test_loader = torch.utils.data.DataLoader(
        ds["test"],
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader
