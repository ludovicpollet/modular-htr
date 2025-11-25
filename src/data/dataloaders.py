import math
import random

import datasets
import torch

from .tokenization import CharTokenizer
from .transforms import make_basic_image_transform


def ctc_collate(batch, pad_index):
    if not batch:
        raise ValueError("Empty batch passed to collate")

    batch = sorted(batch, key=lambda x: x["width"], reverse=True)

    images = [b["image"] for b in batch]
    labels_raw = [b["labels"] for b in batch]
    widths = [b["width"] for b in batch]

    B = len(batch)
    C, H = images[0].shape[0], images[0].shape[1]
    max_W = max(img.shape[-1] for img in images)
    max_T = max(len(lab) for lab in labels_raw)

    images_padded = images[0].new_zeros((B, C, H, max_W))
    for i, img in enumerate(images):
        W_i = img.shape[-1]
        images_padded[i, :, :, :W_i] = img

    labels_padded = torch.full((B, max_T), pad_index, dtype=torch.long)
    label_lengths = []
    for i, lab in enumerate(labels_raw):
        lab_tensor = torch.as_tensor(lab, dtype=torch.long)
        T_i = lab_tensor.shape[0]
        labels_padded[i, :T_i] = lab_tensor
        label_lengths.append(T_i)

    batch_out = {
        "images": images_padded,  # [B, 1, H, W_max]
        "labels": labels_padded,  # [B, T_max]
        "label_lengths": torch.as_tensor(label_lengths, dtype=torch.long),
        "widths": torch.as_tensor(widths, dtype=torch.long),
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


def make_dataloaders(
    ds: datasets.DatasetDict,
    tokenizer: CharTokenizer,
    fixed_height: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    use_bucketing: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """High level convenience to build DataLoaders from a HF DatasetDict"""
    if not isinstance(ds, datasets.DatasetDict):
        raise TypeError(
            "make_dataloaders expects a datasets.DatasetDict with train/test splits."
        )
    if "train" not in ds or "test" not in ds:
        raise KeyError("DatasetDict must contain at least 'train' and 'test' splits.")

    image_transform = make_basic_image_transform(fixed_height=fixed_height)
    ds = ds.with_transform(image_transform)

    if use_bucketing:
        widths_train = [ex["width"] for ex in ds["train"]]
        train_batch_sampler = BucketByWidthSampler(
            widths=widths_train,
            batch_size=batch_size,
            shuffle=True,
        )
        train_loader = torch.utils.data.DataLoader(
            ds["train"],
            batch_sampler=train_batch_sampler,
            collate_fn=lambda b: ctc_collate(b, pad_index=tokenizer.pad_index),
            num_workers=num_workers,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            ds["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: ctc_collate(b, pad_index=tokenizer.pad_index),
            num_workers=num_workers,
        )

    test_loader = torch.utils.data.DataLoader(
        ds["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: ctc_collate(b, pad_index=tokenizer.pad_index),
        num_workers=num_workers,
    )
    return train_loader, test_loader
