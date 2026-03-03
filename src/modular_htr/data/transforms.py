from typing import Callable

import kornia.augmentation as K
import PIL.Image
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor

from .tokenization import CharTokenizer, normalize_text


class GPUAugmentation:
    """
    This pipeline includes the expensive elastic transform.
    Runs on GPU to prevent slowing things down too much.
    """

    def __init__(self, device: torch.device):
        self.module = K.AugmentationSequential(
            K.RandomAffine(
                degrees=1,
                translate=(0.02, 0.02),
                scale=(0.6, 1.2),
                shear=(-30, 30, -5, 5),  # type: ignore (tested, works)
                p=0.5,
            ),
            K.RandomElasticTransform(
                kernel_size=(49, 49),
                sigma=(13.0, 13.0),
                alpha=(0.6, 0.6),
                p=0.25,
            ),
            K.RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.1),
            K.RandomBrightness((0.8, 1.2), p=0.5),
            K.RandomContrast((0.6, 1.0), p=0.4),
            K.RandomGamma((0.8, 1.2), p=0.5),
            data_keys=["input"],
            same_on_batch=False,
        ).to(device)
        self.module.train()

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.module(images)


def resize_keep_aspect(pil_img: PIL.Image.Image, fixed_height: int) -> PIL.Image.Image:
    w, h = pil_img.size
    new_h = fixed_height
    new_w = int(round(w * new_h / h))
    return pil_img.resize((new_w, new_h), resample=PIL.Image.Resampling.HAMMING)


def make_preprocessing_fn(
    fixed_height: int,
    tokenizer: CharTokenizer,
    text_col: str = "text",
    image_col: str = "image",
) -> Callable:
    """
    One-time preprocessing pipeline to avoid repeated maps. Cached by HuggingFace datasets.
    (load) -> (to grayscale) -> resize -> get new width -> tokenize
    """

    def _preprocess(example):
        img = example[image_col]
        # handle lazy-loading images
        if not isinstance(img, PIL.Image.Image):
            img = PIL.Image.open(img)
        # convert to grayscale
        if img.mode != "L":
            img = img.convert("L")
        # resize
        img = resize_keep_aspect(img, fixed_height)

        # normalize text to match the tokenizer's alphabet, then tokenize
        text = normalize_text(
            example[text_col],
            tokenizer.unicode_form,
            tokenizer.strip_space_before_punctuation,
        )
        labels = tokenizer.encode(text)

        return {
            "image": img,
            "width": img.size[0],
            "labels": labels,
            "label_length": len(labels),
            "text": text,
        }

    return _preprocess


class RuntimeTransform:
    """Per-batch transform applied by HF ``with_transform()``.

    Converts PIL images to tensors, with optional CPU augmentation and
    pixel inversion.  Picklable, so it works with forkserver/spawn workers.
    """

    def __init__(self, augment: bool = False, invert: bool = False):
        self.invert = invert
        self.aug: v2.Compose | None = None
        if augment:
            self.aug = v2.Compose(
                [
                    v2.RandomAffine(
                        degrees=(-5.0, 5.0),
                        translate=(0.02, 0.02),
                        scale=(0.95, 1.05),
                        shear=(-3, 3),
                    ),
                    v2.RandomPerspective(distortion_scale=0.1, p=0.2),
                    v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3),
                    v2.ColorJitter(brightness=0.2, contrast=0.2),
                ]
            )

    def __call__(self, batch):
        images = []
        for img in batch["image"]:
            if self.aug is not None:
                img = self.aug(img)
            t = to_tensor(img)
            if self.invert:
                t = 1.0 - t
            images.append(t)
        batch["image"] = images
        return batch


def make_runtime_transform(augment: bool = False, invert: bool = False) -> RuntimeTransform:
    """Return a picklable per-batch transform (to_tensor + optional augmentation/inversion).

    The caller must not set augment=True if the GPU augmentation pipeline is in use.
    If invert is True, pixel values are flipped (1.0 - t) so strokes become bright
    and background dark.
    """
    return RuntimeTransform(augment=augment, invert=invert)
