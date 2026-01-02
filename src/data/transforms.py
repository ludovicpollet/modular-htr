from typing import Callable

import kornia.augmentation as K
import PIL.Image
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor


class GPUAugmentation:
    """
    This pipeline includes the expensive elastic transform.
    Runs on GPU to prevent slowing things down too much.
    """
    def __init__(self, device: torch.device):
        self.module = K.AugmentationSequential(
            K.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                shear=3,
                p=0.8,
            ),
            K.RandomElasticTransform(
                kernel_size=(19, 19),
                sigma=(5.0, 5.0),
                alpha=(0.2, 0.8),
                p=0.2,
            ),
            K.RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.1),
            K.RandomBrightness((0.8, 1.2), p=0.4),
            K.RandomContrast((0.6, 1.0), p=0.4),
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


def make_basic_image_transform(fixed_height: int, augment: bool = False) -> Callable:
    """
    Returns a callable suitable for datasets.Dataset.with_transform.
    Optionnaly includes the lighter transforms that can run on CPU.
    The caller must not set augment=True if the GPU augmentation pipeline is in use.
    """
    aug = None
    if augment:
        aug = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=(-5.0, 5.0),
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                    shear=(-3, 3),
                ),
                # v2.RandomApply([v2.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.4),
                v2.RandomPerspective(distortion_scale=0.1, p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    def _transform(batch):
        images = []
        widths = []
        for img in batch["image"]:
            img = resize_keep_aspect(img, fixed_height)
            if aug is not None:
                img = aug(img)

            widths.append(img.size[0])
            images.append(to_tensor(img))
        batch["image"] = images
        batch["width"] = widths
        return batch

    return _transform
