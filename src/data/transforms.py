from typing import Callable

import PIL.Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor


def resize_keep_aspect(pil_img: PIL.Image.Image, fixed_height: int) -> PIL.Image.Image:
    w, h = pil_img.size
    new_h = fixed_height
    new_w = int(round(w * new_h / h))
    return pil_img.resize((new_w, new_h), resample=PIL.Image.BILINEAR)


def make_basic_image_transform(fixed_height: int, augment: bool = False) -> Callable:
    """returns a callable suitable for datasets.Dataset.with_transform"""
    if augment:
        aug = T.Compose(
            [
                T.RandomAffine(
                    degrees=2.0,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                    shear=(-2, 2),
                ),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            ]
        )
    else:
        aug = None

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
