import PIL.Image
from torchvision.transforms.functional import to_tensor

def resize_keep_aspect(pil_img: PIL.Image.Image, fixed_height: int) -> PIL.Image.Image:
    w, h = pil_img.size
    new_h = fixed_height
    new_w = int(round(w * new_h / h))
    return pil_img.resize((new_w, new_h), resample=PIL.Image.BILINEAR)


def make_basic_image_transform(fixed_height):
    """returns a callable suitable for datasets.Dataset.with_transform"""
    def _transform(batch):
        images = []
        widths = []
        for img in batch["image"]:
            img = resize_keep_aspect(img, fixed_height)
            widths.append(img.size[0])
            images.append(to_tensor(img))
        batch["image"] = images
        batch["width"] = widths
        return batch
    return _transform
