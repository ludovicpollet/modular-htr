import PIL.Image
from torchvision.transforms.functional import to_tensor

FIXED_HEIGHT = 128

def resize_keep_aspect(pil_img):
    w, h = pil_img.size
    new_h = FIXED_HEIGHT
    new_w = int(round(w * new_h / h))
    return pil_img.resize((new_w, new_h), resample=PIL.Image.BILINEAR)


def basic_image_transforms(batch):
    images = []
    widths = []
    for img in batch["image"]:
        img = resize_keep_aspect(img)
        widths.append(img.size[0])
        images.append(to_tensor(img))
    batch["image"] = images
    batch["width"] = widths
    return batch
