from pathlib import Path

import numpy as np
import PIL.Image
from lxml import etree

ALLOWED_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def parse_xml(xml_path):
    tree = etree.parse(xml_path)
    namespace = tree.getroot().tag.split("}")[0] + "}"
    textlines = tree.findall(f".//{namespace}TextLine")

    for line_id, textline in enumerate(textlines):
        text_container = textline.find(f"./{namespace}TextEquiv/{namespace}Unicode")
        if text_container is None:
            continue
        if text_container.text is None:
            continue
        text = text_container.text
        coords_container = textline.find(f"./{namespace}Coords")
        if coords_container is None:
            continue
        coords = np.array(
            [
                [int(x) for x in coords.split(",")]
                for coords in coords_container.attrib["points"].split()
            ],
            np.int32,
        )
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        bbox = (xmin, ymin, xmax, ymax)  # must be consistent with what PIL expects
        yield {"line_id": line_id, "text": text, "bbox": bbox}


def find_matching_image(xml_path, img_root):
    xml_path = Path(xml_path)
    img_root = Path(img_root)
    stem = xml_path.stem
    candidates = [
        p for p in img_root.rglob(f"{stem}.*") if p.suffix.lower() in ALLOWED_EXTS
    ]

    if len(candidates) == 0:
        raise FileNotFoundError(f"No image found for {xml_path} (stem={stem})")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple images found for {xml_path}: {candidates}")

    return candidates[0]


def stream_examples_from_xml(xml_root, img_root, convert_L=True):
    xml_root, img_root = Path(xml_root), Path(img_root)
    for xml_path in xml_root.rglob("*.xml"):
        if xml_path.stem == "mets":
            continue

        img_path = find_matching_image(xml_path, img_root)

        with PIL.Image.open(img_path) as page_img:
            for line in parse_xml(xml_path):
                line_img = page_img.crop(line["bbox"])
                if convert_L:
                    line_img = line_img.convert("L")
                line_id = f"{xml_path.stem}_{line['line_id']:03d}"

                yield {"id": line_id, "image": line_img, "text": line["text"]}


