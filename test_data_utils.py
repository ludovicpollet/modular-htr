# test_data_utils.py
import argparse
from pathlib import Path

import datasets

from data_utils import build_hf_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build and inspect an HF line dataset from PAGE XML."
    )
    parser.add_argument(
        "--xml_root",
        type=str,
        required=True,
        help="Root folder containing PAGE-XML files.",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        required=True,
        help="Root folder containing page images.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to save/load the HuggingFace dataset.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the dataset even if out_dir already exists.",
    )
    args = parser.parse_args()

    xml_root = Path(args.xml_root)
    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and not args.rebuild:
        print(f"Loading existing dataset from {out_dir} ...")
        ds = datasets.load_from_disk(str(out_dir))
    else:
        print(f"Building dataset from XML={xml_root} and IMG={img_root} ...")
        ds = build_hf_dataset(xml_root, img_root, out_dir)

    print("\n=== Dataset structure ===")
    print(ds)
    print("\nSplits:", list(ds.keys()))

    train_ds = ds["train"]

    print(f"\nNumber of training examples: {len(train_ds)}")

    # Show a few examples
    n_show = min(3, len(train_ds))
    print(f"\n=== First {n_show} examples ===")
    for i in range(n_show):
        ex = train_ds[i]
        print(f"\nExample {i}:")
        print(f"  id  : {ex['id']}")
        print(f"  text: {ex['text']!r}")
        print(f"  image type: {type(ex['image'])}")

        # If you want to visually inspect the line image:
        # (this will open the default image viewer)
        ex["image"].show(title=f"Example {i} - {ex['id']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
