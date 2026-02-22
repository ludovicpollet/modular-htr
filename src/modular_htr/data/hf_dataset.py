import datasets

from .xml_helpers import stream_examples_from_xml


def build_hf_dataset(xml_root, img_root, out_dir):
    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "image": datasets.Image(),
            "text": datasets.Value("string"),
        }
    )
    ds = datasets.Dataset.from_generator(
        lambda: stream_examples_from_xml(
            xml_root, img_root
        ),  # wrap in lambda to use parameters
        features=features,
    )
    ds = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore
    ds.save_to_disk(out_dir)
    return ds


def add_width_column(ds: datasets.Dataset) -> datasets.Dataset:
    def _get_width(example):
        w, h = example["image"].size
        return {"width": w}

    return ds.map(_get_width)
