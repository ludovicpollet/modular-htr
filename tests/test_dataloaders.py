import math

import PIL.Image
import pytest
import torch
from torchvision.transforms.functional import to_tensor

from modular_htr.data.dataloaders import Batch, BucketByWidthSampler, ctc_collate
from modular_htr.data.transforms import make_runtime_transform


def _make_pil_batch(widths=(50, 70), height=32, color=128):
    """Create a batch dict with grayscale PIL images, mimicking HF with_transform input."""
    images = [PIL.Image.new("L", (w, height), color=color) for w in widths]
    return {
        "image": images,
        "labels": [[1, 2, 3]] * len(images),
        "width": list(widths),
        "text": ["abc"] * len(images),
        "id": [str(i) for i in range(len(images))],
    }


def _make_collate_example(width, height=32, labels=None):
    """Create a single example dict as ctc_collate expects (post-transform)."""
    if labels is None:
        labels = [1, 2, 3]
    return {
        "image": torch.rand(1, height, width),
        "labels": labels,
        "width": width,
        "text": "abc",
        "id": f"w{width}",
    }


class TestMakeRuntimeTransform:
    def test_basic_no_augment_no_invert(self):
        transform = make_runtime_transform(augment=False, invert=False)
        batch = _make_pil_batch(widths=(50, 70), height=32, color=128)
        originals = [to_tensor(img) for img in batch["image"]]

        result = transform(batch)

        assert len(result["image"]) == 2
        for tensor, expected in zip(result["image"], originals):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == expected.shape  # [1, H, W]
            assert torch.allclose(tensor, expected)

    def test_invert(self):
        transform = make_runtime_transform(augment=False, invert=True)
        batch = _make_pil_batch(widths=(50,), height=32, color=128)
        original = to_tensor(batch["image"][0])

        result = transform(batch)

        tensor = result["image"][0]
        assert torch.allclose(tensor, 1.0 - original)

    def test_augment_preserves_shape_and_range(self):
        transform = make_runtime_transform(augment=True, invert=False)
        batch = _make_pil_batch(widths=(50, 70), height=32, color=128)

        result = transform(batch)

        for tensor in result["image"]:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 3
            assert tensor.shape[0] == 1  # single channel
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0

    def test_picklable(self):
        """RuntimeTransform must survive pickle for forkserver workers."""
        import pickle

        for augment in (False, True):
            for invert in (False, True):
                transform = make_runtime_transform(augment=augment, invert=invert)
                restored = pickle.loads(pickle.dumps(transform))

                batch = _make_pil_batch(widths=(50,), height=32, color=128)
                batch_copy = _make_pil_batch(widths=(50,), height=32, color=128)

                # Seed torch for reproducibility of augmentation.
                torch.manual_seed(42)
                result_original = transform(batch)
                torch.manual_seed(42)
                result_restored = restored(batch_copy)

                assert torch.allclose(
                    result_original["image"][0], result_restored["image"][0]
                )


class TestCtcCollate:
    def test_padding_and_sorting(self):
        examples = [
            _make_collate_example(width=30, labels=[1, 2]),
            _make_collate_example(width=50, labels=[3, 4, 5]),
            _make_collate_example(width=40, labels=[6]),
        ]
        batch = ctc_collate(examples)

        assert isinstance(batch, Batch)
        # Padded to widest (50), sorted descending by width.
        assert batch.images.shape == (3, 1, 32, 50)
        assert batch.widths.tolist() == [50, 40, 30]

        # Padding region is zeros.
        assert (batch.images[1, :, :, 40:] == 0).all()
        assert (batch.images[2, :, :, 30:] == 0).all()

        # Targets concatenated in sorted (descending width) order.
        assert batch.targets.tolist() == [3, 4, 5, 6, 1, 2]
        assert batch.target_lengths.tolist() == [3, 1, 2]

    def test_fixed_width_padding_preserves_content_widths(self):
        examples = [
            _make_collate_example(width=30, labels=[1, 2]),
            _make_collate_example(width=50, labels=[3, 4, 5]),
            _make_collate_example(width=40, labels=[6]),
        ]
        batch = ctc_collate(examples, fixed_width=100)

        # Padded to fixed_width, not to max content width (50).
        assert batch.images.shape == (3, 1, 32, 100)

        # Content widths preserved (sorted descending), not the padded width.
        assert batch.widths.tolist() == [50, 40, 30]

        # Padding region beyond content width is zeros.
        assert (batch.images[0, :, :, 50:] == 0).all()
        assert (batch.images[1, :, :, 40:] == 0).all()
        assert (batch.images[2, :, :, 30:] == 0).all()

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="Empty batch"):
            ctc_collate([])


class TestBucketByWidthSampler:
    def test_coverage_and_grouping(self):
        # Widths that, once sorted, pair up neatly into batches of 2.
        widths = [300, 100, 310, 105, 200, 210]
        sampler = BucketByWidthSampler(widths, batch_size=2, shuffle=False)

        assert len(sampler) == math.ceil(len(widths) / 2)

        all_indices = []
        for batch_indices in sampler:
            assert len(batch_indices) <= 2
            # Within a batch, widths should be close (consecutive in sorted order).
            batch_widths = [widths[i] for i in batch_indices]
            assert max(batch_widths) - min(batch_widths) <= 15
            all_indices.extend(batch_indices)

        assert sorted(all_indices) == list(range(len(widths)))

    def test_deterministic_when_no_shuffle(self):
        widths = [100, 500, 110, 490, 105, 495]
        sampler = BucketByWidthSampler(widths, batch_size=2, shuffle=False)

        run1 = [batch for batch in sampler]
        run2 = [batch for batch in sampler]
        assert run1 == run2
