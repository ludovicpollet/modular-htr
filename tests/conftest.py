import pytest
import torch

from modular_htr.config import (
    ConvNeXtBackboneConfig,
    ConvNeXtTinyPretrainedConfig,
    ConvNeXtTinyPretrainedModelConfig,
    KrakenModelConfig,
    ModernPylaiaModelConfig,
    PylaiaModelConfig,
    RetsinasMHAModelConfig,
    RetsinasModelConfig,
)
from modular_htr.data.dataloaders import Batch
from modular_htr.data.tokenization import CharTokenizer
from modular_htr.types import UnicodeForm


def _convnext_model_config():
    """ConvNeXtTinyPretrainedModelConfig with pretrained weights disabled for testing."""
    cfg = ConvNeXtTinyPretrainedModelConfig()
    assert isinstance(cfg.backbone, ConvNeXtTinyPretrainedConfig)
    cfg.backbone = ConvNeXtBackboneConfig(
        channels=list(cfg.backbone.channels),
        blocks_per_stage=list(cfg.backbone.blocks_per_stage),
        downsample_kernels=list(cfg.backbone.downsample_kernels),
        stem_stride=cfg.backbone.stem_stride,
    )
    return cfg


NAMED_CONFIGS = [
    PylaiaModelConfig,
    ModernPylaiaModelConfig,
    KrakenModelConfig,
    RetsinasModelConfig,
    RetsinasMHAModelConfig,
    _convnext_model_config,
]


@pytest.fixture
def simple_tokenizer() -> CharTokenizer:
    """10-char alphabet, no unicode normalization."""
    chars = list("abcdefghij")
    alphabet = ["<blank>"] + chars
    index = {ch: i for i, ch in enumerate(alphabet)}
    return CharTokenizer(
        alphabet=alphabet,
        index=index,
        unicode_form=None,
        blank_index=0,
        pad_index=0,
    )


@pytest.fixture
def unicode_tokenizer() -> CharTokenizer:
    """4-char alphabet with accented characters, NFC normalization."""
    chars = ["\u00e9", "\u00e8", "a", "b"]  # e-acute, e-grave, a, b
    alphabet = ["<blank>"] + chars
    index = {ch: i for i, ch in enumerate(alphabet)}
    return CharTokenizer(
        alphabet=alphabet,
        index=index,
        unicode_form=UnicodeForm.NFC,
        blank_index=0,
        pad_index=0,
    )


@pytest.fixture(params=NAMED_CONFIGS, ids=lambda c: c.__name__)
def model_config(request):
    """Yields each named model config."""
    return request.param()


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def make_batch(height: int, width: int = 200, num_targets: int = 5) -> Batch:
    """Create a synthetic batch of 2 grayscale images."""
    B = 2
    images = torch.randn(B, 1, height, width)
    targets = torch.randint(1, 10, (B * num_targets,))
    target_lengths = torch.tensor([num_targets] * B, dtype=torch.long)
    widths = torch.tensor([width] * B, dtype=torch.long)
    return Batch(
        images=images,
        targets=targets,
        target_lengths=target_lengths,
        widths=widths,
        ids=[None] * B,
        texts=["dummy"] * B,
    )
