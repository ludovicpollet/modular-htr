"""Model construction and config round-trip tests."""

import pytest
import torch

from modular_htr.config import ConvNeXtBackboneConfig, RetsinasModelConfig
from modular_htr.data.tokenization import CharTokenizer
from modular_htr.model.HTRModel import HTRModel
from modular_htr.training.finetune import freeze_cnn_stages, make_unfreeze_callback

from modular_htr.cli import build_model

from .conftest import NAMED_CONFIGS, make_batch


def test_model_construction(model_config, simple_tokenizer: CharTokenizer):
    """Build each named config and verify output head size."""
    num_classes = len(simple_tokenizer)
    model = build_model(model_config, num_classes)
    assert model.fc.out_features == num_classes


def test_config_round_trip(model_config, simple_tokenizer: CharTokenizer):
    """to_config -> from_config -> load_state_dict produces identical weights."""
    num_classes = len(simple_tokenizer)
    model = build_model(model_config, num_classes)

    config = model.to_config()
    restored = HTRModel.from_config(config, num_classes=num_classes)
    restored.load_state_dict(model.state_dict())

    assert torch.equal(model.fc.weight, restored.fc.weight)
    assert torch.equal(model.fc.bias, restored.fc.bias)


def test_convnext_backbone_round_trip(simple_tokenizer: CharTokenizer):
    """ConvNeXt backbone serializes and restores correctly."""
    cfg = RetsinasModelConfig()
    cfg.backbone = ConvNeXtBackboneConfig()
    num_classes = len(simple_tokenizer)
    model = build_model(cfg, num_classes)

    config = model.to_config()
    restored = HTRModel.from_config(config, num_classes=num_classes)
    restored.load_state_dict(model.state_dict())

    assert torch.equal(model.fc.weight, restored.fc.weight)
    assert torch.equal(model.fc.bias, restored.fc.bias)


def test_convnext_forward_pass(simple_tokenizer: CharTokenizer):
    """ConvNeXt-backed model produces correct output shapes and lengths."""
    from .conftest import _convnext_model_config

    cfg = _convnext_model_config()
    num_classes = len(simple_tokenizer)
    model = build_model(cfg, num_classes)
    model.eval()

    batch = make_batch(height=cfg.fixed_height, width=256)
    with torch.no_grad():
        logits, shortcut = model(batch.images, batch.widths)

    lengths = model.output_lengths(batch.widths)
    T, B, C = logits.shape
    assert B == 2
    assert C == num_classes
    assert T == lengths[0].item()
    assert shortcut is None  # ConvNeXtTinyPretrainedModelConfig has shortcut_ctc=False


@pytest.mark.parametrize(
    "config_factory",
    NAMED_CONFIGS,
    ids=lambda c: c.__name__,
)
def test_freeze_and_unfreeze_stages(config_factory, simple_tokenizer: CharTokenizer):
    """freeze_cnn_stages freezes stem + stages; make_unfreeze_callback restores them."""
    cfg = config_factory()
    num_classes = len(simple_tokenizer)
    model = build_model(cfg, num_classes)

    # Determine available stages
    if hasattr(model, "backbone"):
        stages = list(model.backbone.stages)
    else:
        pytest.skip("Model has no backbone.stages")

    num_to_freeze = min(2, len(stages))
    freeze_cnn_stages(model, num_to_freeze, freeze_stem=True)

    # Verify frozen stages
    for stage in stages[:num_to_freeze]:
        for p in stage.parameters():
            assert not p.requires_grad, "Frozen stage param should not require grad"

    # Verify unfrozen stages still trainable
    for stage in stages[num_to_freeze:]:
        for p in stage.parameters():
            assert p.requires_grad, "Non-frozen stage param should require grad"

    # Verify stem is frozen (all backbone types have a stem)
    stem = getattr(model.backbone, "stem", None)
    if stem is not None:
        for p in stem.parameters():
            assert not p.requires_grad, "Frozen stem param should not require grad"

    # Unfreeze via callback
    callback = make_unfreeze_callback(
        unfreeze_epoch=1, num_stages=num_to_freeze, unfreeze_stem=True
    )
    callback(epoch=1, model=model)

    # Verify everything is unfrozen
    for stage in stages[:num_to_freeze]:
        for p in stage.parameters():
            assert p.requires_grad, "Unfrozen stage param should require grad"

    if stem is not None:
        for p in stem.parameters():
            assert p.requires_grad, "Unfrozen stem param should require grad"
