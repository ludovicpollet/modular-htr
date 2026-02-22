"""Checkpoint save/load round-trip and tokenizer serialization tests."""

import torch

from modular_htr.config import PylaiaModelConfig
from modular_htr.data.tokenization import CharTokenizer
from modular_htr.model.HTRModel import HTRModel
from modular_htr.training.finetune import (
    load_checkpoint,
    merge_tokenizers,
    resize_output_layer,
    restore_model_from_checkpoint,
)
from modular_htr.training.utils import CheckpointManager
from modular_htr.types import NewHeadInit, NewSymbolsInit, UnicodeForm

from modular_htr.cli import build_model


def test_tokenizer_serialization(simple_tokenizer, unicode_tokenizer):
    """to_dict -> from_dict preserves all fields including unicode_form."""
    for tok in [simple_tokenizer, unicode_tokenizer]:
        restored = CharTokenizer.from_dict(tok.to_dict())
        assert restored.alphabet == tok.alphabet
        assert restored.blank_index == tok.blank_index
        assert restored.pad_index == tok.pad_index
        assert restored.unicode_form == tok.unicode_form


def test_tokenizer_encode_decode_round_trip(simple_tokenizer):
    """Encoding then decoding a known string recovers the original."""
    text = "abcde"
    ids = simple_tokenizer.encode(text)
    decoded = simple_tokenizer.decode(ids)
    assert decoded == text


def test_tokenizer_encode_skips_oov(simple_tokenizer):
    """Characters not in the alphabet are silently dropped during encoding."""
    # simple_tokenizer has a-j; 'z' and '!' are OOV
    text = "a!bz"
    ids = simple_tokenizer.encode(text)
    decoded = simple_tokenizer.decode(ids)
    assert decoded == "ab"


def _save_and_reload(model, tokenizer, tmp_path):
    """Helper: save a checkpoint via CheckpointManager and reload it."""
    mgr = CheckpointManager(
        save_dir=tmp_path,
        model_config=model.to_config(),
        tokenizer=tokenizer,
    )
    # Minimal optimizer/scheduler/scaler stubs for save_last
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mgr.save_last(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        val_metrics={"val_loss": 1.0},
    )
    ckpt = load_checkpoint(tmp_path / "last.pt", map_location="cpu")
    return ckpt


def test_checkpoint_round_trip(model_config, simple_tokenizer, tmp_path):
    """Full save -> load -> restore cycle for each named model config."""
    num_classes = len(simple_tokenizer)
    model = build_model(model_config, num_classes)
    model.eval()

    ckpt = _save_and_reload(model, simple_tokenizer, tmp_path)
    restored_model, restored_tok = restore_model_from_checkpoint(ckpt, device="cpu")

    assert isinstance(restored_model, HTRModel)
    assert restored_model.fc.out_features == num_classes
    assert restored_tok is not None
    assert restored_tok.alphabet == simple_tokenizer.alphabet
    assert restored_tok.unicode_form == simple_tokenizer.unicode_form
    assert torch.equal(model.fc.weight, restored_model.fc.weight)


def test_checkpoint_unicode_tokenizer(model_config, unicode_tokenizer, tmp_path):
    """Checkpoint round-trip preserves unicode_form (regression test)."""
    num_classes = len(unicode_tokenizer)
    model = build_model(model_config, num_classes)
    model.eval()

    ckpt = _save_and_reload(model, unicode_tokenizer, tmp_path)
    _restored_model, restored_tok = restore_model_from_checkpoint(ckpt, device="cpu")

    assert restored_tok is not None
    assert restored_tok.unicode_form == UnicodeForm.NFC


def test_finetune_head_rebuild(simple_tokenizer):
    """Shared characters keep their pretrained weights after alphabet merge."""
    # Build a pretrained model with simple_tokenizer: <blank> a b c d e f g h i j
    num_classes = len(simple_tokenizer)
    model = build_model(PylaiaModelConfig(), num_classes)
    model.eval()
    old_weight = model.fc.weight.detach().clone()
    old_bias = model.fc.bias.detach().clone()

    # Finetune tokenizer: overlaps on a-e, drops f-j, adds k-m
    ft_chars = list("abcde") + list("klm")
    ft_alphabet = ["<blank>"] + ft_chars
    ft_index = {ch: i for i, ch in enumerate(ft_alphabet)}
    ft_tokenizer = CharTokenizer(
        alphabet=ft_alphabet,
        index=ft_index,
        unicode_form=None,
        blank_index=0,
        pad_index=0,
    )

    merged = merge_tokenizers(simple_tokenizer, ft_tokenizer, drop_unused=True)
    model = resize_output_layer(
        model,
        simple_tokenizer,
        merged,
        head_init=NewHeadInit.COPY,
        new_class_init=NewSymbolsInit.KAIMING,
    )

    # Verify shared characters kept their original weight rows and biases
    for ch in ["<blank>"] + list("abcde"):
        old_idx = simple_tokenizer.index[ch]
        new_idx = merged.index[ch]
        assert torch.equal(model.fc.weight[new_idx], old_weight[old_idx]), (
            f"Weight mismatch for shared char '{ch}'"
        )
        assert torch.equal(model.fc.bias[new_idx], old_bias[old_idx]), (
            f"Bias mismatch for shared char '{ch}'"
        )


def test_finetune_head_reset(simple_tokenizer):
    """RESET mode zeroes out both weights and biases."""
    num_classes = len(simple_tokenizer)
    model = build_model(PylaiaModelConfig(), num_classes)
    model.eval()

    merged = merge_tokenizers(simple_tokenizer, simple_tokenizer, drop_unused=False)
    model = resize_output_layer(
        model,
        simple_tokenizer,
        merged,
        head_init=NewHeadInit.RESET,
        new_class_init=NewSymbolsInit.ZERO,
    )

    assert (model.fc.bias == 0).all()
    assert (model.fc.weight == 0).all()
