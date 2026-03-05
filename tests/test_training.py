"""Forward and backward pass smoke tests."""

import torch

from modular_htr.data.tokenization import CharTokenizer
from modular_htr.training.ctc import CTCLossWrapper, greedy_ctc_decode

from modular_htr.cli import build_model

from .conftest import make_batch


def test_forward_pass(model_config, simple_tokenizer: CharTokenizer):
    """Forward pass produces correctly shaped logits."""
    num_classes = len(simple_tokenizer)
    model = build_model(model_config, num_classes)
    model.eval()

    batch = make_batch(height=model_config.fixed_height)
    with torch.no_grad():
        logits, _shortcut = model(batch.images, batch.widths)

    B = batch.images.size(0)
    T = logits.size(0)
    assert logits.shape == (T, B, num_classes)

    output_lengths = model.output_lengths(batch.widths)
    assert output_lengths.shape == (B,)
    assert (output_lengths > 0).all()
    assert output_lengths.max() <= T


def test_backward_pass(model_config, simple_tokenizer: CharTokenizer):
    """Gradients flow end-to-end through the model."""
    num_classes = len(simple_tokenizer)
    model = build_model(model_config, num_classes)
    model.train()

    batch = make_batch(height=model_config.fixed_height)
    logits, _shortcut = model(batch.images, batch.widths)
    output_lengths = model.output_lengths(batch.widths)

    loss_fn = CTCLossWrapper(blank=simple_tokenizer.blank_index)
    loss = loss_fn(logits, batch.targets, output_lengths, batch.target_lengths)
    loss.backward()

    assert model.fc.weight.grad is not None
    assert model.fc.weight.grad.abs().sum() > 0


def test_forward_pass_with_right_padding(model_config, simple_tokenizer: CharTokenizer):
    """output_lengths uses content widths, not padded tensor width."""
    num_classes = len(simple_tokenizer)
    model = build_model(model_config, num_classes)
    model.eval()

    B, H, padded_W, content_W = 2, model_config.fixed_height, 512, 200
    images = torch.randn(B, 1, H, padded_W)
    images[:, :, :, content_W:] = 0  # simulate right-padding
    widths = torch.tensor([content_W] * B, dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(images, widths)

    # output_lengths must reflect content widths, not padded tensor width.
    output_lengths = model.output_lengths(widths)
    stride = model.backbone.time_reduction
    expected_lengths = torch.tensor([content_W // stride] * B, dtype=torch.long)
    assert output_lengths.tolist() == expected_lengths.tolist()

    # Logits T dimension corresponds to the full padded tensor (512 / stride),
    # masking happens downstream via input_lengths passed to CTC.
    T = logits.size(0)
    assert T == padded_W // stride


def test_greedy_ctc_decode(simple_tokenizer: CharTokenizer):
    """Greedy CTC decode correctly collapses blanks and repeated characters."""
    # simple_tokenizer: 0=<blank>, 1=a, 2=b, 3=c, ...
    # Craft logits [T=8, B=1, C=11] where argmax path is:
    # [a, a, blank, b, b, b, c, a] -> collapsed: "abca"
    T, B, C = 8, 1, len(simple_tokenizer)
    logits = torch.full((T, B, C), -10.0)
    path = [1, 1, 0, 2, 2, 2, 3, 1]  # a, a, blank, b, b, b, c, a
    for t, idx in enumerate(path):
        logits[t, 0, idx] = 10.0

    input_lengths = torch.tensor([T])
    result = greedy_ctc_decode(logits, input_lengths, simple_tokenizer)

    assert result == ["abca"]


def test_greedy_ctc_decode_respects_input_lengths(simple_tokenizer: CharTokenizer):
    """Greedy decode only reads up to input_length, ignoring padding."""
    T, B, C = 8, 1, len(simple_tokenizer)
    logits = torch.full((T, B, C), -10.0)
    # Full path: [a, b, blank, blank, c, c, c, c]
    path = [1, 2, 0, 0, 3, 3, 3, 3]
    for t, idx in enumerate(path):
        logits[t, 0, idx] = 10.0

    # Only read first 4 timesteps -> path is [a, b, blank, blank] -> "ab"
    input_lengths = torch.tensor([4])
    result = greedy_ctc_decode(logits, input_lengths, simple_tokenizer)

    assert result == ["ab"]
