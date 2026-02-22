"""Tests for src/lm/build.py and src/training/ctc.build_beam_decoder wiring."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from modular_htr.data.tokenization import CharTokenizer
from modular_htr.lm.build import find_kenlm_binary, interpolate_kenlm
from modular_htr.training.ctc import build_beam_decoder

KENLM_AVAILABLE = (
    shutil.which("lmplz") is not None and shutil.which("build_binary") is not None
)


@pytest.fixture
def tokenizer_with_space() -> CharTokenizer:
    """5-char alphabet including a space character."""
    alphabet = ["<blank>", "a", " ", "b", "c"]
    index = {ch: i for i, ch in enumerate(alphabet)}
    return CharTokenizer(
        alphabet=alphabet,
        index=index,
        unicode_form=None,
        blank_index=0,
        pad_index=0,
    )


@pytest.fixture
def tokenizer_with_tab() -> CharTokenizer:
    """5-char alphabet including a tab character (e.g. Tridis checkpoint)."""
    alphabet = ["<blank>", "\t", "a", "b", "c"]
    index = {ch: i for i, ch in enumerate(alphabet)}
    return CharTokenizer(
        alphabet=alphabet,
        index=index,
        unicode_form=None,
        blank_index=0,
        pad_index=0,
    )


def test_find_kenlm_binary_not_found():
    with patch("modular_htr.lm.build.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="lmplz"):
            find_kenlm_binary("lmplz")


def test_find_kenlm_binary_found():
    with patch("modular_htr.lm.build.shutil.which", return_value="/usr/bin/lmplz"):
        result = find_kenlm_binary("lmplz")
        assert result == Path("/usr/bin/lmplz")


def test_interpolate_kenlm_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        interpolate_kenlm(
            model_dirs=[Path("/a"), Path("/b")],
            weights=[0.5],
            output_path=Path("/out.bin"),
        )


@patch("modular_htr.training.ctc.ctc_decoder")
def test_build_beam_decoder_without_lm(mock_decoder, simple_tokenizer):
    build_beam_decoder(simple_tokenizer)

    mock_decoder.assert_called_once()
    kwargs = mock_decoder.call_args.kwargs
    assert kwargs["lm"] is None
    assert kwargs["lexicon"] is None
    assert kwargs["sil_token"] == "<blank>"
    assert kwargs["tokens"] == list(simple_tokenizer.alphabet)


@patch("modular_htr.training.ctc.ctc_decoder")
def test_build_beam_decoder_with_lm(mock_decoder, tokenizer_with_space):
    build_beam_decoder(
        tokenizer_with_space,
        lm_path=Path("/fake/m.bin"),
        lm_weight=0.7,
        word_score=1.0,
        sil_score=-0.5,
    )

    mock_decoder.assert_called_once()
    kwargs = mock_decoder.call_args.kwargs
    assert kwargs["lm"] == "/fake/m.bin"
    assert kwargs["lm_weight"] == 0.7
    assert kwargs["word_score"] == 1.0
    assert kwargs["sil_score"] == -0.5
    assert kwargs["sil_token"] == "<space>"
    # Space should have been remapped to <space> in the token list
    assert " " not in kwargs["tokens"]
    assert "<space>" in kwargs["tokens"]

    # A character-level lexicon should be generated for LexiconDecoder
    lexicon_path = kwargs["lexicon"]
    assert lexicon_path is not None
    lines = Path(lexicon_path).read_text(encoding="utf-8").splitlines()
    # Lexicon should contain all non-blank tokens (one per line)
    non_blank = [t for t in kwargs["tokens"] if t != "<blank>"]
    assert len(lines) == len(non_blank)
    for line, token in zip(lines, non_blank):
        assert line == f"{token} {token}"


@patch("modular_htr.training.ctc.ctc_decoder")
def test_build_beam_decoder_with_lm_no_space(mock_decoder, simple_tokenizer):
    build_beam_decoder(simple_tokenizer, lm_path=Path("/fake/m.bin"))

    kwargs = mock_decoder.call_args.kwargs
    assert kwargs["lm"] == "/fake/m.bin"
    assert kwargs["sil_token"] == "<space>"
    # No space in alphabet, so tokens should be unchanged
    assert kwargs["tokens"] == list(simple_tokenizer.alphabet)

    # Lexicon should still be generated when LM is present
    lexicon_path = kwargs["lexicon"]
    assert lexicon_path is not None
    lines = Path(lexicon_path).read_text(encoding="utf-8").splitlines()
    non_blank = [t for t in kwargs["tokens"] if t != "<blank>"]
    assert len(lines) == len(non_blank)


@patch("modular_htr.training.ctc.ctc_decoder")
def test_build_beam_decoder_with_lm_tab(mock_decoder, tokenizer_with_tab):
    """Tab character is remapped to <tab> and produces a parseable lexicon."""
    build_beam_decoder(
        tokenizer_with_tab,
        lm_path=Path("/fake/m.bin"),
    )

    mock_decoder.assert_called_once()
    kwargs = mock_decoder.call_args.kwargs

    # Tab should have been remapped to <tab> in the token list
    assert "\t" not in kwargs["tokens"]
    assert "<tab>" in kwargs["tokens"]

    # The lexicon file must be parseable (no bare tab characters)
    lexicon_path = kwargs["lexicon"]
    assert lexicon_path is not None
    lines = Path(lexicon_path).read_text(encoding="utf-8").splitlines()
    non_blank = [t for t in kwargs["tokens"] if t != "<blank>"]
    assert len(lines) == len(non_blank)
    for line in lines:
        # Each line should be exactly "token token" -- two whitespace-separated
        # fields. A bare tab would produce 3+ fields or empty fields.
        parts = line.split()
        assert len(parts) == 2, f"Invalid lexicon line: {line!r}"
        assert parts[0] == parts[1]


def _write_synthetic_training_text(path: Path, n_lines: int = 200) -> None:
    """Write synthetic character-level KenLM training data.

    Generates enough variety to satisfy KenLM's Kneser-Ney discount estimation.
    """
    import random

    rng = random.Random(42)
    words = [
        "h e l l o",
        "w o r l d",
        "f o o",
        "b a r",
        "t e s t",
        "t h e",
        "q u i c k",
        "b r o w n",
        "f o x",
        "j u m p s",
        "o v e r",
        "l a z y",
        "d o g",
        "a n d",
        "c a t",
        "r u n s",
        "f a s t",
        "s l o w",
        "b i g",
        "s m a l l",
    ]
    lines = []
    for _ in range(n_lines):
        n_words = rng.randint(3, 8)
        selected = [rng.choice(words) for _ in range(n_words)]
        lines.append(" <space> ".join(selected))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM binaries not on PATH")
def test_build_kenlm_end_to_end(tmp_path):
    from modular_htr.lm.build import build_kenlm

    text_path = tmp_path / "train.txt"
    _write_synthetic_training_text(text_path)

    out = build_kenlm(text_path, tmp_path / "model.bin", order=3)

    assert out.exists()
    assert out.stat().st_size > 0

    import kenlm

    model = kenlm.Model(str(out))
    assert model.order == 3


@pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM binaries not on PATH")
def test_build_kenlm_with_intermediate(tmp_path):
    from modular_htr.lm.build import build_kenlm

    text_path = tmp_path / "train.txt"
    _write_synthetic_training_text(text_path)

    out = build_kenlm(text_path, tmp_path / "model.bin", order=3, intermediate=True)

    assert out.exists()
    intermediate_dir = tmp_path / "model_intermediate"
    assert intermediate_dir.exists()
    assert intermediate_dir.is_dir()
