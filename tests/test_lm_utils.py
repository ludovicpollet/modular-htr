"""Tests for src/lm/utils.py -- pure text preprocessing functions."""

from modular_htr.lm.build import load_texts_from_files
from modular_htr.lm.utils import (
    prepare_kenlm_training_text,
    remap_tokens_for_lm,
    text_to_kenlm_format,
    write_lexicon,
)
from modular_htr.types import UnicodeForm


def test_text_to_kenlm_format_basic():
    assert text_to_kenlm_format("hello world") == "h e l l o <space> w o r l d"


def test_text_to_kenlm_format_single_word():
    assert text_to_kenlm_format("abc") == "a b c"


def test_text_to_kenlm_format_empty():
    assert text_to_kenlm_format("") == ""


def test_text_to_kenlm_format_multiple_spaces():
    # Double spaces are collapsed by normalize_text before LM formatting.
    assert text_to_kenlm_format("a  b") == "a <space> b"


def test_text_to_kenlm_format_unicode_nfd():
    # U+00E9 (e-acute) decomposes to 'e' + combining acute accent under NFD
    result = text_to_kenlm_format("\u00e9", UnicodeForm.NFD)
    assert result == "e \u0301"


def test_text_to_kenlm_format_no_normalization():
    result = text_to_kenlm_format("\u00e9", None)
    assert result == "\u00e9"


def test_text_to_kenlm_format_tab():
    assert text_to_kenlm_format("a\tb") == "a <space> b"


def test_text_to_kenlm_format_strip_space_before_punctuation():
    result = text_to_kenlm_format("hello , world", strip_space_before_punctuation=True)
    assert result == "h e l l o , <space> w o r l d"


def test_remap_tokens_replaces_space():
    tokens = ["<blank>", "a", " ", "b"]
    assert remap_tokens_for_lm(tokens) == ["<blank>", "a", "<space>", "b"]


def test_remap_tokens_no_space():
    tokens = ["<blank>", "a", "b"]
    assert remap_tokens_for_lm(tokens) == ["<blank>", "a", "b"]


def test_remap_tokens_replaces_tab():
    tokens = ["<blank>", "a", "\t", "b"]
    assert remap_tokens_for_lm(tokens) == ["<blank>", "a", "<tab>", "b"]


def test_prepare_kenlm_training_text(tmp_path):
    texts = ["hello world", "foo bar"]
    out = prepare_kenlm_training_text(texts, tmp_path / "sub" / "train.txt")

    assert out == tmp_path / "sub" / "train.txt"
    assert out.exists()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == "h e l l o <space> w o r l d"
    assert lines[1] == "f o o <space> b a r"


def test_prepare_kenlm_training_text_skips_empty(tmp_path):
    texts = ["hello", "", "world"]
    out = prepare_kenlm_training_text(texts, tmp_path / "train.txt")

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_prepare_kenlm_training_text_unicode(tmp_path):
    texts = ["\u00e9t\u00e9"]  # "ete" with accents
    out = prepare_kenlm_training_text(texts, tmp_path / "train.txt", UnicodeForm.NFC)

    content = out.read_text(encoding="utf-8").strip()
    # NFC keeps e-acute as a single codepoint
    assert content == "\u00e9 t \u00e9"


def test_load_texts_from_files(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("line1\n\nline2\n", encoding="utf-8")
    f2.write_text("line3\n", encoding="utf-8")

    result = load_texts_from_files([f1, f2])
    assert result == ["line1", "line2", "line3"]


def test_load_texts_from_files_empty(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")

    assert load_texts_from_files([f]) == []


def test_write_lexicon_basic(tmp_path):
    tokens = ["a", "|", "b", "c"]
    out = write_lexicon(tokens, tmp_path / "lexicon.txt")

    assert out == tmp_path / "lexicon.txt"
    assert out.exists()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert lines[0] == "a a"
    assert lines[1] == "| |"
    assert lines[2] == "b b"
    assert lines[3] == "c c"


def test_write_lexicon_creates_parents(tmp_path):
    out = write_lexicon(["x"], tmp_path / "nested" / "dir" / "lex.txt")
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() == "x x"


def test_write_lexicon_empty(tmp_path):
    out = write_lexicon([], tmp_path / "empty.txt")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""
