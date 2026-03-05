import unicodedata

import PIL.Image
import pytest

from modular_htr.data.transforms import make_preprocessing_fn, resize_to_fit


@pytest.mark.parametrize(
    "input_w, max_w, expected_w",
    [
        (100, 200, 100),  # narrow image: untouched
        (400, 200, 200),  # wide image: scaled down
        (200, 200, 200),  # at limit: untouched
    ],
    ids=["narrow", "wide", "at-limit"],
)
def test_resize_to_fit(input_w, max_w, expected_w):
    """resize_to_fit scales down images wider than max_width, leaves others untouched."""
    fixed_height = 64
    img = PIL.Image.new("L", (input_w, fixed_height))
    result = resize_to_fit(img, fixed_height, max_w)
    assert result.size == (expected_w, fixed_height)


def test_preprocessing_normalizes_whitespace():
    """Verify that make_preprocessing_fn strips and collapses whitespace.

    Leading/trailing spaces, double spaces, and tabs in annotations should
    be cleaned up by normalize_text before becoming training targets.
    """
    from modular_htr.data.tokenization import CharTokenizer

    chars = list("abc ")
    alphabet = ["<blank>"] + chars
    index = {ch: i for i, ch in enumerate(alphabet)}
    tokenizer = CharTokenizer(
        alphabet=alphabet,
        index=index,
        unicode_form=None,
        blank_index=0,
        pad_index=0,
    )

    img = PIL.Image.new("L", (200, 64))
    preprocess = make_preprocessing_fn(fixed_height=64, tokenizer=tokenizer)

    # Leading/trailing spaces stripped.
    result = preprocess({"image": img, "text": "  a b c  "})
    assert result["text"] == "a b c"

    # Double spaces collapsed.
    result = preprocess({"image": img, "text": "a  b  c"})
    assert result["text"] == "a b c"

    # Tabs become spaces and get collapsed.
    result = preprocess({"image": img, "text": "a\t\tb\tc"})
    assert result["text"] == "a b c"


def test_preprocessing_strips_space_before_punctuation():
    """Verify that make_preprocessing_fn strips spaces before punctuation
    when the tokenizer has strip_space_before_punctuation enabled."""
    from modular_htr.data.tokenization import CharTokenizer

    chars = list("helo wrd,.")
    alphabet = ["<blank>"] + sorted(set(chars))
    index = {ch: i for i, ch in enumerate(alphabet)}
    tokenizer = CharTokenizer(
        alphabet=alphabet,
        index=index,
        unicode_form=None,
        blank_index=0,
        pad_index=0,
        strip_space_before_punctuation=True,
    )

    img = PIL.Image.new("L", (200, 64))
    preprocess = make_preprocessing_fn(fixed_height=64, tokenizer=tokenizer)

    result = preprocess({"image": img, "text": "hello , world ."})
    assert result["text"] == "hello, world."


def test_preprocessing_normalizes_text(unicode_tokenizer):
    """Verify that make_preprocessing_fn normalizes text to match the tokenizer's alphabet.

    Simulates the real-world case where source XML contains NFD combining diacritics
    (e.g. 'e' + combining acute) but the tokenizer's alphabet contains precomposed
    NFC characters (e.g. 'é'). The returned text field must be NFC-normalized so that
    batch.texts used for CER/WER evaluation matches the decoded hypotheses.
    """
    # NFD form: 'e' + combining acute accent -> should become NFC 'é'
    nfd_text = unicodedata.normalize("NFD", "é")
    assert len(nfd_text) == 2  # sanity: 'e' + '\u0301'

    img = PIL.Image.new("L", (200, 64))
    preprocess = make_preprocessing_fn(
        fixed_height=64,
        tokenizer=unicode_tokenizer,
    )

    result = preprocess({"image": img, "text": nfd_text})

    assert result["text"] == "é"
    assert len(result["text"]) == 1  # single precomposed character
    assert result["label_length"] == 1
    assert result["labels"] == [unicode_tokenizer.index["é"]]
