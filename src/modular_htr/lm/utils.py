from pathlib import Path

from modular_htr.data.tokenization import normalize_text
from modular_htr.types import UnicodeForm

# Placeholder used to represent the space character in KenLM training data
# and in the token list passed to the beam decoder.
SPACE_PLACEHOLDER = "<space>"

# Placeholder for the tab character in the decoder token list. Tabs are
# treated as spaces in LM training data, but need a distinct placeholder in
# the token list to avoid duplicate entries.
TAB_PLACEHOLDER = "<tab>"


def text_to_kenlm_format(
    text: str,
    unicode_form: UnicodeForm | None = None,
    strip_space_before_punctuation: bool = False,
) -> str:
    """
    Convert a text line to character-level KenLM format.
    Characters are separated by a space, after the actual space is replaced by
    ``SPACE_PLACEHOLDER``.
    """
    text = normalize_text(text, unicode_form, strip_space_before_punctuation)
    chars = [SPACE_PLACEHOLDER if c in (" ", "\t") else c for c in text]
    return " ".join(chars)


def prepare_kenlm_training_text(
    texts: list[str],
    output_path: Path,
    unicode_form: UnicodeForm | None = None,
    strip_space_before_punctuation: bool = False,
) -> Path:
    """
    Write texts to a file in character-level KenLM format.
    Each input text becomes one line in the output file.
    Streams through the list to keep memory usage low.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for text in texts:
            line = text_to_kenlm_format(
                text, unicode_form, strip_space_before_punctuation
            )
            if line:
                f.write(line + "\n")
    return output_path


def write_lexicon(tokens: list[str], output_path: Path) -> Path:
    """
    Write a character-level lexicon file for flashlight's LexiconDecoder.

    Each token is treated as a single-character "word" that maps to itself,
    so the decoder uses ``word_score`` as a per-character insertion bonus.
    This is the approach used by PyLaia (Teklia) to counteract LM length bias.

    The blank token should NOT be included in *tokens* -- it is handled
    separately by the decoder's ``blank_token`` parameter.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for token in tokens:
            f.write(f"{token} {token}\n")
    return output_path


def remap_tokens_for_lm(tokens: list[str]) -> list[str]:
    """
    Replace whitespace characters with unique placeholders in a token list.
    This keeps the beam decoder's token vocabulary consistent with the
    character-level KenLM training format and avoids unparseable lexicon
    lines (tabs would break the whitespace-delimited lexicon format).
    """
    remap = {" ": SPACE_PLACEHOLDER, "\t": TAB_PLACEHOLDER}
    return [remap.get(t, t) for t in tokens]
