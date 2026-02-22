from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from tabulate import tabulate


def _levenshtein(ref: str, hyp: str) -> int:
    """
    Simple levenshtein distance without backtrace for the greedy decoder during training.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    # use 2-row DP to keep it light
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        r_i = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r_i == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[m]


def cer(refs: list[str], hyps: list[str]) -> float:
    assert len(refs) == len(hyps)
    total_edits = 0
    total_chars = 0
    for ref, hyp in zip(refs, hyps):
        ref = ref or ""
        hyp = hyp or ""
        total_edits += _levenshtein(ref, hyp)
        total_chars += len(ref)
    if total_chars == 0:
        return 0.0
    return total_edits / total_chars


def wer(refs: list[str], hyps: list[str]) -> float:
    assert len(refs) == len(hyps)
    total_edits = 0
    total_words = 0
    for ref, hyp in zip(refs, hyps):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_edits += _levenshtein(ref_words, hyp_words)  # type: ignore[arg-type]
        total_words += len(ref_words)
    if total_words == 0:
        return 0.0
    return total_edits / total_words


@dataclass(slots=True, frozen=True)
class EditOp:
    """A single edit operation in an alignment for the edit distance with backtrace."""

    op: str  # "match", "sub", "del", "ins"
    ref_char: str | None  # None for insertions
    hyp_char: str | None  # None for deletions


@dataclass(slots=True)
class Alignment:
    """Full alignment between a reference and hypothesis string."""

    ops: list[EditOp]
    distance: int


def align(ref: str, hyp: str) -> Alignment:
    """
    Full O(n*m) DP with backtrace to recover the optimal edit sequence.
    Used to display detailed information during model evaluation.
    """
    n, m = len(ref), len(hyp)

    # Cost matrix.
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,  # deletion
                D[i][j - 1] + 1,  # insertion
                D[i - 1][j - 1] + cost,  # match or substitution
            )

    # Backtrace from (n, m) to (0, 0).
    ops: list[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if D[i][j] == D[i - 1][j - 1] + cost:
                if cost == 0:
                    ops.append(EditOp("match", ref[i - 1], hyp[j - 1]))
                else:
                    ops.append(EditOp("sub", ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(EditOp("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(EditOp("ins", None, hyp[j - 1]))
            j -= 1

    ops.reverse()
    return Alignment(ops=ops, distance=D[n][m])


# Text normalization

MEDIEVAL_EQUIVALENCES: dict[str, str] = {
    "v": "u",
    "j": "i",
    "t": "c",
    "V": "U",
    "J": "I",
    "T": "C",
}

_MEDIEVAL_TABLE = str.maketrans(MEDIEVAL_EQUIVALENCES)


def normalize_medieval(text: str) -> str:
    """Apply medieval letter equivalences (v->u, j->i, t->c)."""
    return text.translate(_MEDIEVAL_TABLE)


def normalize_lower(text: str) -> str:
    """Lowercase."""
    return text.lower()


_MULTI_WS = re.compile(r"\s+")


def normalize_no_punctuation(text: str) -> str:
    """Remove all Unicode punctuation and symbol characters, then collapse whitespace."""
    stripped = "".join(c for c in text if unicodedata.category(c)[0] not in ("P", "S"))
    return _MULTI_WS.sub(" ", stripped).strip()


def per_sample_cer(
    refs: list[str],
    hyps: list[str],
    normalizer: Callable[[str], str] | None = None,
) -> list[float]:
    """CER for each (ref, hyp) pair.

    Empty ref -> 0.0 if hyp also empty, else 1.0.
    """
    assert len(refs) == len(hyps)
    results: list[float] = []
    for ref, hyp in zip(refs, hyps):
        ref = ref or ""
        hyp = hyp or ""
        if normalizer is not None:
            ref = normalizer(ref)
            hyp = normalizer(hyp)
        if len(ref) == 0:
            results.append(0.0 if len(hyp) == 0 else 1.0)
        else:
            results.append(_levenshtein(ref, hyp) / len(ref))
    return results


def corpus_cer_normalized(
    refs: list[str],
    hyps: list[str],
    normalizer: Callable[[str], str],
) -> float:
    """Apply normalizer to both sides, then compute corpus CER."""
    return cer(
        [normalizer(r or "") for r in refs],
        [normalizer(h or "") for h in hyps],
    )


def confusion_counts(
    refs: list[str],
    hyps: list[str],
) -> Counter[tuple[str, str | None, str | None]]:
    """Accumulate (op_type, ref_char, hyp_char) counts from alignments.

    Only errors are counted (matches excluded).
    """
    counts: Counter[tuple[str, str | None, str | None]] = Counter()
    for ref, hyp in zip(refs, hyps):
        alignment = align(ref or "", hyp or "")
        for op in alignment.ops:
            if op.op != "match":
                counts[(op.op, op.ref_char, op.hyp_char)] += 1
    return counts


def format_confusion_table(
    counts: Counter[tuple[str, str | None, str | None]],
    top_n: int = 20,
) -> str:
    """Format the top N confusions as a readable table."""
    rows = []
    for (op, ref_char, hyp_char), count in counts.most_common(top_n):
        ref_display = repr(ref_char) if ref_char is not None else "-"
        hyp_display = repr(hyp_char) if hyp_char is not None else "-"
        rows.append([op.upper(), ref_display, hyp_display, count])
    return tabulate(
        rows,
        headers=["Type", "Ref", "Hyp", "Count"],
        tablefmt="simple",
        showindex=range(1, len(rows) + 1),
    )
