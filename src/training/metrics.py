def _levenshtein(ref: str, hyp: str) -> int:
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
