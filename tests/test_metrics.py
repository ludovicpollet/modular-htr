import pytest

from modular_htr.training.metrics import (
    _levenshtein,
    align,
    confusion_counts,
    corpus_cer_normalized,
    format_confusion_table,
    normalize_lower,
    normalize_medieval,
    normalize_no_punctuation,
    per_sample_cer,
)


class TestAlign:
    def test_identical(self):
        a = align("abc", "abc")
        assert a.distance == 0
        assert len(a.ops) == 3
        assert all(op.op == "match" for op in a.ops)

    def test_substitution(self):
        a = align("abc", "axc")
        assert a.distance == 1
        ops = [(op.op, op.ref_char, op.hyp_char) for op in a.ops]
        assert ops == [
            ("match", "a", "a"),
            ("sub", "b", "x"),
            ("match", "c", "c"),
        ]

    def test_deletion(self):
        a = align("abc", "ac")
        assert a.distance == 1
        op_types = [op.op for op in a.ops]
        assert "del" in op_types
        # The deleted character should be 'b'.
        deleted = [op for op in a.ops if op.op == "del"]
        assert len(deleted) == 1
        assert deleted[0].ref_char == "b"
        assert deleted[0].hyp_char is None

    def test_insertion(self):
        a = align("ac", "abc")
        assert a.distance == 1
        op_types = [op.op for op in a.ops]
        assert "ins" in op_types
        inserted = [op for op in a.ops if op.op == "ins"]
        assert len(inserted) == 1
        assert inserted[0].ref_char is None
        assert inserted[0].hyp_char == "b"

    def test_empty_both(self):
        a = align("", "")
        assert a.distance == 0
        assert a.ops == []

    def test_empty_ref(self):
        a = align("", "abc")
        assert a.distance == 3
        assert all(op.op == "ins" for op in a.ops)

    def test_empty_hyp(self):
        a = align("abc", "")
        assert a.distance == 3
        assert all(op.op == "del" for op in a.ops)

    def test_distance_matches_levenshtein(self):
        """Verify that align().distance agrees with the existing _levenshtein()."""
        pairs = [
            ("kitten", "sitting"),
            ("saturday", "sunday"),
            ("", "hello"),
            ("hello", ""),
            ("abc", "abc"),
            ("a", "b"),
        ]
        for ref, hyp in pairs:
            assert align(ref, hyp).distance == _levenshtein(ref, hyp), (
                f"Mismatch for ({ref!r}, {hyp!r})"
            )


class TestNormalization:
    def test_normalize_medieval(self):
        assert normalize_medieval("vuij") == "uuii"
        assert normalize_medieval("tT") == "cC"
        # Characters outside the mapping are untouched.
        assert normalize_medieval("abc") == "abc"

    def test_normalize_lower(self):
        assert normalize_lower("AbC") == "abc"


class TestNormalizeNoPunctuation:
    def test_removes_common_punctuation(self):
        assert normalize_no_punctuation("hello, world.") == "hello world"
        assert normalize_no_punctuation("a;b:c!d?") == "abcd"

    def test_preserves_letters_digits_spaces(self):
        assert normalize_no_punctuation("abc 123") == "abc 123"

    def test_removes_unicode_punctuation(self):
        # Guillemets (Pf/Pi), em-dash (Pd), ellipsis (Po).
        assert normalize_no_punctuation("\u00ab\u00bb\u2014\u2026") == ""

    def test_removes_symbols(self):
        assert normalize_no_punctuation("price: $100 + \u20ac50") == "price 100 50"

    def test_collapses_whitespace_after_stripping(self):
        # "hello , world." -> strip punct -> "hello  world" -> collapse -> "hello world"
        assert normalize_no_punctuation("hello , world.") == "hello world"


class TestPerSampleCER:
    def test_basic(self):
        refs = ["abc", "abcd"]
        hyps = ["abc", "abXX"]
        scores = per_sample_cer(refs, hyps)
        assert scores[0] == 0.0  # perfect match
        assert scores[1] == pytest.approx(2.0 / 4.0)  # 2 subs out of 4

    def test_empty_ref_empty_hyp(self):
        assert per_sample_cer([""], [""]) == [0.0]

    def test_empty_ref_nonempty_hyp(self):
        assert per_sample_cer([""], ["x"]) == [1.0]

    def test_with_normalizer(self):
        # "vuij" and "uuii" are identical after medieval normalization.
        refs = ["vuij"]
        hyps = ["uuii"]
        scores_raw = per_sample_cer(refs, hyps)
        scores_norm = per_sample_cer(refs, hyps, normalizer=normalize_medieval)
        assert scores_raw[0] > 0.0
        assert scores_norm[0] == 0.0


class TestCorpusCERNormalized:
    def test_medieval_normalization_reduces_cer(self):
        refs = ["vuij", "abc"]
        hyps = ["uuii", "abc"]
        from modular_htr.training.metrics import cer

        raw = cer(refs, hyps)
        normalized = corpus_cer_normalized(refs, hyps, normalize_medieval)
        assert normalized < raw


class TestConfusionCounts:
    def test_basic(self):
        refs = ["abc", "abc"]
        hyps = ["axc", "axc"]
        counts = confusion_counts(refs, hyps)
        # Two substitutions b->x across the two samples.
        assert counts[("sub", "b", "x")] == 2
        # Matches should not appear.
        assert ("match", "a", "a") not in counts

    def test_deletions_and_insertions(self):
        refs = ["abc"]
        hyps = ["ac"]
        counts = confusion_counts(refs, hyps)
        assert counts[("del", "b", None)] == 1


class TestFormatConfusionTable:
    def test_smoke(self):
        """Just make sure it runs and returns a non-empty string."""
        refs = ["abc"]
        hyps = ["axc"]
        counts = confusion_counts(refs, hyps)
        table = format_confusion_table(counts, top_n=5)
        assert isinstance(table, str)
        assert "SUB" in table
        assert len(table) > 0
