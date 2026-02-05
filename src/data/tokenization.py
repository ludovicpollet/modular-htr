import unicodedata
import collections
from dataclasses import dataclass
from typing import Any

import datasets

from src.types import UnicodeForm

type Alphabet = list[str]
type Index = dict[str, int]


@dataclass
class CharTokenizer:
    alphabet: Alphabet
    index: Index
    unicode_form: UnicodeForm | None
    blank_index: int
    pad_index: int

    def encode(self, text: str) -> list[int]:
        text = normalize_text(text, self.unicode_form)
        return [self.index[c] for c in text if c in self.index]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.alphabet[i]
            for i in ids
            if 0 <= i < len(self.alphabet) and i != self.blank_index
        )

    def __len__(self) -> int:
        return len(self.alphabet)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alphabet": self.alphabet,
            "blank_index": self.blank_index,
            "pad_index": self.pad_index,
            "unicode_form": self.unicode_form.value
            if self.unicode_form is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharTokenizer":
        alphabet_raw = data["alphabet"]
        blank_index_raw = data["blank_index"]
        pad_index_raw = data.get("pad_index", blank_index_raw)
        unicode_form_str = data.get("unicode_form")

        alphabet: Alphabet = list(alphabet_raw)
        index: Index = {ch: i for i, ch in enumerate(alphabet)}
        blank_index = int(blank_index_raw)
        pad_index = int(pad_index_raw)
        if not (0 <= blank_index <= len(alphabet)):
            raise ValueError("Invalid blank index")
        if not (0 <= pad_index <= len(alphabet)):
            raise ValueError("Invalid pad index")
        if not blank_index == 0:
            print("Warning: blank index needs to be 0 for CTC decoding")

        unicode_form = (
            UnicodeForm(unicode_form_str) if unicode_form_str is not None else None
        )

        return cls(
            alphabet=alphabet,
            unicode_form=unicode_form,
            index=index,
            blank_index=blank_index,
            pad_index=pad_index,
        )


def normalize_text(text: str, unicode_form: UnicodeForm | None) -> str:
    if unicode_form is not None:
        text = unicodedata.normalize(unicode_form.value, text)
    return text


def _count_batch(batch, text_col, unicode_form):
    return {
        "chars": ["".join(normalize_text(t, unicode_form) for t in batch[text_col])]
    }


def build_char_tokenizer(
    ds: datasets.Dataset | datasets.DatasetDict,
    text_col: str = "text",
    unicode_form: UnicodeForm | None = None,
    splits: str = "all",
    blank_token: str = "<blank>",
    num_proc: int = 16,
) -> CharTokenizer:
    if isinstance(ds, datasets.DatasetDict):
        if splits == "train":
            iterables = [ds["train"]]
        elif splits == "all":
            iterables = list(ds.values())
        else:
            raise ValueError("Unknown splits option")
    else:
        iterables = [ds]

    counter: collections.Counter[str] = collections.Counter()
    for split_ds in iterables:
        tmp = split_ds.map(
            _count_batch,
            fn_kwargs={"text_col": text_col, "unicode_form": unicode_form},
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=split_ds.column_names,
            desc="Counting characters",
        )
        for s in tmp["chars"]:
            counter.update(s)
    charset = sorted(counter.keys())

    alphabet: Alphabet = [blank_token] + charset
    index: Index = {ch: i for i, ch in enumerate(alphabet)}

    blank_index = index[blank_token]
    pad_index = blank_index

    return CharTokenizer(
        alphabet=alphabet,
        index=index,
        unicode_form=unicode_form,
        blank_index=blank_index,
        pad_index=pad_index,
    )


def _tokenize_example(
    example: dict[str, Any],
    tokenizer: CharTokenizer,
    text_col: str,
) -> dict[str, Any]:
    text = example[text_col]
    labels = tokenizer.encode(text)
    example["labels"] = labels
    example["label_length"] = len(labels)
    return example


def apply_ctc_tokenizer(
    ds: datasets.Dataset | datasets.DatasetDict,
    tokenizer: CharTokenizer,
    text_col: str = "text",
) -> datasets.Dataset | datasets.DatasetDict:
    return ds.map(
        _tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "text_col": text_col},
        desc="Applying CTC tokenizer",
    )
