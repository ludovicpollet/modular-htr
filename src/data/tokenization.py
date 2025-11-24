import collections
from dataclasses import dataclass
import datasets
from typing import Any


type Alphabet = list[str]
type Index = dict[str, int]


@dataclass
class CharTokenizer:
    alphabet: Alphabet
    index: Index
    blank_index: int
    pad_index: int

    def encode(self, text: str) -> list[int]:
        return [self.index[c] for c in text if c in self.index]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.alphabet[i]
            for i in ids
            if 0 <= i < len(self.alphabet)
            and i not in (self.blank_index, self.pad_index)
        )

    def __len__(self) -> int:
        return len(self.alphabet)


def build_char_tokenizer(
    ds: datasets.Dataset | datasets.DatasetDict,
    text_col: str = "text",
    splits: str = "all",
    blank_token: str = "<blank>",
    pad_token: str = "<pad>",
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
        for ex in split_ds:
            counter.update(ex[text_col])
    charset = sorted(counter.keys())

    alphabet: Alphabet = [blank_token, pad_token] + charset
    index: Index = {ch: i for i, ch in enumerate(alphabet)}

    blank_index = index[blank_token]
    pad_index = index[pad_token]

    return CharTokenizer(
        alphabet=alphabet,
        index=index,
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
    )
