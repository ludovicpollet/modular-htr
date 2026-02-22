import tempfile
from pathlib import Path
from typing import Literal

import torch
from torchaudio.models.decoder import ctc_decoder

from modular_htr.data.tokenization import CharTokenizer


class CTCLossWrapper(torch.nn.Module):
    def __init__(
        self, blank: int = 0, normalize: Literal["target", "batch"] = "target"
    ):
        super().__init__()
        self.normalize = normalize
        self.ctc = torch.nn.CTCLoss(
            blank=blank,
            reduction="sum",
            zero_infinity=True,  # to avoid NaNs when target longer than input
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = logits.float().log_softmax(
            dim=-1
        )  # cast to float to stay in fp32 even under autocast
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        if self.normalize == "target":
            denom = target_lengths.sum().clamp_min(1)
        elif self.normalize == "batch":
            denom = target_lengths.numel()
        else:
            raise ValueError(f"Unknown normalize={self.normalize!r}")
        return loss / denom


def greedy_ctc_decode(
    logits: torch.Tensor, input_lengths: torch.Tensor, tokenizer: CharTokenizer
) -> list[str]:
    blank = tokenizer.blank_index
    T, B, C = logits.shape

    probs = logits

    best_paths = probs.argmax(dim=-1)

    texts = []
    for b in range(B):
        t_max = int(input_lengths[b].item())
        path = best_paths[:t_max, b]
        collapsed = []
        prev = None
        for t in range(t_max):
            idx = int(path[t].item())
            if idx == blank:
                prev = None
                continue
            if prev is not None and idx == prev:
                # = repeated non-blank, skip
                continue
            collapsed.append(idx)
            prev = idx
        text = tokenizer.decode(collapsed)
        texts.append(text)
    return texts


def build_beam_decoder(
    tokenizer: CharTokenizer,
    beam_size: int = 20,
    nbest: int = 1,
    lm_path: Path | None = None,
    lm_weight: float = 0.5,
    word_score: float = 0.0,
    sil_score: float = 0.0,
):
    tokens = list(tokenizer.alphabet)
    blank = tokens[tokenizer.blank_index]

    # When using an LM, remap space to a placeholder so it is not confused
    # with the KenLM word delimiter.
    # Also generate a character-level lexicon so that flashlight uses its
    # LexiconDecoder, which supports word_score (LexiconFreeDecoder ignores it).
    sil_token = blank
    lexicon = None
    if lm_path is not None:
        from modular_htr.lm.utils import (
            SPACE_PLACEHOLDER,
            remap_tokens_for_lm,
            write_lexicon,
        )

        tokens = remap_tokens_for_lm(tokens)
        sil_token = SPACE_PLACEHOLDER

        # This is an idea from the current Pylaia implementation (maintained by Teklia):
        # --> Build a character-level lexicon. Each character is a "word" that
        # spells itself.  This makes word_score act as a per-character
        # insertion bonus, counteracting the LM's length penalty.
        non_blank_tokens = [t for t in tokens if t != blank]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".lexicon", delete=False, encoding="utf-8"
        )
        tmp.close()
        lexicon = str(write_lexicon(non_blank_tokens, Path(tmp.name)))

    return ctc_decoder(
        lexicon=lexicon,
        tokens=tokens,
        blank_token=blank,
        sil_token=sil_token,
        nbest=nbest,
        beam_size=beam_size,
        beam_size_token=min(beam_size, len(tokens)),
        lm=str(lm_path) if lm_path is not None else None,
        lm_weight=lm_weight,
        word_score=word_score,
        sil_score=sil_score,
    )


def beam_ctc_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    decoder,
    tokenizer: CharTokenizer,
    temperature: float = 1.0,
) -> list[str]:
    # Permute logits from [T, B, C] to [B, T, C] before passing.
    scores = logits.permute(1, 0, 2)
    if temperature != 1.0:
        scores = scores / temperature
    log_probs = scores.log_softmax(-1).cpu()
    lengths = input_lengths.cpu()
    results = decoder(log_probs, lengths)
    return [(tokenizer.decode(hyp[0].tokens)).strip() for hyp in results]
