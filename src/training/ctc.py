import torch
from torchaudio.models.decoder import ctc_decoder

from src.data.tokenization import CharTokenizer


class CTCLossWrapper(torch.nn.Module):
    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc = torch.nn.CTCLoss(
            blank=blank,
            reduction="mean",
            zero_infinity=True,  # to avoid NaNs when target longer than input
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = logits.log_softmax(dim=-1)
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


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


def build_beam_decoder(tokenizer: CharTokenizer, beam_size: int = 20, nbest: int = 1):
    tokens = list(tokenizer.alphabet)
    blank = tokens[tokenizer.blank_index]
    return ctc_decoder(
        lexicon=None,
        tokens=tokens,
        blank_token=blank,
        sil_token=blank,
        unk_word=None,
        nbest=nbest,
        beam_size=beam_size,
        beam_size_token=min(beam_size, len(tokens)),
    )


def beam_ctc_decode(
    logits: torch.Tensor, input_lengths: torch.Tensor, decoder, tokenizer: CharTokenizer
) -> list[str]:
    # permute logits from [T, B, C] to [B, T, C] before passing
    log_probs = logits.permute(1, 0, 2).log_softmax(-1).cpu()
    lengths = input_lengths.cpu()
    results = decoder(log_probs, lengths)
    return [tokenizer.decode(hyp[0].tokens) for hyp in results]
