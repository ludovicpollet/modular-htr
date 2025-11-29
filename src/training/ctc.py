import torch

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
