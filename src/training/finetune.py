import math
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass

import torch

from src.data.tokenization import CharTokenizer, build_char_tokenizer
from src.model.CRNN import CRNN
from src.types import NewHeadInit, NewSymbolsInit


@dataclass
class CharsetDiff:
    shared: list[str]
    only_pretrained: list[str]
    only_finetune: list[str]


def compare_charsets(pretrained: CharTokenizer, finetune: CharTokenizer) -> CharsetDiff:
    blank_tok = pretrained.alphabet[pretrained.blank_index]

    def strip_blank(tok: CharTokenizer) -> set[str]:
        return {
            c
            for i, c in enumerate(tok.alphabet)
            if c != blank_tok and i != tok.blank_index
        }

    base, new = strip_blank(pretrained), strip_blank(finetune)
    return CharsetDiff(
        shared=sorted(base & new),
        only_pretrained=sorted(base - new),
        only_finetune=sorted(new - base),
    )


def merge_tokenizers(
    pretrained: CharTokenizer,
    finetune: CharTokenizer,
    drop_unused: bool = False,
) -> CharTokenizer:
    # keep original ordering for shared chars; append unseen ones
    alphabet = list(pretrained.alphabet)
    blank_token = pretrained.alphabet[pretrained.blank_index]

    for ch in finetune.alphabet:
        if ch == blank_token or ch in pretrained.index:
            continue
        alphabet.append(ch)

    if drop_unused:
        keep = {blank_token, *finetune.index.keys()}
        alphabet = [ch for ch in alphabet if ch in keep]

    index = {c: i for i, c in enumerate(alphabet)}
    blank_index = index[blank_token]
    pad_index = index.get(alphabet[pretrained.pad_index], blank_index)
    return CharTokenizer(
        alphabet=alphabet, index=index, blank_index=blank_index, pad_index=pad_index
    )


def build_head(
    old_fc,
    old_tok,
    new_tok,
    device,
    head_init: NewHeadInit,
    new_class_init: NewSymbolsInit,
):
    new_fc = torch.nn.Linear(old_fc.in_features, len(new_tok), device=device)
    with torch.no_grad():
        torch.nn.init.zeros_(new_fc.bias)
        if new_class_init is NewSymbolsInit.KAIMING:
            torch.nn.init.kaiming_uniform(new_fc.weight, a=math.sqrt(5))
        else:  # ZERO
            new_fc.weight.zero_()

        if head_init is NewHeadInit.COPY:
            for ch, old_idx in old_tok.index.items():
                if ch not in new_tok.index:
                    continue
                new_idx = new_tok.index[ch]
                new_fc.weight[new_idx] = old_fc.weight[old_idx]
                new_fc.bias[new_idx] = old_fc.bias[old_idx]
        # the RESET case is handled by the zero init above
    return new_fc


def resize_output_layer(
    model: CRNN,
    old_tok: CharTokenizer,
    new_tok: CharTokenizer,
    head_init: NewHeadInit,
    new_class_init: NewSymbolsInit,
) -> CRNN:
    if len(old_tok) == len(new_tok) and head_init is NewHeadInit.COPY:
        return model

    device = next(model.parameters()).device  # keep new head on the same device
    new_fc = build_head(
        old_fc=model.fc,
        old_tok=old_tok,
        new_tok=new_tok,
        device=device,
        head_init=head_init,
        new_class_init=new_class_init,
    )

    model.fc = new_fc
    return model


def prepare_finetune_model(
    checkpoint_path: str | Path,
    finetune_ds,
    text_col: str = "text",
    device: torch.device | str = "cpu",
    drop_unused: bool = False,
    override_config: dict[str, Any] | None = None,
    head_init: NewHeadInit | str = NewHeadInit.COPY,
    new_class_init: NewSymbolsInit | str = NewSymbolsInit.KAIMING,
) -> tuple[CRNN, CharTokenizer, CharsetDiff]:
    """High level convenience to build a model for finetuning from a checkpoint and handle alphabet differences"""
    # coerce eventual strings arguments to enum types
    head_init = NewHeadInit(head_init)
    new_class_init = NewSymbolsInit(new_class_init)

    model, pretrained_tok, _ckpt = load_pretrained_model(
        checkpoint_path, device=device, override_config=override_config
    )
    if pretrained_tok is None:
        raise ValueError("Checkpoint is missing a tokenizer.")

    blank_token = pretrained_tok.alphabet[pretrained_tok.blank_index]
    ft_tok = build_char_tokenizer(
        finetune_ds, text_col=text_col, blank_token=blank_token
    )
    diff = compare_charsets(pretrained_tok, ft_tok)
    merged_tok = merge_tokenizers(pretrained_tok, ft_tok, drop_unused=drop_unused)
    model = resize_output_layer(
        model,
        pretrained_tok,
        merged_tok,
        head_init=head_init,
        new_class_init=new_class_init,
    )
    return model, merged_tok, diff


def load_checkpoint(
    path: str | Path, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=map_location)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Missing 'model_state_dict' in checkpoint")
    if "tokenizer" not in checkpoint:
        print(
            "Warning: Missing tokenizer in checkpoint, will only infer basic info from config"
        )

    return checkpoint


def restore_tokenizer_from_checkpoint(ckpt: dict[str, Any]) -> CharTokenizer | None:
    tok_dict = ckpt.get("tokenizer")
    if tok_dict is None:
        return None
    return CharTokenizer.from_dict(tok_dict)


def restore_model_from_checkpoint(
    checkpoint: dict[str, Any], device, override_config: dict[str, Any] | None = None
) -> tuple[CRNN, CharTokenizer | None]:
    model_cfg = checkpoint.get("model_config") or checkpoint["config"].get("model")
    if override_config is not None:
        model_cfg = override_config
    if model_cfg is None:
        raise ValueError("Checkpoint is missing model configuration.")

    tokenizer = restore_tokenizer_from_checkpoint(checkpoint)
    if tokenizer is None:
        print(
            "Warning: Checkpoint is missing a tokenizer. Will infer num_classes from model size."
        )

    try:
        conv_channels = model_cfg["conv_channels"]
        pool_kernels = [tuple(k) for k in model_cfg["pool_kernels"]]
        rnn_hidden = model_cfg["rnn_hidden"]
        dropout = model_cfg["dropout"]
        rnn_layers = model_cfg["rnn_layers"]
        img_channels = model_cfg["img_channels"]
    except KeyError as e:
        raise ValueError(f"Checkpoint model config is missing key: {e.args[0]}") from e

    num_classes = len(tokenizer) if tokenizer else model_cfg.get("num_classes")
    if num_classes is None:
        raise ValueError(
            "Could not determine number of output classes to restore model."
        )

    model = CRNN(
        img_channels=img_channels,
        num_classes=num_classes,
        rnn_layers=rnn_layers,
        conv_channels=conv_channels,
        pool_kernels=pool_kernels,
        rnn_hidden=rnn_hidden,
        dropout=dropout,
    )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        print("Could not load model")
        raise e
    model.to(device)
    model.eval()
    return model, tokenizer


def load_pretrained_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    override_config: dict[str, Any] | None = None,
) -> tuple[CRNN, CharTokenizer | None, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model, tokenizer = restore_model_from_checkpoint(
        checkpoint, device, override_config
    )
    return model, tokenizer, checkpoint


def freeze_cnn_stages(model: CRNN, num_stages: int) -> None:
    for stage in list(model.cnn_stages[:num_stages]):
        for p in stage.parameters():
            p.requires_grad = False


def unfreeze_all(model: CRNN) -> None:
    for p in model.parameters():
        p.requires_grad = True


def make_unfreeze_callback(unfreeze_epoch, num_stages) -> Callable:
    def on_epoch_start(epoch, model):
        if epoch != unfreeze_epoch:
            return
        if num_stages is None:
            unfreeze_all(model)
        else:
            for stage in model.cnn_stages[:num_stages]:
                for p in stage.parameters():
                    p.requires_grad = True

    return on_epoch_start
