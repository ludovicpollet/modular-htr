from datasets import load_from_disk

from src.data.hf_dataset import build_hf_dataset
from src.data.tokenization import build_char_tokenizer, apply_ctc_tokenizer
from src.data.dataloaders import make_dataloaders

if __name__ == "__main__":

    #    build_hf_dataset("data/dg_31/page/", "data/dg_31", "data/hfds")

    ds = load_from_disk("data/hfds")

    tokenizer = build_char_tokenizer(ds, text_col="text")

    ds = apply_ctc_tokenizer(ds, tokenizer, text_col="text")

    train_loader, test_loader = make_dataloaders(
            ds,
            tokenizer=tokenizer,
            fixed_height=128,
            batch_size=32,
            num_workers=4,
            use_bucketing=True,
            )
    batch = next(iter(train_loader))
    print("images:", batch["images"].shape)         # [B, 1, H, W_max]
    print("labels:", batch["labels"].shape)        # [B, T_max]
    print("widths:", batch["widths"])              # [B]
    print("ids[0]:", batch["ids"][0])
    print("texts[0]:", batch["texts"][0])
