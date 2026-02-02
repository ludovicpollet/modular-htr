import os
from functools import partial

import numpy as np
from tabulate import tabulate

from src.training.utils import get_dataset
import src.config


def get_map_params(dataset_len):
    """
    Compute reasonable num_proc and batch_size for dataset.map().
    There are no tests behing it, this is me purely guessing and trying to avoid large batches
    and multiprocessing for small datasets.
    """
    MAX_NUM_PROC = max((os.cpu_count() or 1) - 2, 1)
    MAX_BATCH_SIZE = 256
    MIN_SAMPLES_PER_PROC = 1000

    # Scale down processes for small datasets
    num_proc = min(MAX_NUM_PROC, max(1, dataset_len // MIN_SAMPLES_PER_PROC))
    # Aim for at least a few batches per process, but cap at MAX_BATCH_SIZE
    if num_proc == 1:
        batch_size = min(MAX_BATCH_SIZE, dataset_len)
    else:
        samples_per_proc = dataset_len // num_proc
        # try to find a decent number of batches per process
        batch_size = min(MAX_BATCH_SIZE, max(32, samples_per_proc // 4))

    return num_proc, batch_size


def compute_dimensions(batch, img_col: str, text_col: str, fixed_height: int):
    """Compute resized widths and text lengths for a batch."""
    widths = []
    lengths = []
    for img, txt in zip(batch[img_col], batch[text_col]):
        w, h = img.size
        widths.append(round(w * fixed_height / h))
        lengths.append(len(txt))
    return {"resized_width": widths, "text_len": lengths}


def percentile(arr, q):
    return float(np.percentile(arr, q))


def print_title(title: str, width: int):
    pad = max(0, width - len(title) - 2)
    left = pad // 2
    right = pad - left
    print(f"{'=' * left} {title} {'=' * right}")


def compute_ctc_stats(W, L, strides, margin, width_mask=None, blank_worst_case: bool = False):
    """Compute CTC feasibility stats for given strides."""
    if width_mask is None:
        width_mask = np.ones(len(W), dtype=bool)

    W = W[width_mask]
    L = L[width_mask]
    T_req = 2 * L - 1 if blank_worst_case else L


    rows = []
    for stride in strides:
        T = W // stride
        ratio = T / np.maximum(T_req, 1)
        rows.append(
            {
                "stride": stride,
                "feasible": (T >= T_req).mean() * 100,
                "with_margin": (T >= margin * T_req).mean() * 100,
                "ratio_p50": percentile(ratio, 50),
                "ratio_p10": percentile(ratio, 10),
                "ratio_p90": percentile(ratio, 90),
            }
        )
    return rows


def run_analysis(cfg: src.config.AnalyseDataset):
    PERCENTILES = [25, 50, 75, 90, 95, 99]
    SPLITS = ["train", "test"]
    #ds = load_from_disk(DATA_PATH)
    #ds = load_dataset("magistermilitum/Tridis")
    ds = get_dataset(cfg.dataset)

    compute_dims_callable = partial(
        compute_dimensions,
        img_col = cfg.dataset.img_col,
        text_col=cfg.dataset.text_col,
        fixed_height=cfg.fixed_height)
    
    


    for split in SPLITS:
        split_ds = ds[split]
        num_proc, batch_size = get_map_params(len(split_ds))


        tmp = split_ds.map( #type: ignore
            compute_dims_callable,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"Computing stats for {split}",
        )

        W = np.asarray(tmp["resized_width"], dtype=np.int32)
        L = np.asarray(tmp["text_len"], dtype=np.int32)

        # Drop empty labels
        valid = L > 0
        W, L = W[valid], L[valid]

        W_p95 = percentile(W, 95)
        mask_p95 = W <= W_p95

        # Summary table
        summary_rows = [
            ["resized_width_px"]
            + [int(W.min())]
            + [int(percentile(W, p)) for p in PERCENTILES]
            + [int(W.max())],
            ["text_len_chars"]
            + [int(L.min())]
            + [int(percentile(L, p)) for p in PERCENTILES]
            + [int(L.max())],
        ]
        summary = tabulate(
            summary_rows,
            headers=["metric", "min"] + [f"p{p}" for p in PERCENTILES] + ["max"],
            tablefmt="github",
        )

        # Stride analysis: all samples vs p95-filtered
        all_stats = compute_ctc_stats(W, L, cfg.strides, cfg.ctc_margin)
        p95_stats = compute_ctc_stats(W, L, cfg.strides, cfg.ctc_margin, mask_p95)

        stride_rows = [
            [
                s["stride"],
                s["feasible"],
                s["with_margin"],
                s["ratio_p50"],
                s["ratio_p10"],
                s["ratio_p90"],
                p["feasible"],
                p["with_margin"],
                p["ratio_p50"],
                p["ratio_p10"],
                p["ratio_p90"],
            ]
            for s, p in zip(all_stats, p95_stats)
        ]
        stride_table = tabulate(
            stride_rows,
            headers=[
                "stride",
                "feas%",
                f"m{cfg.ctc_margin}%",
                "r_p50",
                "r_p10",
                "r_p90",
                "feas@p95%",
                f"m{cfg.ctc_margin}@p95%",
                "r50@p95",
                "r10@p95",
                "r90@p95",
            ],
            floatfmt=".2f",
            tablefmt="github",
        )

        # Print
        table_width = max(
            len(summary.splitlines()[0]), len(stride_table.splitlines()[0])
        )
        print_title(f"{split.upper()}  (N={len(W):,}, H={cfg.fixed_height}, W_p95={int(W_p95)}px)", table_width)
        print(summary)
        print()
        print(stride_table)
        print()