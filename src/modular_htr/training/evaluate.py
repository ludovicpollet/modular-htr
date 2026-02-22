import json
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from tabulate import tabulate
from tqdm.auto import tqdm

from modular_htr import config
from modular_htr.config import HubDataset, LocalDataset, WidthFilters
from modular_htr.data.dataloaders import _preprocess_split, ctc_collate
from modular_htr.data.transforms import make_runtime_transform
from modular_htr.training.ctc import (
    beam_ctc_decode,
    build_beam_decoder,
    greedy_ctc_decode,
)
from modular_htr.training.finetune import load_pretrained_model
from modular_htr.training.metrics import (
    cer,
    confusion_counts,
    corpus_cer_normalized,
    format_confusion_table,
    normalize_lower,
    normalize_medieval,
    normalize_no_punctuation,
    per_sample_cer,
    wer,
)
from modular_htr.training.utils import get_device, get_single_dataset
from modular_htr.types import CTCDecoderMode

type ConfusionKey = tuple[str, str | None, str | None]


@dataclass(slots=True)
class EvalReport:
    checkpoint: str
    dataset: str
    split: str
    num_samples: int
    corpus_metrics: dict[str, float]
    distribution: dict[str, float]
    refs: list[str]
    hyps: list[str]
    sample_ids: list[str | None]
    sample_cers: list[float]
    ranked_indices: list[int]
    confusions: Counter[ConfusionKey] | None


def _dataset_label(dataset_cfg: config.EvalDataset) -> str:
    """Human-readable label for the dataset source."""
    if isinstance(dataset_cfg, LocalDataset):
        return str(dataset_cfg.path)
    if isinstance(dataset_cfg, HubDataset):
        return dataset_cfg.name
    return str(type(dataset_cfg).__name__)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _render_kv_table(rows: list[tuple[str, str]]) -> str:
    return tabulate(rows, headers=["Metric", "Value"], tablefmt="simple")


def _resolve_filter_config(cfg: config.Evaluate) -> WidthFilters:
    if cfg.use_width_filters:
        return cfg.data.width_filters
    return WidthFilters(filter_large_images=False, enforce_ctc_width=False)


def prepare_eval(
    cfg: config.Evaluate,
) -> tuple[torch.device, Any, Any, torch.utils.data.DataLoader, str, int]:
    device = get_device()
    model, tokenizer, _ckpt = load_pretrained_model(cfg.checkpoint, device=device)
    if tokenizer is None:
        raise ValueError("Checkpoint is missing a tokenizer.")

    fixed_height = model.to_config().get("fixed_height")
    if fixed_height is None:
        raise ValueError(
            "Could not determine fixed_height from checkpoint model config."
        )

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)  # type: ignore

    ds = get_single_dataset(cfg.data.dataset)
    if cfg.split not in ds:
        available = list(ds.keys())
        raise KeyError(
            f"Split '{cfg.split}' not found in dataset. Available: {available}"
        )

    split_ds = ds[cfg.split]
    split_size = len(split_ds)
    print(f"Evaluating on split '{cfg.split}': {len(split_ds)} samples")

    split_ds = _preprocess_split(
        split_ds,
        fixed_height=fixed_height,
        tokenizer=tokenizer,
        text_col=cfg.data.dataset.text_col,
        image_col=cfg.data.dataset.img_col,
        filter_config=_resolve_filter_config(cfg),
        split_name=cfg.split,
        num_proc=cfg.data.num_workers or 1,
    )

    transform = make_runtime_transform(augment=False, invert=cfg.data.invert_image)
    split_ds = split_ds.with_transform(transform)

    loader = torch.utils.data.DataLoader(
        split_ds,  # type: ignore[arg-type]
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=ctc_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type != "cpu",
    )
    dataset_label = _dataset_label(cfg.data.dataset)
    return device, model, tokenizer, loader, dataset_label, split_size


def run_inference(
    model,
    tokenizer,
    loader: torch.utils.data.DataLoader,
    decoder_cfg: config.CTCDecoder,
    device: torch.device,
) -> tuple[list[str], list[str], list[str | None]]:
    beam_decoder = None
    if decoder_cfg.mode is CTCDecoderMode.BEAM:
        beam_decoder = build_beam_decoder(
            tokenizer=tokenizer,
            beam_size=decoder_cfg.beam_size,
            lm_path=decoder_cfg.lm_path,
            lm_weight=decoder_cfg.lm_alpha,
            word_score=decoder_cfg.word_score,
            sil_score=decoder_cfg.sil_score,
        )

    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    sample_ids: list[str | None] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            batch = batch.to(device)
            input_lengths = model.output_lengths(batch.widths)
            logits, _ = model(batch.images)

            if decoder_cfg.mode is CTCDecoderMode.BEAM:
                decoded = beam_ctc_decode(
                    logits,
                    input_lengths,
                    beam_decoder,
                    tokenizer,
                    temperature=decoder_cfg.temperature,
                )
            else:
                decoded = greedy_ctc_decode(logits, input_lengths, tokenizer)

            refs.extend(batch.texts)
            hyps.extend(decoded)
            sample_ids.extend(batch.ids)

    return refs, hyps, sample_ids


def build_report(
    cfg: config.Evaluate,
    refs: list[str],
    hyps: list[str],
    sample_ids: list[str | None],
    dataset_label: str,
) -> EvalReport:
    n = len(refs)
    cer_score = cer(refs, hyps)
    wer_score = wer(refs, hyps)

    corpus_metrics: dict[str, float] = {
        "cer": cer_score,
        "wer": wer_score,
    }

    if cfg.case_insensitive_cer:
        ci_cer = corpus_cer_normalized(refs, hyps, normalize_lower)
        corpus_metrics["cer_case_insensitive"] = ci_cer

    if cfg.no_punctuation_cer:
        np_cer = corpus_cer_normalized(refs, hyps, normalize_no_punctuation)
        corpus_metrics["cer_no_punctuation"] = np_cer

    if cfg.medieval_cer:
        med_cer = corpus_cer_normalized(refs, hyps, normalize_medieval)
        corpus_metrics["cer_medieval"] = med_cer

    sample_cers = per_sample_cer(refs, hyps)
    perfect_count = sum(1 for c in sample_cers if c == 0.0)

    distribution: dict[str, float] = {}
    if n > 0:
        distribution["mean"] = statistics.mean(sample_cers)
        distribution["median"] = statistics.median(sample_cers)
        distribution["std"] = statistics.stdev(sample_cers) if n > 1 else 0.0
        sorted_cers = sorted(sample_cers)
        p95_idx = min(int(0.95 * n), n - 1)
        p99_idx = min(int(0.99 * n), n - 1)
        distribution["p95"] = sorted_cers[p95_idx]
        distribution["p99"] = sorted_cers[p99_idx]
        distribution["perfect_count"] = float(perfect_count)

    ranked = sorted(range(n), key=lambda i: sample_cers[i])
    confusions = confusion_counts(refs, hyps) if cfg.top_confusions > 0 else None

    return EvalReport(
        checkpoint=str(cfg.checkpoint),
        dataset=dataset_label,
        split=cfg.split,
        num_samples=n,
        corpus_metrics=corpus_metrics,
        distribution=distribution,
        refs=refs,
        hyps=hyps,
        sample_ids=sample_ids,
        sample_cers=sample_cers,
        ranked_indices=ranked,
        confusions=confusions,
    )


def render_plain_report(report: EvalReport, cfg: config.Evaluate) -> str:
    lines: list[str] = []
    n = report.num_samples

    lines.append("")
    lines.append("=" * 50)
    lines.append("  Evaluation Report")
    lines.append("=" * 50)
    lines.append(f"Checkpoint: {report.checkpoint}")
    lines.append(f"Dataset:    {report.dataset} ({report.split})")
    lines.append(f"Samples:    {n}")

    lines.append("")
    lines.append("--- Corpus Metrics ---")
    corpus_rows: list[tuple[str, str]] = [
        ("CER", _format_pct(report.corpus_metrics["cer"])),
    ]
    if cfg.case_insensitive_cer:
        corpus_rows.append(
            (
                "CER (case-insensitive)",
                _format_pct(report.corpus_metrics["cer_case_insensitive"]),
            )
        )
    if cfg.no_punctuation_cer:
        corpus_rows.append(
            (
                "CER (no punctuation)",
                _format_pct(report.corpus_metrics["cer_no_punctuation"]),
            )
        )
    if cfg.medieval_cer:
        corpus_rows.append(
            (
                "CER (medieval-norm)",
                _format_pct(report.corpus_metrics["cer_medieval"]),
            )
        )
    corpus_rows.append(("WER", _format_pct(report.corpus_metrics["wer"])))
    lines.append(_render_kv_table(corpus_rows))

    if report.distribution:
        lines.append("")
        lines.append("--- Per-Sample CER Distribution ---")
        distribution_rows = [
            ("Mean", _format_pct(report.distribution["mean"])),
            ("Median", _format_pct(report.distribution["median"])),
            ("Std", _format_pct(report.distribution["std"])),
            ("P95", _format_pct(report.distribution["p95"])),
            ("P99", _format_pct(report.distribution["p99"])),
        ]
        perfect_count = int(report.distribution.get("perfect_count", 0))
        distribution_rows.append(
            ("Perfect", f"{perfect_count} / {n} ({perfect_count / n * 100:.1f}%)")
        )
        lines.append(_render_kv_table(distribution_rows))

    if cfg.show_worst > 0 and n > 0:
        lines.append("")
        lines.append(f"--- Worst {cfg.show_worst} Samples ---")
        worst_indices = report.ranked_indices[-cfg.show_worst :][::-1]
        for idx in worst_indices:
            sid = report.sample_ids[idx]
            label = f"[{sid}]" if sid is not None else f"[#{idx}]"
            lines.append(f"{label}  CER={_format_pct(report.sample_cers[idx])}")
            lines.append(f'  ref: "{report.refs[idx]}"')
            lines.append(f'  hyp: "{report.hyps[idx]}"')

    if cfg.show_best > 0 and n > 0:
        lines.append("")
        lines.append(f"--- Best {cfg.show_best} Samples ---")
        best_indices = report.ranked_indices[: cfg.show_best]
        for idx in best_indices:
            sid = report.sample_ids[idx]
            label = f"[{sid}]" if sid is not None else f"[#{idx}]"
            lines.append(f"{label}  CER={_format_pct(report.sample_cers[idx])}")
            lines.append(f'  ref: "{report.refs[idx]}"')
            lines.append(f'  hyp: "{report.hyps[idx]}"')

    if report.confusions and cfg.top_confusions > 0:
        lines.append("")
        lines.append(f"--- Top {cfg.top_confusions} Character Confusions ---")
        lines.append(
            format_confusion_table(report.confusions, top_n=cfg.top_confusions)
        )

    return "\n".join(lines)


def write_report_json(report: EvalReport, cfg: config.Evaluate) -> None:
    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)

        per_sample_records = []
        for i in report.ranked_indices:
            per_sample_records.append(
                {
                    "id": report.sample_ids[i],
                    "index": i,
                    "cer": round(report.sample_cers[i], 6),
                    "ref": report.refs[i],
                    "hyp": report.hyps[i],
                }
            )

        confusion_records = []
        if report.confusions:
            for (op, ref_char, hyp_char), count in report.confusions.most_common(
                cfg.top_confusions
            ):
                confusion_records.append(
                    {
                        "op": op,
                        "ref": ref_char,
                        "hyp": hyp_char,
                        "count": count,
                    }
                )

        result = {
            "checkpoint": report.checkpoint,
            "dataset": report.dataset,
            "split": report.split,
            "num_samples": report.num_samples,
            "corpus_metrics": {
                k: round(v, 6) for k, v in report.corpus_metrics.items()
            },
            "distribution": {
                k: round(v, 6) if k != "perfect_count" else int(v)
                for k, v in report.distribution.items()
            },
            "per_sample": per_sample_records,
            "top_confusions": confusion_records,
        }

        cfg.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nResults written to {cfg.output_json}")


def run_evaluate(cfg: config.Evaluate) -> None:
    device, model, tokenizer, loader, dataset_label, split_size = prepare_eval(cfg)
    print(f"Device: {device}")
    print(f"Evaluating on split '{cfg.split}': {split_size} samples")

    refs, hyps, sample_ids = run_inference(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        decoder_cfg=cfg.decoder,
        device=device,
    )
    report = build_report(
        cfg=cfg,
        refs=refs,
        hyps=hyps,
        sample_ids=sample_ids,
        dataset_label=dataset_label,
    )
    print(render_plain_report(report, cfg))
    write_report_json(report, cfg)
