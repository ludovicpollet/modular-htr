import logging
import shutil
import subprocess
from pathlib import Path

import datasets

from modular_htr.lm.utils import prepare_kenlm_training_text

logger = logging.getLogger(__name__)


def find_kenlm_binary(name: str) -> Path:
    """
    Locate a KenLM binary (``lmplz``, ``build_binary``, ``interpolate``) on PATH.

    Raises if the binary is not found.
    """
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find '{name}' on PATH. "
            "Install KenLM from https://github.com/kpu/kenlm and ensure "
            "the binaries (lmplz, build_binary, interpolate) are on your PATH."
        )
    return Path(path)


def build_kenlm(
    training_text_path: Path,
    output_path: Path,
    order: int = 6,
    intermediate: bool = False,
) -> Path:
    """
    Build a binary KenLM model from a training text file.

    Runs ``lmplz`` to estimate an ARPA model, then ``build_binary`` to
    convert it to the fast binary format.

    When *intermediate* is True, ``lmplz`` is invoked with
    ``--intermediate`` and the resulting directory is kept alongside the
    output so it can later be used for interpolation.
    """
    lmplz = find_kenlm_binary("lmplz")
    build_binary = find_kenlm_binary("build_binary")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arpa_path = output_path.with_suffix(".arpa")

    lmplz_cmd: list[str | Path] = [
        lmplz,
        "--order",
        str(order),
        "--text",
        str(training_text_path),
        "--arpa",
        str(arpa_path),
    ]
    if intermediate:
        intermediate_dir = output_path.with_name(output_path.stem + "_intermediate")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        lmplz_cmd.extend(["--intermediate", str(intermediate_dir)])

    logger.info("Running lmplz (order=%d) ...", order)
    subprocess.run([str(c) for c in lmplz_cmd], check=True)

    logger.info("Running build_binary ...")
    subprocess.run(
        [str(build_binary), str(arpa_path), str(output_path)],
        check=True,
    )
    logger.info("Built LM: %s", output_path)
    return output_path


def interpolate_kenlm(
    model_dirs: list[Path],
    weights: list[float],
    output_path: Path,
) -> Path:
    """
    Interpolate multiple KenLM models (log-linear) and produce a binary LM.

    Each entry in *model_dirs* should be a directory produced by
    ``lmplz --intermediate``.  *weights* are the interpolation weights
    (should sum to 1).
    """
    if len(model_dirs) != len(weights):
        raise ValueError(
            f"model_dirs ({len(model_dirs)}) and weights ({len(weights)}) "
            "must have the same length."
        )

    interpolate = find_kenlm_binary("interpolate")
    build_binary = find_kenlm_binary("build_binary")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arpa_path = output_path.with_suffix(".arpa")

    interp_cmd: list[str] = [str(interpolate)]
    for d, w in zip(model_dirs, weights):
        interp_cmd.extend(["-m", str(d), "-w", str(w)])
    interp_cmd.extend(["-o", str(arpa_path)])

    logger.info("Interpolating %d models ...", len(model_dirs))
    subprocess.run(interp_cmd, check=True)

    logger.info("Running build_binary ...")
    subprocess.run(
        [str(build_binary), str(arpa_path), str(output_path)],
        check=True,
    )
    logger.info("Built interpolated LM: %s", output_path)
    return output_path


def extract_texts_from_dataset(dataset_source) -> list[str]:
    """Load a HuggingFace dataset and return the text column values."""
    from modular_htr.training.utils import get_dataset

    ds = get_dataset(dataset_source.dataset)
    split_names = [s.strip() for s in dataset_source.splits.split(",")]

    if isinstance(ds, datasets.DatasetDict):
        parts = []
        for split in split_names:
            if split not in ds:
                available = list(ds.keys())
                raise ValueError(f"Split '{split}' not found. Available: {available}")
            parts.append(ds[split])
    else:
        parts = [ds]

    texts: list[str] = []
    text_col = dataset_source.text_col
    for part in parts:
        texts.extend(part[text_col])

    return texts


def load_texts_from_files(paths: list[Path]) -> list[str]:
    """Read lines from one or more text files."""
    texts: list[str] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    texts.append(line)
    return texts


def run_build_lm(cfg) -> None:
    """Build a character-level KenLM model from a ``BuildLM`` config."""
    import tempfile

    from modular_htr.config import DatasetLMSource, TextFileLMSource

    if isinstance(cfg.source, DatasetLMSource):
        texts = extract_texts_from_dataset(cfg.source)
    elif isinstance(cfg.source, TextFileLMSource):
        texts = load_texts_from_files(cfg.source.paths)
    else:
        raise ValueError(f"Unknown LM source type: {type(cfg.source)}")

    logger.info("Collected %d text samples.", len(texts))

    with tempfile.TemporaryDirectory() as tmpdir:
        training_text_path = Path(tmpdir) / "training_text.txt"
        prepare_kenlm_training_text(
            texts,
            training_text_path,
            cfg.unicode_normalize,
            cfg.strip_space_before_punctuation,
        )
        build_kenlm(
            training_text_path,
            output_path=cfg.output,
            order=cfg.order,
            intermediate=cfg.intermediate,
        )


def run_interpolate_lm(cfg) -> None:
    """Interpolate KenLM models from an ``InterpolateLM`` config."""
    interpolate_kenlm(cfg.model_dirs, cfg.weights, cfg.output)
