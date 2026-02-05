from dataclasses import dataclass, field
from typing import Annotated
from pathlib import Path

import tyro

from src import types


@dataclass
class LocalDataset:
    # Path to a local HuggingFace datasets.dataset.
    path: Path
    # Name of the column that stores the labels.
    text_col: str = "text"
    # Name of the column that stores the line images.
    img_col: str = "image"


@dataclass
class HubDataset:
    # The name (username/dataset) of a HuggingFace Hub dataset
    name: str
    # Name of the column that stores the labels.
    text_col: str = "text"
    # Name of the column that stores the line images.
    img_col: str = "image"


type Dataset = Annotated[LocalDataset, tyro.conf.subcommand("local")] | Annotated[HubDataset, tyro.conf.subcommand("hub")]

@dataclass
class WidthFilters:
    # Filter unusually large images for memory efficiency.
    filter_large_images: bool = True
    # Percentile of images widths to use as threshold for filtering.
    width_percentile: float = 97.0
    # Ignore the percentile and use a specific width as threshold. 
    max_width: int | None = None
    # Filter samples which would not yield enough timesteps per character for CTC to work.
    enforce_ctc_width: bool = True
    # Margin to account for CTC blank characters in timesteps/chars ratio computations.
    ctc_margin: float = 1.1
    # The total reduction factor of the model along the time dimension.
    # Should be set according to the settings of the chosen CNN backbone.
    time_reduction_factor: int = 8


@dataclass
class Data:
    # The Hugging Face dataset to use for training. Can be local or hosted on HF hub.
    # PageXML datasets must first be compiled with the compile-dataset subcommand to be used.
    dataset: Dataset
    # Which unicode normalization to use. Set to None to disable.
    unicode_normalize: types.UnicodeForm | None = None
    # Height of a line image after resize.
    fixed_height: int = 96
    # Number of workers to spawn for dataset processing AND dataloading.
    num_workers: int = 24
    # Training batch size.
    batch_size: int = 32
    # Bucket batches by width to avoid excessive padding.
    use_bucketing: bool = True
    # Type of data augmentation to use. GPU uses the full pipeline, CPU disables the most expensive ones.
    augmentation: types.Augmentation = types.Augmentation.CPU
    # Configuration options for the width filters
    width_filters: WidthFilters = field(default_factory=WidthFilters)


@dataclass
class Checkpoint:
    # Set to False to disable computing the character/word error rates.
    compute_error_rates: bool = True
    # The metric to monitor when deciding which checkpoints to keep.
    # Will be forced to val_loss if compute_error_rates is set to False.
    monitor: types.MonitorMetric = types.MonitorMetric.CER
    # Use min if smaller is better for the chosen metric.
    metric_mode: types.MetricMode = types.MetricMode.MIN
    # Max number of checkpoints to keep.
    top_k: int = 3
    # Number of epochs between evals (those won't be considered for checkpointing).
    full_eval_interval: int = 1
    # Number of samples to print after eval for visual inspection. Zero to disable.
    print_samples: int = 0


@dataclass
class CTCDecoder:
    # The type of decoder to use.
    mode: types.CTCDecoderMode = types.CTCDecoderMode.GREEDY
    # Size of the beam (unused for mode=greedy).
    beam_size: int = 20
    # Path to a KenLM language model (only implemented for the beam decoder).
    lm_path: Path | None = None
    # Relative importance given to the LM (beam mode only).
    lm_alpha: float = 0.5


@dataclass
class Trainer:
    # Whether to use cuDNN benchmarking algorithms to speed up training.
    # Warning: considerably slows down the first epoch because of variable widths batches.
    # The training script will automatically try to mitigate by turning on binning of batch width to common multiples.
    torch_benchmark: bool = False
    # Number of times we step over the whole dataset.
    epochs: int = 20
    # Number of gradient accumulation steps before each optimizer update.
    accum_steps: int = 1
    # Maximum gradient norm for gradient clipping. Low values may help stabilize training.
    grad_clip_norm: float = 5.0
    # Enables Automatic Mixed Precision [AMP] casting in chosen regions to improve performance; will also enable gradient scaling to improve convergence.
    # Setting to true might make CTC training brittle.
    amp: bool = False
    # Use NHWC tensor layout in memory instead of NCHW to optimise CNN computations on modern GPUs. 
    channels_last: bool = True
    # Enables logging of per-batch diagnostics. Use "print" to emit to console, "log" to only write CSV, None to disable.
    debug: types.DebugMode | None = None
    # Disable the progress bars
    disable_pbars: bool = False

    checkpoint: Checkpoint = field(default_factory=Checkpoint)


@dataclass
class CRNNConfig:
    # Number of input image channels (e.g. 1 for grayscale).
    img_channels: int = 1
    # Number of recurrent layers stacked after the convolutional encoder.
    rnn_layers: int = 2
    # Hidden size of the recurrent layers.
    rnn_hidden: int = 320
    # Number of output channels for each convolutional layer in the encoder.
    conv_channels: list[int] = field(
        default_factory=lambda: [64, 128, 256, 256, 384, 512]
    )
    # Pooling kernel sizes for each conv block. This is what determines sequence length reduction along the time dimension.
    pool_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2), (2, 1), (2, 1)]
    )
    # Dropout probability applied to encoder features.
    dropout: float = 0.3

@dataclass
class DropoutConfig:
    """Dropout configuration for all model components"""

    # Dropout2d rate after CNN stages (spatial dropout).
    conv: float = 0.1
    # Dropout2d rate inside ResidualBlocks.
    residual: float = 0.1
    # Dropout rate in TemporalConvBlock.
    temporal: float = 0.15
    # Dropout rate on attention weights in the height collapse.
    height_attention: float = 0.15
    # Dropout rate after position encoding.
    pos_encoding: float = 0.1
    # Sequence encoder dropout rate (between layers if LSTM, internal if Transfomer).
    encoder: float = 0.1
    # Dropout rate before final FC layer.
    classifier: float = 0.1

@dataclass
class ModelConfig:
    # Number of input image channels (e.g. 1 for grayscale).
    img_channels: int = 1
    # Number of channels for the stem stage. Zero to skip that step.
    stem_channels: int = 32
    # Size of the kernel for the stem stage.
    stem_kernel: int = 7
    # Outputs channels for each cnn stage (lenght determines number of stages).
    conv_channels: list[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    # Pooling kernels (H, W) for each stage. Drives spatial reduction along height and time dimensions.
    pool_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2), (2, 2)]
    )
    # Whether to use residual blocks in the cnn stages.
    use_resblock_stack: bool = True
    # How many residual blocks get stacked at each stage.
    blocks_per_stage: list[int] = field(default_factory=lambda: [2, 4, 4])
    # Use bottlenecks residual blocks to save compute above the defined channel count.
    # None/0 to never/always use them.
    use_bottleneck_above: int | None = 256
    # Whether to use the squeeze-excitation mechanism in the convolution stages.
    squeeze_excitation: bool = False
    # How to collapse height after CNN
    height_collapse: types.HeightCollapseMode = types.HeightCollapseMode.MAX
    # Add residual 1D convolutions over time before the seq encoder
    temporal_convolution: bool = False
    # Whether to add a CTC shortcut head after the convolutional backbone to help training.
    shortcut_ctc: bool = True 
    # Type of the sequence encoder block.
    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.LSTM
    # Number of recurrent layers stacked after the convolutional encoder.
    num_layers: int = 3
    # Hidden size of the recurrent layers.
    hidden_size: int = 256
    # Dropout configuration.
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    # Normalization type. Group is more stable for small batches.
    norm_type: types.NormType = types.NormType.BATCH


@dataclass
class OptimAdamwConfig:
    # Main learning rate selection. Will be used as max_lr for the schedulers.
    lr: float = 6e-4
    # Regularization param. Weight decay encourages learning simpler interpolations by pushing the weights gradually towards zero.
    # Mind its interaction with batch normalization.
    weight_decay: float = 0.005


@dataclass
class Onecycle:
    # Annealing strategy for LR schedule
    anneal_strategy: types.AnnealStrategy = types.AnnealStrategy.COS
    # Fraction of total training where LR increases before annealing.
    pct_start: float = 0.1
    # Initial LR = max_lr / div_factor.
    div_factor: float = 10.0
    # Final LR = max_lr / final_div_factor.
    final_div_factor: float = 10.0
    # Update the LR every batch instead of every epoch.
    step_per_batch: tyro.conf.Fixed[bool] = True

    # epochs must be inferred from the trainer config
    # max_lr must be inferred from optim config


@dataclass
class Cosine:
    # Minimum learning rate reached at the end of cosine schedule.
    eta_min: float = 5e-5
    # Update the LR every epoch.
    step_per_batch: tyro.conf.Fixed[bool] = False
    # max_T must be inferred from the trainer config


@dataclass
class Strategy:
    # If true, will drop the symbols from the pretrained head that do not appear in the finetuning dataset.
    # Makes no difference if head_init_mode=="reset"
    drop_unused_symbols: bool = True
    # Whether or not the weights from the pretrained head are copied to the new one.
    head_init_mode: types.NewHeadInit = types.NewHeadInit.COPY
    # How to initialise the new weights.
    # Makes no difference if head_init_mode=="copy" and drop_unused_symbols=="true"
    new_symbols_init: types.NewSymbolsInit = types.NewSymbolsInit.KAIMING
    # The number of convolution stages (conv + pool) to freeze (zero to disable).
    freeze_n_stages: int = 0
    # Number of epochs after which we unfreeze the whole network (zero to keep frozen).
    unfreeze_epoch: int = 5

type Model = Annotated[CRNNConfig, tyro.conf.subcommand("vanilla")] | Annotated[ModelConfig, tyro.conf.subcommand("v2")]
type Scheduler = Cosine | Onecycle
type Config = Train | Finetune | AnalyseDataset | CompileDataset


@dataclass
class Train:
    """Configure and train a model from scratch."""
    # Descriptive name that will be appended to the date and time to create the run directory.
    run_name: str

    data: Data
    # The random seed for reproducible experiments.
    seed: int = 42
    # Directory where training runs artifacts are stored
    base_dir: str = "runs"

    model: Model = field(default_factory=CRNNConfig)
    optim: OptimAdamwConfig = field(default_factory=OptimAdamwConfig)
    # Choose a scheduler with its subcommand to see its relevant parameters and defaults.
    scheduler: Scheduler = field(default_factory=Cosine)

    trainer: Trainer = field(default_factory=Trainer)

    decoder: CTCDecoder = field(default_factory=CTCDecoder)


@dataclass
class Finetune:
    """Rebuild a model and reload weights from a checkpoint for finetuning."""
    # Descriptive name that will be appended to the date and time to create the run directory.
    run_name: str

    data: Data
    # Path to pretrained model checkpoint from which to restore tokenizer, architecture and weights.
    from_checkpoint: Path
    # The random seed for reproducible experiments.
    seed: int = 42
    # Directory where training runs artifacts are stored.
    base_dir: str = "runs"


    strategy: Strategy = field(default_factory=Strategy)
    optim: OptimAdamwConfig = field(default_factory=OptimAdamwConfig)
    # Choose a scheduler with its subcommand to see its relevant parameters and defaults.
    scheduler: Scheduler = field(default_factory=Onecycle)

    trainer: Trainer = field(default_factory=Trainer)

    decoder: CTCDecoder = field(default_factory=CTCDecoder)

@dataclass
class AnalyseDataset:
    """Compute and print statistics for a given dataset to help making informed decisions when choosing model architecture hyperparameters."""
    # The dataset to run the analysis on. Can be either local or pulled from HF Hub.
    dataset: Dataset
    # Margin to account for CTC blank character in timestep/chars computations.
    ctc_margin: float = 1.1
    # Height is used to compute width at a fixed aspect ratio.
    fixed_height: int = 128
    # Compute the ratios for the specified list of network stride/pooling.
    strides: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 12, 16])


@dataclass
class CompileDataset:
    """
    Create a compressed HuggingFace datasets.Dataset from page images and pageXML transcription files.
    This also splits the samples between a "train" and a "test" (10%) dataset.
    """
    # Path to the directory containing the pageXML files.
    xml_path: Annotated[Path, tyro.conf.arg(name="xml")]
    # Path to the directory containing the corresponding images.
    img_path: Annotated[Path, tyro.conf.arg(name="img")]
    # Directory where the resulting HuggingFace datasets.Dataset will be created.
    out_path: Annotated[Path, tyro.conf.arg(name="out")]


# helper for quick testing:
if __name__ == "__main__":
    import tyro
    from pprint import pprint
    config = tyro.cli(Config, compact_help=True, config=(tyro.conf.CascadeSubcommandArgs, tyro.conf.FlagConversionOff,)) # type: ignore
    pprint(config)
