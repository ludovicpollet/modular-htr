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


@dataclass
class HubDataset:
    # The name (username/dataset) of a HuggingFace Hub dataset
    name: str
    # Name of the column that stores the labels.
    text_col: str = "text"


type Dataset = Annotated[LocalDataset, tyro.conf.subcommand("local")] | Annotated[HubDataset, tyro.conf.subcommand("hub")]


@dataclass
class Data:
    dataset: Dataset
    # Height of a line image after resize.
    fixed_height: int = 96
    # Number of workers to spawn for dataset processing AND dataloading.
    num_workers: int = 24
    # Training batch size.
    batch_size: int = 32
    # Bucket batches by width to avoid excessive padding.
    use_bucketing: bool = True
    # Type of data augmentation to use.
    augmentation: types.Augmentation = types.Augmentation.CPU


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

    # Dropout2d after CNN stages (spatial dropout)
    conv: float = 0.1
    # Dropout2d inside ResidualBlocks
    residual: float = 0.1
    # Dropout in TemporalConvBlock
    temporal: float = 0.15
    # Dropout on attention weights in the height collapse
    height_attention: float = 0.15
    # Dropout after position encoding
    pos_encoding: float = 0.1
    # Sequence encoder dropout (between layers if LSTM, internal if Transfomer)
    encoder: float = 0.3
    # Dropout before final FC layer
    classifier: float = 0.3

@dataclass
class ModelConfig:
    # Number of input image channels (e.g. 1 for grayscale).
    img_channels: int = 1

    # Outputs channels for each cnn stage (lenght determines number of stages).
    conv_channels: list[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    # Pooling kernels (H, W) for each stage. Drives spatial reduction along height and time dimensions.
    pool_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2), (2, 2)]
    )
    # Whether to use residual blocks in the cnn stages.
    # Automatically uses bottleneck blocks for high channel count stages to save compute.
    use_res_blocks: bool = True
    # How to collapse height after CNN
    height_collapse: types.HeightCollapseMode = types.HeightCollapseMode.MEAN
    # Add residual 1D convolutions over time before the seq encoder
    temporal_convolution: bool = False
    # Type of the sequence encoder block
    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.TRANSFORMER
    # Number of recurrent layers stacked after the convolutional encoder.
    num_layers: int = 3
    # Hidden size of the recurrent layers.
    hidden_size: int = 256
    # Whether to use self excitation
    self_excitation: bool = False
    # Dropout configuration
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    # Normalization type. Group is more stable for small batches.
    norm_type: types.NormType = types.NormType.GROUP


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
type Config = Train | Finetune


@dataclass
class Train:
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


# helper for quick testing:
if __name__ == "__main__":
    import tyro
    from pprint import pprint
    config = tyro.cli(Config, compact_help=True, config=(tyro.conf.CascadeSubcommandArgs, tyro.conf.FlagConversionOff,)) # type: ignore
    pprint(config)
