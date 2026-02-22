from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from modular_htr import types


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


@dataclass
class DatasetSource:
    """A single HuggingFace Hub dataset source for multi-dataset training."""

    # HuggingFace Hub dataset name (e.g. "CATMuS/medieval").
    name: str
    # Name of the image column in this source.
    img_col: str = "image"
    # Name of the text column in this source.
    text_col: str = "text"
    # Name of the training split. None to skip (val-only source).
    train_split: str | None = "train"
    # Validation split: "auto" to detect (validation > val > test), None to skip,
    # or an explicit split name.
    val_split: str | None = "auto"


@dataclass
class MultiDataset:
    """Multiple HuggingFace Hub datasets, configured via a JSON file.

    The JSON file should contain an array of objects, each with at least a "name"
    field. Optional fields: img_col, text_col, train_split, val_split.
    After loading, all sources are normalized to canonical "image"/"text" columns.
    """

    # Path to a JSON file listing dataset sources.
    sources_file: Path
    # Canonical text column name after normalization (used by downstream pipeline).
    text_col: str = "text"
    # Canonical image column name after normalization (used by downstream pipeline).
    img_col: str = "image"


type Dataset = (
    Annotated[LocalDataset, tyro.conf.subcommand("local")]
    | Annotated[HubDataset, tyro.conf.subcommand("hub")]
    | Annotated[MultiDataset, tyro.conf.subcommand("multi")]
)

type EvalDataset = (
    Annotated[LocalDataset, tyro.conf.subcommand("local")]
    | Annotated[HubDataset, tyro.conf.subcommand("hub")]
)


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
    # Dataset source (local path or HuggingFace Hub).
    dataset: Dataset
    # Which unicode normalization to use. Set to None to disable.
    unicode_normalize: types.UnicodeForm | None = types.UnicodeForm.NFKC
    # Remove whitespace immediately before punctuation characters.
    strip_space_before_punctuation: bool = False
    # Number of workers to spawn for dataset processing AND dataloading.
    num_workers: int = 24
    # Training batch size.
    batch_size: int = 32
    # Bucket batches by width to avoid excessive padding.
    use_bucketing: bool = True
    # Type of data augmentation to use. GPU uses the full pipeline, CPU disables the most expensive ones.
    augmentation: types.Augmentation = types.Augmentation.GPU
    # Whether to invert the image (so that strokes are bright and background is dark).
    invert_image: bool = True
    # Configuration options for the width filters
    width_filters: WidthFilters = field(default_factory=WidthFilters)


@dataclass
class EvalData:
    # Dataset source for evaluation (local path or HuggingFace Hub).
    dataset: EvalDataset
    # Which unicode normalization to use. Set to None to disable.
    unicode_normalize: types.UnicodeForm | None = types.UnicodeForm.NFKC
    # Remove whitespace immediately before punctuation characters.
    strip_space_before_punctuation: bool = False
    # Number of workers to spawn for dataset processing AND dataloading.
    num_workers: int = 24
    # Training batch size.
    batch_size: int = 32
    # Bucket batches by width to avoid excessive padding.
    use_bucketing: bool = True
    # Type of data augmentation to use. GPU uses the full pipeline, CPU disables the most expensive ones.
    augmentation: types.Augmentation = types.Augmentation.GPU
    # Whether to invert the image (so that strokes are bright and background is dark).
    invert_image: bool = True
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
    lm_alpha: float = 1.5
    # Word insertion bonus (beam mode only).
    word_score: float = 0.8
    # Silence insertion score (beam mode only).
    sil_score: float = 0.0
    # Temperature for logit scaling before beam decoding (beam mode only).
    # Values > 1 flatten the distribution, giving the LM more influence.
    temperature: float = 2.0


@dataclass
class Trainer:
    # Whether to use cuDNN benchmarking algorithms to speed up training.
    # Warning: considerably slows down the first epoch because of variable widths batches.
    # The training script will automatically try to mitigate by turning on binning of batch width to common multiples.
    torch_benchmark: bool = False
    # Number of times we step over the whole dataset.
    epochs: int = 25
    # Number of gradient accumulation steps before each optimizer update.
    accum_steps: int = 1
    # Maximum gradient norm for gradient clipping. Low values may help stabilize training.
    grad_clip_norm: float = 3.0
    # Enables Automatic Mixed Precision [AMP] casting in chosen regions to improve performance; will also enable gradient scaling to improve convergence.
    # Setting to true might make CTC training brittle.
    amp: bool = True
    # Use NHWC tensor layout in memory instead of NCHW to optimise CNN computations on modern GPUs.
    channels_last: bool = True
    # Enables logging of per-batch diagnostics. Use "print" to emit to console, "log" to only write CSV, None to disable.
    debug: types.DebugMode | None = None
    # Disable the progress bars
    disable_pbars: bool = False

    checkpoint: Checkpoint = field(default_factory=Checkpoint)


# Backbone configs
# ----------------


@dataclass
class CNNBackboneConfig:
    """Fully modular CNN backbone. All parameters must be specified explicitly."""

    # Number of channels for the stem stage. Zero to skip.
    stem_channels: int
    # Size of the kernel for the stem stage.
    stem_kernel: int
    # Output channels for each CNN stage (length determines number of stages).
    conv_channels: list[int]
    # Pooling kernels (H, W) for each stage. Drives spatial reduction along height and time dimensions.
    pool_kernels: list[tuple[int, int]]
    # Whether to use residual blocks in the CNN stages.
    use_resblock_stack: bool
    # How many residual blocks get stacked at each stage.
    blocks_per_stage: list[int]
    # Use bottleneck residual blocks to save compute above this channel count.
    # None/0 to never/always use them.
    use_bottleneck_above: int | None
    # Whether to use the squeeze-excitation mechanism in the convolution stages.
    squeeze_excitation: bool
    # Normalization type. Group is more stable for small batches.
    norm_type: types.NormType
    # Per-stage Dropout2d rates applied after each CNN stage (before pooling).
    # Length must match conv_channels. Empty list disables stage dropout.
    stage_dropout: list[float] = field(default_factory=list)
    # Dropout2d rate inside ResidualBlocks.
    residual_dropout: float = 0.0
    backbone_type: tyro.conf.Fixed[str] = "cnn"


@dataclass
class PylaiaBackboneConfig(CNNBackboneConfig):
    """Puigcerver 2017: the paper configuration, simple Conv-Pool stack, no residual blocks, no stem."""

    stem_channels: int = 0
    stem_kernel: int = 7
    conv_channels: list[int] = field(default_factory=lambda: [16, 32, 48, 64, 80])
    pool_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2), (2, 2)]
    )
    use_resblock_stack: bool = False
    blocks_per_stage: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    use_bottleneck_above: int | None = None
    squeeze_excitation: bool = False
    norm_type: types.NormType = types.NormType.BATCH
    stage_dropout: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.2, 0.2, 0.2]
    )


@dataclass
class ModernPylaiaBackboneConfig(CNNBackboneConfig):
    """Teklia's current default Pylaia: smaller CNN."""

    stem_channels: int = 0
    stem_kernel: int = 7
    conv_channels: list[int] = field(default_factory=lambda: [16, 16, 32, 32])
    pool_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2), (2, 2)]
    )
    use_resblock_stack: bool = False
    blocks_per_stage: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    use_bottleneck_above: int | None = None
    squeeze_excitation: bool = False
    norm_type: types.NormType = types.NormType.BATCH


@dataclass
class KrakenBackboneConfig(CNNBackboneConfig):
    """Kraken default: small 2-stage CNN."""

    stem_channels: int = 0
    stem_kernel: int = 7
    conv_channels: list[int] = field(default_factory=lambda: [32, 64])
    pool_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2)]
    )
    use_resblock_stack: bool = False
    blocks_per_stage: list[int] = field(default_factory=lambda: [1, 1])
    use_bottleneck_above: int | None = None
    squeeze_excitation: bool = False
    norm_type: types.NormType = types.NormType.BATCH
    stage_dropout: list[float] = field(default_factory=lambda: [0.1, 0.1])


@dataclass
class RetsinasBackboneConfig(CNNBackboneConfig):
    """Retsinas et al. 2022: residual backbone with stem."""

    stem_channels: int = 32
    stem_kernel: int = 7
    conv_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    pool_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2)]
    )
    use_resblock_stack: bool = True
    blocks_per_stage: list[int] = field(default_factory=lambda: [2, 4, 3])
    use_bottleneck_above: int | None = 256
    squeeze_excitation: bool = False
    norm_type: types.NormType = types.NormType.BATCH
    residual_dropout: float = 0.0


@dataclass
class ConvNeXtBackboneConfig:
    """ConvNeXt backbone (Liu et al. 2022), modified for HTR."""

    backbone_type: tyro.conf.Fixed[str] = "convnext"
    # Output channels for each stage.
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    # Number of ConvNeXt blocks per stage.
    blocks_per_stage: list[int] = field(default_factory=lambda: [2, 4, 2])
    # Downsample kernels between stages (length = len(channels) - 1).
    downsample_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 2)]
    )
    # Patchify stem stride (H, W). Kernel size equals stride (non-overlapping patches).
    stem_stride: tuple[int, int] = (4, 2)
    # Depthwise conv kernel size in each block.
    kernel_size: int = 7
    # Inverted bottleneck expansion factor.
    expansion: int = 4
    # Initial value for layer scale gamma.
    layer_scale_init: float = 1e-6
    # Maximum stochastic depth rate (linearly increases across blocks).
    drop_path_rate: float = 0.1


@dataclass
class ConvNeXtTinyPretrainedConfig(ConvNeXtBackboneConfig):
    """ConvNeXt-Tiny with pretrained weights from timm.

    Uses symmetric stem stride (4,4) for direct weight transfer from ImageNet
    pretraining, with asymmetric downsamples to keep width reduction at 8x.

    Spatial reduction:
        Height: 4 (stem) x 2 x 2 x 2 = 32x
        Width:  4 (stem) x 2 x 1 x 1 = 8x
    """

    # timm model name to load pretrained weights from.
    pretrained: str = "convnext_tiny.fb_in22k_ft_in1k"
    # Fixed to match timm ConvNeXt-Tiny architecture.
    channels: tyro.conf.Fixed[list[int]] = field(
        default_factory=lambda: [96, 192, 384, 768]
    )
    blocks_per_stage: tyro.conf.Fixed[list[int]] = field(
        default_factory=lambda: [3, 3, 9, 3]
    )
    stem_stride: tyro.conf.Fixed[tuple[int, int]] = (4, 4)
    # Asymmetric downsamples: (2,2) for stage 1, (2,1) for stages 2-3.
    downsample_kernels: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 1), (2, 1)]
    )


type Backbone = (
    Annotated[PylaiaBackboneConfig, tyro.conf.subcommand("pylaia")]
    | Annotated[ModernPylaiaBackboneConfig, tyro.conf.subcommand("modern-pylaia")]
    | Annotated[KrakenBackboneConfig, tyro.conf.subcommand("kraken")]
    | Annotated[RetsinasBackboneConfig, tyro.conf.subcommand("retsinas")]
    | Annotated[CNNBackboneConfig, tyro.conf.subcommand("modular")]
    | Annotated[ConvNeXtBackboneConfig, tyro.conf.subcommand("convnext")]
    | Annotated[
        ConvNeXtTinyPretrainedConfig, tyro.conf.subcommand("convnext-tiny-pretrained")
    ]
)


# Model configs
# ----------------


@dataclass
class ModelConfig:
    """Fully modular model. All parameters must be specified explicitly."""

    # Height of input images after resize.
    fixed_height: int
    # Number of input image channels (e.g. 1 for grayscale).
    img_channels: int
    # CNN backbone architecture.
    backbone: Backbone
    # How to collapse height after CNN.
    height_collapse: types.HeightCollapseMode
    # Add residual 1D convolutions over time before the seq encoder.
    temporal_convolution: bool
    # Whether to add a CTC shortcut head after the convolutional backbone to help training.
    shortcut_ctc: bool
    # Type of the sequence encoder block.
    seq_encoder: types.SequenceEncoderType
    # Number of recurrent/transformer layers.
    num_layers: int
    # Hidden size of the recurrent/transformer layers.
    hidden_size: int
    # Dropout rate in TemporalConvBlock.
    temporal_dropout: float = 0.0
    # Dropout rate on attention weights in the height collapse.
    height_attention_dropout: float = 0.0
    # Dropout rate after position encoding.
    pos_encoding_dropout: float = 0.0
    # Sequence encoder dropout rate (between layers if LSTM, internal if Transformer).
    encoder_dropout: float = 0.0
    # Dropout rate before final FC layer.
    classifier_dropout: float = 0.0
    # Dropout rate before the shortcut CTC head.
    shortcut_dropout: float = 0.0


@dataclass
class PylaiaModelConfig(ModelConfig):
    """Puigcerver 2017: Pylaia CNN + biLSTM."""

    fixed_height: int = 128
    img_channels: int = 1
    backbone: Backbone = field(default_factory=PylaiaBackboneConfig)
    height_collapse: types.HeightCollapseMode = types.HeightCollapseMode.MEAN
    temporal_convolution: bool = False
    shortcut_ctc: bool = False
    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.LSTM
    num_layers: int = 5
    hidden_size: int = 256
    encoder_dropout: float = 0.5
    classifier_dropout: float = 0.5


@dataclass
class ModernPylaiaModelConfig(ModelConfig):
    """Teklia's current default: smaller CNN + 3-layer LSTM."""

    fixed_height: int = 128
    img_channels: int = 1
    backbone: Backbone = field(default_factory=ModernPylaiaBackboneConfig)
    height_collapse: types.HeightCollapseMode = types.HeightCollapseMode.MEAN
    temporal_convolution: bool = False
    shortcut_ctc: bool = False
    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.LSTM
    num_layers: int = 3
    hidden_size: int = 256
    encoder_dropout: float = 0.5
    classifier_dropout: float = 0.5


@dataclass
class KrakenModelConfig(ModelConfig):
    """
    Kraken default: small CNN + single LSTM.
    This is not the configuration they recommend for handwritten text, which is
    not implemented here (yet), because it uses tall kernels)
    """

    fixed_height: int = 48
    img_channels: int = 1
    backbone: Backbone = field(default_factory=KrakenBackboneConfig)
    height_collapse: types.HeightCollapseMode = types.HeightCollapseMode.MEAN
    temporal_convolution: bool = False
    shortcut_ctc: bool = False
    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.LSTM
    num_layers: int = 1
    hidden_size: int = 100
    classifier_dropout: float = 0.5


@dataclass
class RetsinasModelConfig(ModelConfig):
    """Retsinas et al. 2022: residual CNN + LSTM + shortcut CTC."""

    fixed_height: int = 128
    img_channels: int = 1
    backbone: Backbone = field(default_factory=RetsinasBackboneConfig)
    height_collapse: types.HeightCollapseMode = types.HeightCollapseMode.MAX
    temporal_convolution: bool = False
    shortcut_ctc: bool = True
    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.LSTM
    num_layers: int = 3
    hidden_size: int = 256
    encoder_dropout: float = 0.2
    classifier_dropout: float = 0.5
    shortcut_dropout: float = 0.5


@dataclass
class RetsinasMHAModelConfig(RetsinasModelConfig):
    """Retsinas backbone + Transformer encoder, no shortcut."""

    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.TRANSFORMER
    num_layers: int = 12
    shortcut_ctc: bool = False
    temporal_dropout: float = 0.15
    height_attention_dropout: float = 0.15
    pos_encoding_dropout: float = 0.1
    encoder_dropout: float = 0.15
    classifier_dropout: float = 0.2
    shortcut_dropout: float = 0.0


@dataclass
class ConvNeXtTinyPretrainedModelConfig(ModelConfig):
    """ConvNeXt-Tiny pretrained backbone + Transformer encoder."""

    fixed_height: int = 128
    img_channels: int = 1
    backbone: Backbone = field(default_factory=ConvNeXtTinyPretrainedConfig)
    height_collapse: types.HeightCollapseMode = types.HeightCollapseMode.MAX
    temporal_convolution: bool = False
    shortcut_ctc: bool = False
    seq_encoder: types.SequenceEncoderType = types.SequenceEncoderType.TRANSFORMER
    num_layers: int = 12
    hidden_size: int = 256
    pos_encoding_dropout: float = 0.1
    encoder_dropout: float = 0.2
    classifier_dropout: float = 0.3


type Model = (
    Annotated[ModelConfig, tyro.conf.subcommand("modular")]
    | Annotated[PylaiaModelConfig, tyro.conf.subcommand("pylaia-paper")]
    | Annotated[ModernPylaiaModelConfig, tyro.conf.subcommand("pylaia-modern")]
    | Annotated[KrakenModelConfig, tyro.conf.subcommand("kraken-small")]
    | Annotated[RetsinasModelConfig, tyro.conf.subcommand("retsinas")]
    | Annotated[RetsinasMHAModelConfig, tyro.conf.subcommand("retsinas-mha")]
    | Annotated[
        ConvNeXtTinyPretrainedModelConfig,
        tyro.conf.subcommand("convnext-tiny-pretrained"),
    ]
)


@dataclass
class OptimAdamwConfig:
    # Main learning rate selection. Will be used as max_lr for the schedulers.
    lr: float = 3e-4
    # Regularization param. Weight decay encourages learning simpler interpolations by pushing the weights gradually towards zero.
    # Mind its interaction with batch normalization.
    weight_decay: float = 5e-3


@dataclass
class Onecycle:
    # Annealing strategy for LR schedule
    anneal_strategy: types.AnnealStrategy = types.AnnealStrategy.COS
    # Fraction of total training where LR increases before annealing.
    pct_start: float = 0.05
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


type Scheduler = Cosine | Onecycle


# Language model configs
# ----------------------


@dataclass
class DatasetLMSource:
    """Extract transcriptions from a HuggingFace dataset."""

    dataset: Dataset
    # Which splits to extract text from (comma-separated, e.g. "train" or "train,test").
    splits: str = "train"
    # Name of the column that stores the text.
    text_col: str = "text"


@dataclass
class TextFileLMSource:
    """Plain text files (one line per sample)."""

    paths: list[Path]


type LMSource = (
    Annotated[DatasetLMSource, tyro.conf.subcommand("dataset")]
    | Annotated[TextFileLMSource, tyro.conf.subcommand("text")]
)


@dataclass
class BuildLM:
    """Build a character-level KenLM n-gram language model."""

    source: LMSource
    # Path for the output binary LM file.
    output: Path
    # N-gram order.
    order: int = 6
    # Unicode normalization applied to text before building the LM.
    unicode_normalize: types.UnicodeForm | None = types.UnicodeForm.NFKC
    # Remove whitespace immediately before punctuation characters.
    strip_space_before_punctuation: bool = False
    # Produce intermediate files needed for later interpolation with other LMs.
    intermediate: bool = False


@dataclass
class InterpolateLM:
    """Interpolate multiple KenLM models (log-linear).

    All models must have been built with --intermediate.
    """

    # Directories containing intermediate ARPA files from build-lm --intermediate.
    model_dirs: list[Path]
    # Interpolation weights (must sum to 1, same length as model_dirs).
    weights: list[float]
    # Output path for the interpolated binary LM file.
    output: Path


@dataclass
class Train:
    """Configure and train a model from scratch."""

    # Descriptive name that will be appended to the date and time to create the run directory.
    run_name: str

    data: Data
    # Model architecture. Use a named preset or model:modular for full control.
    model: Model
    # The random seed for reproducible experiments.
    seed: int = 42
    # Directory where training runs artifacts are stored.
    base_dir: str = "runs"

    optim: OptimAdamwConfig = field(default_factory=OptimAdamwConfig)
    # Learning rate scheduler.
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
class Evaluate:
    """Evaluate a trained model checkpoint on a dataset split."""

    # Path to the model checkpoint.
    checkpoint: Path
    data: EvalData
    # Which split to evaluate (e.g. "test", "validation", "train").
    split: str = "test"
    # Apply configured width/CTC filters before evaluation.
    use_width_filters: bool = False

    decoder: CTCDecoder = field(default_factory=CTCDecoder)
    # Number of best samples by CER to display (0 to disable).
    show_best: int = 5
    # Number of worst samples by CER to display (0 to disable).
    show_worst: int = 5
    # Number of top character confusions to display (0 to disable).
    top_confusions: int = 20
    # Compute medieval-normalized CER (v=u, j=i, t=c).
    medieval_cer: bool = True
    # Compute case-insensitive CER.
    case_insensitive_cer: bool = True
    # Compute CER ignoring punctuation and symbols.
    no_punctuation_cer: bool = True
    # Write full results to a JSON file.
    output_json: Path | None = None


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


type Config = (
    Train
    | Finetune
    | Evaluate
    | AnalyseDataset
    | CompileDataset
    | BuildLM
    | InterpolateLM
)
