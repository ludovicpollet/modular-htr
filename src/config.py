from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

import tyro


@dataclass
class Dataset:
    # The name of the dataset.
    name: str
    # Path to a HuggingFace datasets.dataset.
    path: Path
    # Name of the column that stores the labels.
    text_col: str = "text"


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


@dataclass
class Checkpoint:
    # The metric to monitor when deciding which checkpoints to keep.
    monitor: Literal["cer", "wer", "val_loss"] = "cer"
    # Use min if smaller is better for the chosen metric.
    metric_mode: Literal["min", "max"] = "min"
    # Max number of checkpoints to keep.
    top_k: int = 3
    # Number of epochs between evals (those won't be considered for checkpointing).
    full_eval_interval: int = 1
    # Number of samples to print after eval for visual inspection. Zero to disable. 
    print_samples: int = 3


@dataclass
class CTCDecoder:
    # The type of decoder to use.
    mode: Literal["greedy", "beam"] = "beam"
    # Size of the beam (unused for mode=greedy).
    beam_size: int = 20
    # Path to a KenLM language model (only implemented for the beam decoder).
    lm_path: Path | None = None
    # Relative importance given to the LM (beam mode only).
    lm_alpha: float = 0.5


@dataclass
class Trainer:
    # Whether or not to use cuDNN benchmarking algorithms to speed up training. 
    # Warning: considerably slows down the first epoch because of variable widths batches.
    # The training script will automatically try to mitigate by turning on binning of batch width to common multiples.
    torch_benchmark: bool = False
    # Number of times we step over the whole dataset.
    epochs: int = 20
    # Number of gradient accumulation steps before each optimizer update.
    accum_steps: int = 1
    # Maximum gradient norm for gradient clipping. Low values may help stabilize training.
    grad_clip_norm: float = 1.0

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
class OptimAdamwConfig:
    # Main learning rate selection. Will be used as max_lr for the schedulers.
    lr: float = 6e-4
    # Regularization param. Weight decay encourages learning simpler interpolations by pushing the weights gradually towards zero.
    # Mind its interaction with batch normalization.
    weight_decay: float = 1e-4


@dataclass
class Onecycle:
    # Annealing strategy for LR schedule
    anneal_strategy: Literal["cos", "linear"] = "cos"
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
    eta_min: float = 1e-5
    # Update the LR every epoch.
    step_per_batch: tyro.conf.Fixed[bool] = False
    # max_T must be inferred from the trainer config


@dataclass
class Strategy:
    # If true, will drop the symbols from the pretrained head that do not appear in the finetuning dataset.
    # Makes no difference if head_init_mode=="reset"
    drop_unused_symbols: bool = True
    # Whether or not the weights from the pretrained head are copied to the new one.
    head_init_mode: Literal["copy", "reset"] = "copy"
    # How to initialise the new weights.
    # Makes no difference if head_init_mode=="copy" and drop_unused_symbols=="true"
    new_symbols_init: Literal["zero", "kaiming"] = "kaiming"
    # The number of convolution stages (conv + pool) to freeze (zero to disable).
    freeze_n_stages: int = 0
    # Number of epochs after which we unfreeze the whole network (zero to keep frozen).
    unfreeze_epoch: int = 5

    
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

    model: CRNNConfig = field(default_factory=CRNNConfig)
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
    