import enum


class Augmentation(str, enum.Enum):
    CPU = "cpu"
    GPU = "gpu"
    NONE = "none"


class NormType(str, enum.Enum):
    BATCH = "batch"
    GROUP = "group"


class SequenceEncoderType(str, enum.Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"


class HeightCollapseMode(str, enum.Enum):
    CONV = "conv"
    ATTENTION = "attention"
    MEAN = "mean"
    MAX = "max"


class CTCDecoderMode(str, enum.Enum):
    GREEDY = "greedy"
    BEAM = "beam"


class NewHeadInit(str, enum.Enum):
    COPY = "copy"
    RESET = "reset"


class AnnealStrategy(str, enum.Enum):
    COS = "cos"
    LINEAR = "linear"


class NewSymbolsInit(str, enum.Enum):
    ZERO = "zero"
    KAIMING = "kaiming"


class MonitorMetric(str, enum.Enum):
    CER = "cer"
    WER = "wer"
    VAL_LOSS = "val_loss"


class MetricMode(str, enum.Enum):
    MIN = "min"
    MAX = "max"


class DebugMode(str, enum.Enum):
    LOG = "log"
    PRINT = "print"
