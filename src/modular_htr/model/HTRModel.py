import enum
import logging
from abc import ABC, abstractmethod
from typing import Literal, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from modular_htr.types import HeightCollapseMode, NormType, SequenceEncoderType

from .ConvNeXt import ConvNeXtBackbone, load_pretrained_timm_weights
from .MHA import RelBiasTransformerStack

logger = logging.getLogger(__name__)


def get_norm(norm_type: NormType | str, num_channels: int) -> nn.Module:
    """Create normalization layer."""
    norm_type = NormType(norm_type)
    if norm_type == NormType.BATCH:
        return nn.BatchNorm2d(num_channels)
    elif norm_type == NormType.GROUP:
        num_groups = min(8, num_channels)
        while num_groups > 1 and (num_channels % num_groups) != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)
    raise ValueError(f"Unknown norm_type '{norm_type}'")


class ConvBlock(nn.Module):
    """Conv -> Norm -> Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int = 3,
        stride: int = 1,
        padding: tuple[int, int] | int = 1,
        norm_type: NormType | str = NormType.GROUP,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.norm = get_norm(norm_type, out_channels)
        self.act = (
            nn.GELU() if out_channels >= 128 else nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Pre-activation style residual block. Avoids inplace issues."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        use_se: bool = False,
        norm_type: NormType | str = NormType.GROUP,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = get_norm(norm_type, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        self.projection_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection_shortcut(x)
        out = self.act(self.norm1(x))
        out = self.conv1(out)
        out = self.act(self.norm2(out))
        out = self.conv2(out)
        out = self.se(out)
        out = self.dropout(out)
        return out + identity


class BottleneckResBlock(nn.Module):
    """Bottleneck residual block to save compute when channel count is high"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        reduction: int = 4,
        use_se: bool = False,
        norm_type: NormType | str = NormType.GROUP,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        mid = out_channels // reduction

        self.norm1 = get_norm(norm_type, in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)

        self.norm2 = get_norm(norm_type, mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)

        self.norm3 = get_norm(norm_type, mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)

        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        self.projection_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        nn.init.zeros_(self.conv3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection_shortcut(x)

        out = self.act(self.norm1(x))
        out = self.conv1(out)

        out = self.act(self.norm2(out))
        out = self.conv2(out)

        out = self.act(self.norm3(out))
        out = self.conv3(out)

        out = self.se(out)

        out = self.dropout(out)
        return out + identity


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CNNBackbone(nn.Module):
    """
    CNN feature extractor with configurable stages.
    Each stage: ConvBlock -> (ResidualBlock) -> (SE) -> (Pool)
    """

    def __init__(
        self,
        in_channels: int,
        stage_channels: list[int] | None = None,  # None to use defaults
        pool_kernels: list[tuple[int, int]] | None = None,  # None to use defaults
        blocks_per_stage: list[int] | None = None,
        norm_type: NormType | str = NormType.GROUP,
        stem_channels: int | None = 32,  # None to not use it
        stem_kernel: int = 7,
        use_resblock_stack: bool = True,
        use_se: bool = False,
        use_bottleneck_above: int | None = 256,  # None to never use it
        stage_dropout: list[float] | None = None,
        residual_dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_type = NormType(norm_type)

        stage_channels = stage_channels or [32, 32, 64, 64, 128, 128]
        pool_kernels = pool_kernels or [(2, 2), (2, 2), (2, 2), (2, 1)]
        blocks_per_stage = blocks_per_stage or [1] * len(stage_channels)

        if stage_dropout and len(stage_dropout) != len(stage_channels):
            raise ValueError(
                f"stage_dropout length ({len(stage_dropout)}) must match "
                f"conv_channels length ({len(stage_channels)})"
            )

        self.stage_channels = list(stage_channels)
        self.pool_kernels = [tuple(k) for k in pool_kernels]
        self.use_se = use_se
        self.use_bottleneck_above = use_bottleneck_above

        # Compute reduction factors
        self.time_reduction = 1
        self.height_reduction = 1
        for i, (k_h, k_w) in enumerate(self.pool_kernels):
            self.time_reduction *= k_w
            self.height_reduction *= k_h

        if stem_channels is not None:
            stem_pad = stem_kernel // 2
            stem_stride = (4, 2)
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    stem_channels,
                    stem_kernel,
                    stride=stem_stride,
                    padding=stem_pad,
                    bias=False,
                ),
                get_norm(norm_type, stem_channels),
                nn.GELU(),
            )
            in_ch = stem_channels

            self.height_reduction *= stem_stride[0]
            self.time_reduction *= stem_stride[1]

        else:
            self.stem = None
            in_ch = in_channels

        # Build stages
        self.stages = nn.ModuleList()

        for i, (out_ch, num_blocks) in enumerate(
            zip(self.stage_channels, blocks_per_stage)
        ):
            layers = []

            if use_resblock_stack:
                for b in range(num_blocks):
                    block_in = in_ch if b == 0 else out_ch
                    layers.append(
                        self._make_resblock(
                            block_in, out_ch, self.norm_type, residual_dropout
                        )
                    )
            else:
                layers.append(ConvBlock(in_ch, out_ch, norm_type=self.norm_type))
                for _ in range(num_blocks - 1):
                    layers.append(
                        self._make_resblock(
                            out_ch, out_ch, self.norm_type, residual_dropout
                        )
                    )

            if stage_dropout and i < len(stage_dropout) and stage_dropout[i] > 0:
                layers.append(nn.Dropout2d(stage_dropout[i]))

            if i < len(self.pool_kernels):
                k_h, k_w = self.pool_kernels[i]
                layers.append(nn.MaxPool2d(kernel_size=(k_h, k_w)))

            self.stages.append(nn.Sequential(*layers))
            in_ch = out_ch

        self.out_channels = self.stage_channels[-1]

    def _make_resblock(
        self, in_ch: int, out_ch: int, norm_type: NormType, dropout: float
    ) -> nn.Module:
        """Factory to create the appropriate resblock variant"""
        use_bottleneck = (
            self.use_bottleneck_above is not None
            and out_ch >= self.use_bottleneck_above
        )

        if use_bottleneck:
            return BottleneckResBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                use_se=self.use_se,
                norm_type=norm_type,
                dropout=dropout,
            )
        else:
            return ResidualBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                use_se=self.use_se,
                norm_type=norm_type,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stem is not None:
            x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class HeightCollapse(nn.Module, ABC):
    """Abstract base for height collapse strategies."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse height dimension: [B, C, H, W] -> [B, C, W]"""
        pass


class PoolHeightCollapse(HeightCollapse):
    """Collapse height via simple pooling, no learnable params."""

    def __init__(self, mode: Literal["mean", "max"] = "mean"):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, C, W]
        if self.mode == "mean":
            return x.mean(dim=2)
        else:
            return x.max(dim=2).values


class ConvHeightCollapse(HeightCollapse):
    """Collapse height via learned convolution."""

    def __init__(self, channels: int, height: int | None = None):
        super().__init__()
        self.channels = channels
        self.expected_height = height
        self.conv: nn.Conv2d | None = None

        # initialize eagerly if height is provided (compile-friendly)
        if height is not None:
            self._init_conv(height)

    def _init_conv(self, height: int) -> None:
        self.conv = nn.Conv2d(
            self.channels, self.channels, kernel_size=(height, 1), bias=True
        )
        self.expected_height = height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, _ = x.size()

        # Lazy initialization
        if self.conv is None:
            self._init_conv(h)
            assert self.conv is not None, "ConvHeightCollapse initialization failed"
            self.conv = self.conv.to(device=x.device)

        # Sanity check
        if h != self.expected_height:
            raise RuntimeError(
                f"Height mismatch: expected {self.expected_height}, got {h}. "
                "Input dimensions must be consistent."
            )

        return self.conv(x).squeeze(2)  # [B, C, 1, W] -> [B, C, W]


class AttentionHeightCollapse(HeightCollapse):
    """
    Collapse height via attention.
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Conv2d(channels, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = self.query(x)  # [B, 1, H, W]
        attn = self.dropout(attn)
        attn = F.softmax(attn / 1.3, dim=2)  # Normalize over height
        # TODO: expose the magic 1.3 temperature to be set by the caller and added to the config.

        out = (x * attn).sum(dim=2)  # [B, C, W]
        return out


def create_height_collapse(
    collapse_type: HeightCollapseMode | str,
    channels: int,
    height: int | None = None,
    dropout: float = 0.0,
) -> HeightCollapse:
    """Factory for height collapse strategies."""
    collapse_type = HeightCollapseMode(collapse_type)
    if collapse_type == HeightCollapseMode.CONV:
        return ConvHeightCollapse(channels, height)
    elif collapse_type == HeightCollapseMode.ATTENTION:
        return AttentionHeightCollapse(channels, dropout=dropout)
    elif collapse_type == HeightCollapseMode.MEAN:
        return PoolHeightCollapse("mean")
    elif collapse_type == HeightCollapseMode.MAX:
        return PoolHeightCollapse("max")
    raise ValueError(f"Unknown collapse type: {collapse_type}")


class TemporalConvBlock(nn.Module):
    """1D convolutions over time dimension with residual connection."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(
                        channels, channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.LeakyReLU(negative_slope=0.01),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return x + self.conv(x)  # Residual


class SequenceEncoder(nn.Module, ABC):
    """Abstract base for sequence encoders."""

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Output feature dimension."""
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode sequence.
        Input: [B, C, T] (batch first, from CNN)
        Lengths: [B] sequence lengths before padding
        Output: [T, B, output_size] (time first, for CTC)
        """
        pass


class LSTMEncoder(SequenceEncoder):
    """Bidirectional LSTM sequence encoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pack_sequences: bool = True,
    ):
        super().__init__()
        self.pack_sequences = pack_sequences
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def output_size(self) -> int:
        return 2 * self.hidden_size

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, C, T] -> [T, B, C]
        seq = x.permute(2, 0, 1)

        T, _B, _C = seq.shape

        if lengths is not None and self.pack_sequences:
            # lengths should already be int 64 on CPU (from collate)
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths, enforce_sorted=True
            )
            packed_out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, total_length=T)
        else:
            out, _ = self.rnn(seq)
        return out  # [T, B, 2*hidden]


class RelBiasTransformerEncoder(SequenceEncoder):
    """
    Transformer encoder using relative sinusoidal bias-only PE
    """

    _positions: torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int = 4,
        dim_feedforward: int | None = None,
        max_seq_len: int = 4096,
        max_rel: int = 256,
        rel_dim: int = 32,
    ):
        super().__init__()

        if hidden_size % nhead != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by nhead ({nhead})."
            )

        self._output_size = hidden_size
        dim_feedforward = dim_feedforward or 4 * hidden_size

        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )

        self.transformer = RelBiasTransformerStack(
            num_layers=num_layers,
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_rel=max_rel,
            rel_dim=rel_dim,
        )

        self.register_buffer(
            "_positions",
            torch.arange(max_seq_len, dtype=torch.long),
            persistent=False,
        )

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        seq = x.permute(0, 2, 1)
        seq = self.input_proj(seq)
        _B, T, _C = seq.shape

        mask = None
        if lengths is not None:
            if lengths.device != seq.device:
                lengths = lengths.to(seq.device, non_blocking=True)
            mask = self._positions[:T].unsqueeze(0) >= lengths.unsqueeze(
                1
            )  # [B,T] bool

        out = self.transformer(seq, src_key_padding_mask=mask)  # [B,T,C]
        return out.permute(1, 0, 2)  # [T,B,C] for CTC


def create_sequence_encoder(
    type: SequenceEncoderType | str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    pos_encoding_dropout: float,  # legacy configs
    **kwargs,
) -> SequenceEncoder:
    """Factory for sequence encoders."""
    # coerce to enum type
    type = SequenceEncoderType(type)
    if type == SequenceEncoderType.LSTM:
        return LSTMEncoder(input_size, hidden_size, num_layers, dropout)
    elif type == SequenceEncoderType.TRANSFORMER:
        return RelBiasTransformerEncoder(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
    raise ValueError(f"Unknown sequence encoder type: {type}")


def create_backbone(
    backbone_config: dict,
    in_channels: int,
) -> CNNBackbone | ConvNeXtBackbone:
    """Factory to create the appropriate backbone from a config dict.

    Dispatches on backbone_config["backbone_type"]:
      - "cnn" -> CNNBackbone
      - "convnext" -> ConvNeXtBackbone
    """
    backbone_type = backbone_config.get("backbone_type", "cnn")

    if backbone_type == "convnext":
        cfg = dict(backbone_config)
        cfg.pop("backbone_type", None)
        pretrained = cfg.pop("pretrained", None)
        backbone = ConvNeXtBackbone(in_channels=in_channels, **cfg)
        if pretrained:
            load_pretrained_timm_weights(backbone, pretrained)
        return backbone

    elif backbone_type == "cnn":
        cfg = dict(backbone_config)
        cfg.pop("backbone_type", None)
        norm_type = cfg.pop("norm_type", NormType.BATCH)
        residual_dropout = cfg.pop("residual_dropout", 0.0)
        stage_dropout = cfg.pop("stage_dropout", None)

        return CNNBackbone(
            in_channels=in_channels,
            stage_channels=cfg.get("conv_channels"),
            pool_kernels=cfg.get("pool_kernels"),
            blocks_per_stage=cfg.get("blocks_per_stage"),
            norm_type=norm_type,
            stem_channels=cfg.get("stem_channels") or None,
            stem_kernel=cfg.get("stem_kernel", 7),
            use_resblock_stack=cfg.get("use_resblock_stack", True),
            use_se=cfg.get("squeeze_excitation", False),
            use_bottleneck_above=cfg.get("use_bottleneck_above"),
            stage_dropout=stage_dropout,
            residual_dropout=residual_dropout,
        )

    raise ValueError(f"Unknown backbone_type: {backbone_type}")


def _plain_dict(d: dict) -> dict:
    """Convert a config dict to plain types safe for torch.save (weights_only)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _plain_dict(v)
        elif isinstance(v, enum.Enum):
            out[k] = v.value
        else:
            out[k] = v
    return out


class HTRModel(nn.Module):
    """
    Modular CNN-RNN architecture for CTC-based text recognition.
    CNN Backbone -> Height Collapse -> (CTC Shortcut) - > (Temporal Conv) -> Sequence Encoder -> FC
    Allows the creation of a residual backbone with optional CTC shortcut like the one used by Retsinas et al. in "Best practices for a handwritten text recognition system" (2022), combined with an LSTM or a multihead attention encoder like the one used by Diaz et al. "Rethinking Text Line Recognition models" (2021).
    Simpler architectures like the one used by J. Puigcerver in "Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?" (2017) are also configurable for comparison.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_config: dict,
        img_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 3,
        seq_encoder: SequenceEncoderType = SequenceEncoderType.LSTM,
        height_collapse: HeightCollapseMode = HeightCollapseMode.MEAN,
        temporal_convolution: bool = False,
        fixed_height: int | None = None,
        shortcut_ctc: bool = True,
        temporal_dropout: float = 0.0,
        height_attention_dropout: float = 0.0,
        pos_encoding_dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        shortcut_dropout: float = 0.0,
    ):
        super().__init__()

        # Store config
        backbone_for_config = _plain_dict(backbone_config)
        backbone_for_config.pop("pretrained", None)
        self._config = {
            "model_type": "HTRModel",
            "img_channels": img_channels,
            "num_classes": num_classes,
            "backbone": backbone_for_config,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "temporal_dropout": temporal_dropout,
            "height_attention_dropout": height_attention_dropout,
            "pos_encoding_dropout": pos_encoding_dropout,
            "encoder_dropout": encoder_dropout,
            "classifier_dropout": classifier_dropout,
            "shortcut_dropout": shortcut_dropout,
            "seq_encoder": seq_encoder.value,
            "height_collapse": height_collapse.value,
            "temporal_convolution": temporal_convolution,
            "fixed_height": fixed_height,
            "shortcut_ctc": shortcut_ctc,
        }

        # CNN backbone (dispatched via factory)
        self.backbone = create_backbone(
            backbone_config,
            in_channels=img_channels,
        )
        feature_size = self.backbone.out_channels

        # Height collapse
        collapsed_height = None
        if fixed_height is not None:
            collapsed_height = fixed_height // self.backbone.height_reduction
            if collapsed_height < 1:
                raise ValueError(
                    f"fixed_height={fixed_height} too small for pooling "
                    f"(reduction={self.backbone.height_reduction})"
                )

        self.height_collapse_layer = create_height_collapse(
            height_collapse,
            feature_size,
            collapsed_height,
            dropout=height_attention_dropout,
        )

        if shortcut_ctc:
            self.shortcut_dropout_layer = (
                nn.Dropout(shortcut_dropout) if shortcut_dropout > 0 else nn.Identity()
            )
            self.shortcut_head = nn.Conv1d(
                feature_size, num_classes, kernel_size=3, padding=1
            )

        # Temporal convolution
        self.temporal_conv_layer: TemporalConvBlock | None = None
        if temporal_convolution:
            self.temporal_conv_layer = TemporalConvBlock(
                feature_size, dropout=temporal_dropout
            )

        # Sequence encoder
        self.seq_encoder = create_sequence_encoder(
            type=seq_encoder,
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=encoder_dropout,
            pos_encoding_dropout=pos_encoding_dropout,
        )

        # Output projection
        self.dropout_layer = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(self.seq_encoder.output_size, num_classes)

        # Expose for external use
        self.time_reduction = self.backbone.time_reduction
        self.height_reduction = self.backbone.height_reduction

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """Map input widths (pixels) to output sequence lengths for CTC."""
        lengths = widths.to(dtype=torch.int64)
        lengths = lengths // self.backbone.time_reduction
        return lengths

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: Input images [B, C, H, W]
        Returns Logits [T, B, num_classes] for CTC loss, logits [same] for shortcut CTC loss
        """
        # CNN features: [B, C, H, W] -> [B, Cf, Hc, Wc]
        features = self.backbone(x)

        # Height collapse: [B, Cf, Hc, Wc] -> [B, Cf, Wc]
        features = self.height_collapse_layer(features)

        shortcut_logits = None
        if hasattr(self, "shortcut_head"):
            shortcut_logits = self.shortcut_dropout_layer(features)
            shortcut_logits = self.shortcut_head(shortcut_logits)
            shortcut_logits = shortcut_logits.permute(2, 0, 1)

        # Temporal conv: [B, Cf, T]
        if self.temporal_conv_layer is not None:
            features = self.temporal_conv_layer(features)

        seq_lengths = None
        if lengths is not None:
            actual_T = features.size(-1)
            seq_lengths = self.output_lengths(lengths).clamp(min=1, max=actual_T)

        # Sequence encoding: [B, Cf, T] -> [T, B, D]
        seq_out = self.seq_encoder(features, seq_lengths)

        # Output projection: [T, B, D] -> [T, B, num_classes]
        seq_out = self.dropout_layer(seq_out)
        logits = self.fc(seq_out)

        return logits, shortcut_logits

    def to_config(self) -> dict:
        """Export configuration for serialization."""
        config = self._config.copy()
        # Add computed values
        config["time_reduction"] = self.time_reduction
        config["height_reduction"] = self.height_reduction
        return config

    @classmethod
    def from_config(
        cls, config: dict, *, num_classes: int, fixed_height: int | None = None
    ) -> Self:
        """Create model from configuration dict."""
        config = config.copy()

        saved_height = config.get("fixed_height")
        if (
            fixed_height is not None
            and saved_height is not None
            and fixed_height != saved_height
        ):
            config["fixed_height"] = fixed_height
            logger.warning(
                "Model was trained with fixed height %s; rebuilding with %s.",
                saved_height,
                fixed_height,
            )
        if fixed_height is not None:
            config["fixed_height"] = fixed_height
        elif saved_height is not None:
            pass  # already in
        else:
            raise ValueError("Missing fixed_height (needs to be passed if not stored).")

        config["num_classes"] = num_classes
        # Remove computed values that aren't constructor args
        config.pop("time_reduction", None)
        config.pop("height_reduction", None)
        config.pop("model_type", None)
        # rewrap enums if they're strings
        config["seq_encoder"] = SequenceEncoderType(config["seq_encoder"])
        config["height_collapse"] = HeightCollapseMode(config["height_collapse"])
        # Config stores "backbone" but __init__ expects "backbone_config"
        if "backbone" in config:
            config["backbone_config"] = config.pop("backbone")
        return cls(**config)
