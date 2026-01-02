import math
from abc import ABC, abstractmethod
from typing import Literal, Self
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DropoutConfig
from src.types import HeightCollapseMode, NormType, SequenceEncoderType


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
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: NormType | str = NormType.GROUP,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.norm = get_norm(norm_type, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Pre-activation style residual block. Avoids inplace issues."""

    def __init__(
        self,
        channels: int,
        norm_type: NormType | str = NormType.GROUP,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = get_norm(norm_type, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm_type, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(x))
        out = self.conv1(out)
        out = self.act(self.norm2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        return out + identity

class BottleneckResBlock(nn.Module):
    """Bottleneck residual block to save compute when channel count is high"""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        norm_type: NormType | str = NormType.GROUP,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid = channels // reduction

        self.norm1 = get_norm(norm_type, channels)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias = False)

        self.norm2 = get_norm(norm_type, mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)

        self.norm3 = get_norm(norm_type, mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)

        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.conv3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act(self.norm1(x))
        out = self.conv1(out)

        out = self.act(self.norm2(out))
        out = self.conv2(out)

        out = self.act(self.norm3(out))
        out = self.conv3(out)

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
            nn.ReLU(inplace=True),
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

    Each stage: ConvBlock -> ResidualBlock -> (optional SE) -> (optional Pool)
    """

    def __init__(
        self,
        in_channels: int,
        stage_channels: list[int] | None = None,
        pool_kernels: list[tuple[int, int]] | None = None,
        norm_type: NormType | str = NormType.GROUP,
        use_res_blocks: bool = True,
        use_se: bool = False,
        spatial_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ):
        super().__init__()

        stage_channels = stage_channels or [64, 128, 256, 256, 512, 512]
        pool_kernels = pool_kernels or [(2, 2), (2, 2), (2, 2), (2, 1)]

        self.stage_channels = list(stage_channels)
        self.pool_kernels = [tuple(k) for k in pool_kernels]
        self.use_se = use_se

        # Compute reduction factors
        self.time_reduction = 1
        self.height_reduction = 1
        for i, (k_h, k_w) in enumerate(self.pool_kernels):
            self.time_reduction *= k_w
            self.height_reduction *= k_h

        # Build stages
        self.stages = nn.ModuleList()
        in_ch = in_channels

        for i, out_ch in enumerate(self.stage_channels):
            layers: list[nn.Module] = [
                ConvBlock(in_ch, out_ch, norm_type=norm_type),
            ]
            if use_res_blocks and i > 2:
                if out_ch >= 256:
                    layers.append(BottleneckResBlock(out_ch, norm_type=norm_type, dropout=residual_dropout))
                else:
                    layers.append(ResidualBlock(out_ch, norm_type=norm_type, dropout=residual_dropout))

            if use_se:
                layers.append(SEBlock(out_ch))

            if spatial_dropout > 0:
                layers.append(nn.Dropout2d(spatial_dropout))

            if i < len(self.pool_kernels):
                k_h, k_w = self.pool_kernels[i]
                layers.append(nn.MaxPool2d(kernel_size=(k_h, k_w)))

            self.stages.append(nn.Sequential(*layers))
            in_ch = out_ch

        self.out_channels = self.stage_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Collapse height via simple average pooling, no learnable params"""

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
    Collapse height via attention. This is untested yet.
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Conv2d(channels, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = self.query(x)  # [B, 1, H, W]
        attn = F.softmax(attn, dim=2)  # Normalize over height
        attn = self.dropout(attn)
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


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    pe: torch.Tensor  # declare buffer type

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(SequenceEncoder):
    """Transformer encoder with positional encoding."""

    _positions: torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pos_encoding_dropout: float,
        nhead: int = 4,
        dim_feedforward: int | None = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        if hidden_size % nhead != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by nhead ({nhead}). "
                    f"Try hidden_size={nhead * (hidden_size // nhead)} or nhead={hidden_size // (hidden_size // nhead)}"
                )

        # projection from last conv channels to hidden size

        if input_size != hidden_size:
                self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            self.input_proj = nn.Identity()

        self._output_size = hidden_size
        dim_feedforward = dim_feedforward or 4 * hidden_size

        self.pos_encoding = SinusoidalPositionalEncoding(
            hidden_size, max_len=max_seq_len, dropout=pos_encoding_dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
            norm=nn.LayerNorm(hidden_size),
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

        # create a padding mask (ignore position where True)
        mask = None
        if lengths is not None:
            if lengths.device != seq.device:
                lengths = lengths.to(seq.device, non_blocking=True)
            mask = self._positions[:T].unsqueeze(0) >= lengths.unsqueeze(1)

        seq = self.pos_encoding(seq)
        out = self.transformer(seq, src_key_padding_mask=mask)  # [B, T, C]
        return out.permute(1, 0, 2)  # [T, B, C] for CTC


def create_sequence_encoder(
    type: SequenceEncoderType | str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    pos_encoding_dropout: float,
    **kwargs,
) -> SequenceEncoder:
    """Factory for sequence encoders."""
    # coerce to enum type
    type = SequenceEncoderType(type)
    if type == SequenceEncoderType.LSTM:
        return LSTMEncoder(input_size, hidden_size, num_layers, dropout)
    elif type == SequenceEncoderType.TRANSFORMER:
        return TransformerEncoder(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pos_encoding_dropout=pos_encoding_dropout,
            **kwargs,
        )
    raise ValueError(f"Unknown sequence encoder type: {type}")


class HTRModel(nn.Module):
    """
    Modular CNN-RNN architecture for CTC-based text recognition.
    CNN Backbone -> Height Collapse -> (Temporal Conv) -> Sequence Encoder -> FC
    """

    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        dropout: DropoutConfig,
        conv_channels: list[int] | None = None,
        pool_kernels: list[tuple[int, int]] | None = None,
        use_res_blocks: bool = True,
        hidden_size: int = 384,
        num_layers: int = 3,
        seq_encoder: SequenceEncoderType = SequenceEncoderType.LSTM,
        norm_type: NormType = NormType.GROUP,
        height_collapse: HeightCollapseMode = HeightCollapseMode.MEAN,
        temporal_convolution: bool = True,
        self_excitation: bool = False,
        input_height: int | None = None,
        shortcut_ctc: bool = True,
    ):
        super().__init__()
        self.dropout_conf = dropout

        # Store config
        self._config = {
            "model_type":"HTRModel",
            "img_channels": img_channels,
            "num_classes": num_classes,
            "conv_channels": conv_channels,
            "pool_kernels": pool_kernels,
            "use_res_blocks": use_res_blocks,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": asdict(self.dropout_conf),
            "seq_encoder": seq_encoder.value,
            "norm_type": norm_type.value,
            "height_collapse": height_collapse.value,
            "temporal_convolution": temporal_convolution,
            "self_excitation": self_excitation,
            "input_height": input_height,
            "shortcut_ctc": shortcut_ctc,
        }

        # CNN backbone
        self.backbone = CNNBackbone(
            in_channels=img_channels,
            stage_channels=conv_channels,
            pool_kernels=pool_kernels,
            use_res_blocks=use_res_blocks,
            norm_type=norm_type,
            use_se=self_excitation,
            spatial_dropout=self.dropout_conf.conv,
            residual_dropout=self.dropout_conf.residual,
        )
        feature_size = self.backbone.out_channels

        # Height collapse
        collapsed_height = None
        if input_height is not None:
            collapsed_height = input_height // self.backbone.height_reduction
            if collapsed_height < 1:
                raise ValueError(
                    f"input_height={input_height} too small for pooling "
                    f"(reduction={self.backbone.height_reduction})"
                )

        self.height_collapse_layer = create_height_collapse(
            height_collapse,
            feature_size,
            collapsed_height,
            dropout=self.dropout_conf.height_attention,
        )

        if shortcut_ctc:
            self.shortcut_head = nn.Conv1d(feature_size, num_classes, kernel_size=3, padding=1)

        # Temporal convolution
        self.temporal_conv_layer: TemporalConvBlock | None = None
        if temporal_convolution:
            self.temporal_conv_layer = TemporalConvBlock(
                feature_size, dropout=self.dropout_conf.temporal
            )

        # Sequence encoder
        self.seq_encoder = create_sequence_encoder(
            type=seq_encoder,
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_conf.encoder,
            pos_encoding_dropout=self.dropout_conf.pos_encoding,
        )

        # Output projection
        self.dropout_layer = nn.Dropout(self.dropout_conf.classifier)
        self.fc = nn.Linear(self.seq_encoder.output_size, num_classes)

        # Expose for external use
        self.time_reduction = self.backbone.time_reduction
        self.height_reduction = self.backbone.height_reduction

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """Map input widths (pixels) to output sequence lengths for CTC."""
        lengths = widths.to(dtype=torch.int64)
        for _, k_w in self.backbone.pool_kernels:
            lengths = lengths // k_w
        return lengths


    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: Input images [B, C, H, W]
        Returns Logits [T, B, num_classes] for CTC loss, Logits [same] for shortcut CTC loss
        """
        # CNN features: [B, C, H, W] -> [B, Cf, Hc, Wc]
        features = self.backbone(x)

        # Height collapse: [B, Cf, Hc, Wc] -> [B, Cf, Wc]
        features = self.height_collapse_layer(features)

        shortcut_logits = None
        if hasattr(self, 'shortcut_head'):
                   shortcut_logits = self.shortcut_head(features)
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
    def from_config(cls, config: dict, *, num_classes: int, input_height: int | None = None) -> Self:
        """Create model from configuration dict."""
        config = config.copy()
        saved_height = config.get("input_height")
        if input_height is not None and saved_height is not None and input_height != saved_height:
            config["input_height"] = input_height
            print(f"[WARNING] Model was trained with input height {saved_height}; rebuilding with {input_height}.")
        if input_height is not None:
            config["input_height"] = input_height
        elif saved_height is not None:
            pass # already in
        else:
            raise ValueError("Missing input_height (needs to be passed if not stored).")

        config["num_classes"] = num_classes
        # Remove computed values that aren't constructor args
        config.pop("time_reduction", None)
        config.pop("height_reduction", None)
        # rewrap enums if they're strings
        config["seq_encoder"] = SequenceEncoderType(config["seq_encoder"])
        config["norm_type"] = NormType(config["norm_type"])
        config["height_collapse"] = HeightCollapseMode(config["height_collapse"])
        # rebuild DropoutConfig if it's a dict
        if isinstance(config.get("dropout"), dict):
            config["dropout"] = DropoutConfig(**config["dropout"])
        return cls(**config)
