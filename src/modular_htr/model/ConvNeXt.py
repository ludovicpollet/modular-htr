"""
ConvNeXt backbone for HTR, adapted from "A ConvNet for the 2020s" (Liu et al. 2022).

The code in this file is not a direct copy but it is heavily inspired from the
"official" implementation, and also draws from the timm repository to simplify using
the pretrained weights: naming conventions match timm's implementation so that pretrained
weight loading reduces to a simple filtered load_state_dict (no key remapping).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Stochastic depth: per-sample dropout of the entire residual branch."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # Shape [B, 1, 1, 1, ...] to broadcast over all spatial dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor()
        return x * mask / keep


class ChannelLastLayerNorm(nn.LayerNorm):
    """LayerNorm for 4D [B, C, H, W] tensors, applied in channel-last format."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.permute(0, 2, 3, 1)
        input = super().forward(input)
        input = input.permute(0, 3, 1, 2)
        return input


class Mlp(nn.Module):
    """Two-layer MLP with GELU activation (matches timm naming)."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ConvNeXtBlock(nn.Module):
    """
    Single ConvNeXt block.

    DWConv -> LayerNorm -> MLP (expand -> GELU -> project) -> LayerScale -> DropPath

    Attribute names match timm: conv_dw, norm, mlp.fc1, mlp.fc2, gamma.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()
        mid = dim * expansion
        padding = kernel_size // 2

        self.conv_dw = nn.Conv2d(
            dim, dim, kernel_size, padding=padding, groups=dim, bias=True
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mid)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_dw(x)
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.gamma * x
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x)
        return residual + x


class ConvNeXtStage(nn.Module):
    """A ConvNeXt stage with optional downsample and a sequence of blocks.

    Named children match timm: ``downsample`` and ``blocks``.
    """

    def __init__(
        self,
        downsample: nn.Module | None,
        blocks: list[nn.Module],
    ):
        super().__init__()
        self.downsample = downsample or nn.Identity()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.blocks(x)


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt feature extractor for HTR.

    Stem (large-stride conv + LN) -> N stages of ConvNeXt blocks with
    inter-stage downsampling via LN + strided conv.

    Exposes `out_channels`, `time_reduction`, `height_reduction`, and `stages`
    (ModuleList) for compatibility with freeze_cnn_stages.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        blocks_per_stage: list[int],
        downsample_kernels: list[tuple[int, int]],
        stem_stride: tuple[int, int] = (4, 2),
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        num_stages = len(channels)
        if len(blocks_per_stage) != num_stages:
            raise ValueError(
                f"blocks_per_stage length ({len(blocks_per_stage)}) must match "
                f"channels length ({num_stages})"
            )
        if len(downsample_kernels) != num_stages - 1:
            raise ValueError(
                f"downsample_kernels length ({len(downsample_kernels)}) must be "
                f"channels length - 1 ({num_stages - 1})"
            )

        # Compute reduction factors
        self.height_reduction = stem_stride[0]
        self.time_reduction = stem_stride[1]
        for k_h, k_w in downsample_kernels:
            self.height_reduction *= k_h
            self.time_reduction *= k_w

        # Linearly increasing drop path rates across all blocks
        total_blocks = sum(blocks_per_stage)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Patchify stem: kernel_size == stride, no padding (non-overlapping patches)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=stem_stride,
                stride=stem_stride,
                padding=0,
                bias=True,
            ),
            ChannelLastLayerNorm(channels[0]),
        )

        # Build stages with named downsample + blocks (matches timm structure).
        # Stage 0 has no downsample (stem already projects to channels[0]).
        # Stages 1..N-1 start with a downsample layer.
        self.stages = nn.ModuleList()
        block_idx = 0

        for i in range(num_stages):
            downsample_layer: nn.Module | None = None

            # Downsample at the start of stages 1+
            if i > 0:
                ds_k = downsample_kernels[i - 1]
                downsample_layer = nn.Sequential(
                    ChannelLastLayerNorm(channels[i - 1]),
                    nn.Conv2d(
                        channels[i - 1],
                        channels[i],
                        kernel_size=ds_k,
                        stride=ds_k,
                    ),
                )

            # ConvNeXt blocks
            stage_blocks: list[nn.Module] = []
            for _ in range(blocks_per_stage[i]):
                stage_blocks.append(
                    ConvNeXtBlock(
                        dim=channels[i],
                        kernel_size=kernel_size,
                        expansion=expansion,
                        layer_scale_init=layer_scale_init,
                        drop_path=dp_rates[block_idx],
                    )
                )
                block_idx += 1

            self.stages.append(ConvNeXtStage(downsample_layer, stage_blocks))

        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


def load_pretrained_timm_weights(
    backbone: ConvNeXtBackbone,
    model_name: str,
) -> None:
    """
    Load pretrained weights from a timm ConvNeXt model into the backbone.

    Because the naming matches timm's, this is a direct key-match load with
    two adaptations:
    - RGB -> grayscale stem weight averaging
    - Symmetric -> asymmetric downsample kernel adaptation
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for loading the ImageNet pretrained weights. "
            "Install it with: uv sync --extra pretrained"
        )

    timm_model = timm.create_model(model_name, pretrained=True)
    src = timm_model.state_dict()
    our_sd = backbone.state_dict()
    new_sd: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, tensor in src.items():
        if key not in our_sd:
            skipped.append(key)
            continue

        target_shape = our_sd[key].shape

        # Adapt stem conv: RGB (3 input channels) -> grayscale (1 channel)
        if key == "stem.0.weight" and tensor.shape[1] != target_shape[1]:
            tensor = tensor.mean(dim=1, keepdim=True)

        # Adapt downsample conv kernels: (2,2) -> (2,1) if needed
        if (
            tensor.dim() == 4
            and tensor.shape != target_shape
            and tensor.shape[2] == target_shape[2]
            and tensor.shape[3] != target_shape[3]
        ):
            tensor = tensor.mean(dim=3, keepdim=True)

        if tensor.shape != target_shape:
            logger.warning(
                "Shape mismatch for %s: timm %s vs ours %s, skipping",
                key,
                tensor.shape,
                target_shape,
            )
            skipped.append(key)
            continue

        new_sd[key] = tensor

    # Load with strict=False so we can report what was missing
    missing = set(our_sd.keys()) - set(new_sd.keys())
    backbone.load_state_dict(new_sd, strict=False)

    if skipped:
        logger.info(
            "Skipped %d timm keys not in backbone (head, norm_pre, etc.)",
            len(skipped),
        )
    if missing:
        logger.warning(
            "Missing %d backbone keys after loading: %s",
            len(missing),
            list(missing)[:10],
        )
    else:
        logger.info(
            "Successfully loaded all %d pretrained weights from %s",
            len(new_sd),
            model_name,
        )
