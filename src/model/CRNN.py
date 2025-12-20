from typing import Self

import torch
import torch.nn as nn



def conv_block(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(inplace=True),
    )


class CRNN(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        rnn_layers: int = 2,
        conv_channels: list[int] | None = None,
        pool_kernels: list[tuple[int, int]] | None = None,
        rnn_hidden: int = 256,
        dropout: float = 0.2,
    ):
        """
        Minimal Pylaia-style CRNN model with configurable layers.
        """
        super().__init__()

        # store config for checkpointing
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        self.dropout_prob = dropout

        conv_channels = conv_channels or [64, 128, 256, 256, 512]
        pool_kernels = pool_kernels or [(2, 2), (2, 2), (2, 1), (2, 1)]
        self.conv_channels = list(conv_channels)
        self.pool_kernels = [tuple(k) for k in pool_kernels]

        layers: list[nn.Module] = []
        stages = []
        in_ch = self.img_channels
        time_reduction = 1

        for i, out_ch in enumerate(self.conv_channels):
            layers.append(conv_block(in_ch, out_ch))
            if i < len(self.pool_kernels):
                k_h, k_w = self.pool_kernels[i]
                layers.append(nn.MaxPool2d(kernel_size=(k_h, k_w)))
                stages.append(nn.Sequential(*layers)) # conv + pool grouped as a stage for (un)freezing
                layers = []
                time_reduction *= k_w
            in_ch = out_ch
        if layers: # leftover conv without pool
            stages.append(nn.Sequential(*layers))
       
        # Collapse height to 1, keep time dimension
        # This is its own stage
        stages.append(nn.AdaptiveAvgPool2d((1, None)))

        self.cnn_stages = nn.ModuleList(stages)
        self.cnn = nn.Sequential(*stages)
        self.feature_size = self.conv_channels[-1]
        self.time_reduction = time_reduction

        self.rnn = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.dropout_layer = nn.Dropout(self.dropout_prob)

        # Linear projection to classes for CTC (needs 2*rnn_hidden because bidirectional)
        self.fc = nn.Linear(2 * self.rnn_hidden, self.num_classes)

        # try to avoid full blank collapse at the beginning:
        # with torch.no_grad():
        #     self.fc.bias.zero_()
        #     self.fc.bias[0] = -5.0

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """Map image widths (post preprocessing, in pixels) to sequence lengths (T) for CTC"""
        return widths // self.time_reduction

    def forward(self, x, lengths=None):
        features = self.cnn(x)
        _B, _C, H, _W = features.size()
        assert H == 1
        features = features.squeeze(2)

        features = features.permute(2, 0, 1)

        rnn_out, _ = self.rnn(features)
        rnn_out = self.dropout_layer(rnn_out)

        logits = self.fc(rnn_out)
        return logits
    
    def to_config(self) -> dict:
        return {
            "model_type": "crnn",
            "img_channels": self.img_channels,
            "num_classes": self.num_classes,
            "rnn_layers": self.rnn_layers,
            "conv_channels": self.conv_channels,
            "pool_kernels": self.pool_kernels,
            "rnn_hidden": self.rnn_hidden,
            "dropout": self.dropout_prob,
            "time_reduction": self.time_reduction,
        }
    
    @classmethod
    def from_config(cls, config: dict, *, num_classes: int) -> Self:
        cfg = dict(config)

        cfg.pop("time_reduction", None)
        cfg.pop("model_type", None)

        cfg["num_classes"] = num_classes
        return cls(**cfg)
