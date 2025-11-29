import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(inplace=True),
    )


class CRNN(nn.Module):
    def __init__(self, img_channels, num_classes, rnn_layers=2):
        super().__init__()
        self.time_reduction = 4

        self.cnn = nn.Sequential(
            # Input: (B, 1, H, W)
            conv_block(img_channels, 64),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (H/2, W/2)
            conv_block(64, 128),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (H/4, W/4)
            conv_block(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (H/8, W/4)
            conv_block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (H/16, W/4)
            conv_block(256, 512),
            # Collapse height to 1
            nn.AdaptiveAvgPool2d((1, None)),  # (1, W/4)
        )
        cnn_out_channels = 512
        self.feature_size = cnn_out_channels

        self.rnn = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=256,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.dropout = nn.Dropout(0.2)

        # Linear projection to classes for CTC (needs 2*256 because bidirectionial)
        self.fc = nn.Linear(2 * 256, num_classes)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """Map image widths (post preprocessing, in pixels) to sequence lengths (T) for CTC"""
        return widths // self.time_reduction

    def forward(self, x):
        features = self.cnn(x)
        _B, _C, H, _W = features.size()
        assert H == 1
        features = features.squeeze(2)

        features = features.permute(2, 0, 1)

        rnn_out, _ = self.rnn(features)
        rnn_out = self.dropout(rnn_out)

        logits = self.fc(rnn_out)
        return logits
