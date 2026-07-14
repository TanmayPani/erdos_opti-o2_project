import torch
from torch import nn
import torch.nn.functional as F


class Logistic(torch.nn.Module):
    """Multinomial logistic regression as a Module — returns raw class **logits** (no
    terminal sigmoid/softmax) so it drops straight into a CrossEntropy/focal-loss +
    softmax training loop (`training.torch_fit_predict`). For `n_outputs` classes this is
    softmax regression; recover a probability with softmax (multiclass) at inference."""

    def __init__(self, n_outputs):
        super().__init__()
        self.linear = torch.nn.LazyLinear(n_outputs)

    def forward(self, x):
        return self.linear(x)


class _InceptionBlock(nn.Module):
    """Single Inception block: parallel 1D convs at multiple scales + maxpool shortcut."""

    def __init__(self, in_ch, n_filters=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, n_filters, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(in_ch, n_filters, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(in_ch, n_filters, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(in_ch, n_filters, kernel_size=7, padding=3, bias=False)
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(n_filters * 4)

    def forward(self, x):
        # x: (batch, in_ch, T)
        out = torch.cat(
            [self.conv1(x), self.conv3(x), self.conv5(x), self.pool_conv(x)], dim=1
        )
        return F.gelu(self.bn(out))


class InceptionTimeLite(nn.Module):
    """Compact InceptionTime for small-n multivariate time series."""

    def __init__(
        self, n_channels=5, n_classes=2, n_filters=32, n_blocks=3, dropout=0.3
    ):
        super().__init__()
        blocks = []
        in_ch = n_channels
        for _ in range(n_blocks):
            blocks.append(_InceptionBlock(in_ch, n_filters))
            in_ch = n_filters * 4  # 4 branches
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)
