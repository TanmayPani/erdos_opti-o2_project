import torch
from torch import nn
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """Focal loss (Lin et al. 2017) for class-imbalanced binary classification, on raw
    class logits. `alpha` up-weights class 1 (the minority oxic pulse); `gamma` down-
    weights easy examples. `targets` are long class indices."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    p_t = torch.exp(-ce)
    alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


class FocalLoss(nn.Module):
    """`focal_loss` as a Module, so it can be passed as a skorch `criterion=`."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        return focal_loss(logits, targets, self.alpha, self.gamma)
