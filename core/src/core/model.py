from functools import partial

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator, TransformerMixin
from skorch import NeuralNetClassifier


def _softmax_np(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


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


class LogitsNetClassifier(NeuralNetClassifier):
    """skorch classifier for a logits-output `nn.Module`: `predict_proba` softmaxes the
    raw logits so the net's probabilities feed the sklearn CV harness and ROC/PR metrics
    (the stock classifier assumes a log-prob output and would mis-scale them)."""

    def predict_proba(self, X):
        return _softmax_np(super().predict_proba(X))


class LogisticRegression(LogitsNetClassifier):
    """skorch net that reproduces sklearn `LogisticRegression(penalty='l2', C,
    class_weight='balanced')` — same objective, no hand-written loop:

      * balanced-class-weighted cross-entropy (`fit` sets `criterion__weight` per fold to
        `n / (n_classes * count_c)`, summed to match sklearn's Σᵢ);
      * L2 on the **weight matrix only** (intercept unpenalized), added in `get_loss` and
        scaled to sklearn's `0.5‖W‖² + C·Σᵢ sᵢ·CEᵢ`;
      * minimized by **LBFGS** (skorch drives the closure), sklearn's default solver.

    The module is the 2-output softmax `TorchLogReg`, so this matches sklearn's
    *multinomial* form — identical probabilities to the binary default, up to the L2 gauge."""

    def __init__(self, *args, C=0.5, **kwargs):
        super().__init__(Logistic, *args, **kwargs)
        self.C = C

    def fit(self, X, y, **fit_params):
        counts = np.bincount(np.asarray(y))
        cw = len(y) / (len(counts) * counts)  # class_weight='balanced'
        self.set_params(
            criterion__weight=torch.tensor(cw, dtype=torch.float32, device=self.device)
        )
        return super().fit(X, y, **fit_params)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        ce_sum = super().get_loss(y_pred, y_true, X=X, training=training)
        w = self.module_.linear.weight
        return self.C * ce_sum + 0.5 * (w**2).sum()


KERNEL_LENGTHS: tuple[int, ...] = (7, 9, 11)


@torch.no_grad()
def _get_kernel_inputs(
    num_channels, num_steps, num_kernels, kernel_lengths=None, generator=None
):
    channels = torch.randint(0, num_channels, (num_kernels,), generator=generator)

    _kernel_lengths = torch.as_tensor(
        kernel_lengths if kernel_lengths is not None else KERNEL_LENGTHS,
        dtype=torch.long,
    )
    _len_choices = torch.randint(
        0, len(_kernel_lengths), (num_kernels,), generator=generator
    )
    lengths = _kernel_lengths[_len_choices]

    _max_exp = ((num_steps - 1) / (lengths - 1)).log2_().floor_().clamp_(min=0).long()
    _exps = torch.rand(num_kernels, generator=generator).mul_(_max_exp + 1).long()
    dilations = torch.pow(2, _exps)  # long: integer power of two

    paddings = (lengths - 1).mul_(dilations).floor_divide_(2)
    _keep = torch.bernoulli(torch.full((num_kernels,), 0.5), generator=generator).long()
    paddings.mul_(_keep)

    return channels, torch.stack([lengths, dilations, paddings], dim=1)


@torch.no_grad()
def _init_one_rocket_kernel(conv, generator=None):
    conv.weight.normal_(generator=generator)
    conv.weight -= conv.weight.mean(dim=-1, keepdim=True)  # ... mean-centred per kernel
    conv.bias.uniform_(-1, 1, generator=generator)  # bias ~ U(-1, 1)


# ---------------------------------------------------------------------------
# ROCKET (Dempster et al. 2020) — random convolutional kernel transform for the DL
# track. Kernel sampling stays in numpy; the transform is a torch/GPU reimplementation
# of the original numba loops (identical features: RAW input, max + ppv pooling, one
# channel per kernel). Torch expresses ROCKET as what it is — a bank of dilated 1-D
# convolutions — so the whole 10k-kernel transform is a handful of grouped `conv1d`
# calls
# ---------------------------------------------------------------------------


class RocketEncoder(nn.Module):
    """Frozen-by-default ROCKET featurizer, sampled natively in torch. forward() is
    gradient-transparent; freezing is parameter state, so unfreeze() -> Tier 2 (learnable)."""

    def __init__(
        self,
        n_channels,
        n_steps,
        n_kernels=10_000,
        generator=None,
        kernel_length_opts=None,
    ):
        super().__init__()
        self.n_kernels = n_kernels
        # --- sample per-kernel specs ---
        channels, specs = _get_kernel_inputs(
            n_channels,
            n_steps,
            n_kernels,
            generator=generator,
            kernel_lengths=kernel_length_opts,
        )

        group_specs, group_indices, group_counts = specs.unique(
            dim=0, sorted=False, return_inverse=True, return_counts=True
        )

        # --- one grouped Conv1d per (L, d, p) bucket; weights sampled STRAIGHT into the params ---
        self.convs = nn.ModuleList()

        _kernel_idx = torch.arange(n_kernels)
        for _gr, _gr_spec in enumerate(group_specs):
            L, d, p = _gr_spec.tolist()
            _nch = int(group_counts[_gr])
            _conv1d = nn.Conv1d(_nch, _nch, L, dilation=d, padding=p, groups=_nch)
            _conv1d.apply(partial(_init_one_rocket_kernel, generator=generator))
            self.convs.append(_conv1d)
            _gr_kernel_idx = _kernel_idx[group_indices == _gr]
            self.register_buffer(f"ch_{_gr}", channels[_gr_kernel_idx])
            self.register_buffer(f"slot_{_gr}", _gr_kernel_idx)

        self.requires_grad_(False)  # frozen by default; call unfreeze() for Tier 2

    def freeze(self):
        self.requires_grad_(False)

    def unfreeze(self):  # Tier 2: kernels become learnable
        self.requires_grad_(True)

    @torch.no_grad()
    def kernel_specs(self):
        """Per-kernel (channel, length, dilation, padding) arrays indexed by kernel slot,
        for feature-importance interpretation (feature 2k -> max, 2k+1 -> ppv of kernel k)."""
        specs = {
            k: np.zeros(self.n_kernels, dtype=np.int64)
            for k in ("channel", "length", "dilation", "padding")
        }
        for _i, _conv in enumerate(self.convs):
            _slot = getattr(self, f"slot_{_i}").cpu().numpy()
            specs["channel"][_slot] = getattr(self, f"ch_{_i}").cpu().numpy()
            specs["length"][_slot] = _conv.kernel_size[0]
            specs["dilation"][_slot] = _conv.dilation[0]
            specs["padding"][_slot] = _conv.padding[0]
        return specs

    def forward(self, x):  # x: (N, C, T) -> (N, 2*n_kernels)
        x = torch.nan_to_num(x)
        out = x.new_zeros(x.shape[0], 2 * self.n_kernels)
        for _kr_idx, _kr in enumerate(self.convs):
            _ch, _slot = (
                getattr(self, f"ch_{_kr_idx}"),
                getattr(self, f"slot_{_kr_idx}"),
            )
            _c = _kr(x[:, _ch, :])
            out[:, _slot * 2] = _c.amax(-1)  # max
            out[:, _slot * 2 + 1] = (
                (_c > 0).float().mean(-1)
            )  # ppv (soft-surrogate if trainable)
        return out


class RocketTransform(BaseEstimator, TransformerMixin):
    """sklearn-style featurizer: sample a frozen ROCKET bank in fit(), return the
    (N, 2*n_kernels) [max, ppv] matrix in transform(). The inference context lives HERE,
    not in the Module, so the same Module stays usable in a training loop if unfrozen."""

    def __init__(self, n_kernels=10_000, seed=42, device=None):
        self.n_kernels, self.seed, self.device = n_kernels, seed, device

    def fit(self, X, y=None):
        X = np.asarray(X)
        _, C, T = X.shape
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        _g = torch.Generator().manual_seed(self.seed)
        self.module_ = (
            RocketEncoder(C, T, self.n_kernels, generator=_g).to(self.device_).eval()
        )  # .eval(): hygiene
        return self

    @torch.no_grad()
    def transform(self, X):
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device_)
        out = self.module_(x)
        return np.nan_to_num(out.detach().cpu().numpy())


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


class ChannelScaler(BaseEstimator, TransformerMixin):
    """Per-channel z-normalization for 3-D `(N, C, T)` sequence batches, fit on TRAIN-fold
    statistics — the leakage-safe, 3-D analogue of `StandardScaler` (which only handles
    2-D) for the sequence nets in an sklearn `Pipeline`. NaN-aware: stats use `nanmean`/
    `nanstd` (so a channel missing on some units — e.g. precip in WY2024-25 — keeps its
    signal on the units that have it) and missing entries scrub to the post-norm channel
    mean (0), matching `RocketTransform`'s `nan_to_num`. Emits float32 for the torch models."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = np.nanmean(X, axis=(0, 2), keepdims=True)
        self.sd_ = np.nanstd(X, axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num((X - self.mu_) / self.sd_).astype(np.float32)


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
