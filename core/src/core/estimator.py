import typing
import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from skorch import NeuralNetClassifier

from core.nn import Logistic, RocketEncoder


def _softmax_np(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


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
            criterion__weight=torch.as_tensor(
                cw, dtype=torch.float32, device=self.device
            )
        )
        return super().fit(X, y, **fit_params)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        ce_sum = super().get_loss(y_pred, y_true, X=X, training=training)
        w = self.module_.linear.weight
        return self.C * ce_sum + 0.5 * (w**2).sum()


# ---------------------------------------------------------------------------
# ROCKET (Dempster et al. 2020) — random convolutional kernel transform for the DL
# track. Kernel sampling stays in numpy; the transform is a torch/GPU reimplementation
# of the original numba loops (identical features: RAW input, max + ppv pooling, one
# channel per kernel). Torch expresses ROCKET as what it is — a bank of dilated 1-D
# convolutions — so the whole 10k-kernel transform is a handful of grouped `conv1d`
# calls
# ---------------------------------------------------------------------------


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
