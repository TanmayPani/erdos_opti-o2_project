import torch
from functools import partial

from torch import nn

_KERNEL_LENGTHS: tuple[int, ...] = (7, 9, 11)


@torch.no_grad()
def _get_kernel_inputs(
    num_channels, num_steps, num_kernels, kernel_lengths=None, generator=None
):
    channels = torch.randint(0, num_channels, (num_kernels,), generator=generator)

    _kernel_lengths = torch.as_tensor(
        kernel_lengths if kernel_lengths is not None else _KERNEL_LENGTHS,
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
            k: torch.zeros(self.n_kernels, dtype=torch.long, requires_grad=False)
            for k in ("channel", "length", "dilation", "padding")
        }
        for _i, _conv in enumerate(self.convs):
            _slot = getattr(self, f"slot_{_i}").detach().cpu()
            specs["channel"][_slot] = getattr(self, f"ch_{_i}").detach().cpu()
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
