"""Networks: a tanh MLP, Fourier features, and random weight factorisation.

Every choice here maps onto a numbered step of the recipe in chapter 10.

Why tanh and never ReLU: a PINN loss contains d2u/dx2, and ReLU's second derivative is zero
almost everywhere, which makes the physics term vacuous. You need an activation in C^2 with a
non-degenerate second derivative.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding - recipe step 2.

    gamma(x) = [cos(2*pi*B x), sin(2*pi*B x)],  B_ij ~ N(0, sigma^2)

    B is a FIXED random projection (a buffer, not a parameter). This is the direct antidote to
    spectral bias: it shifts the network's NTK eigen-directions into the frequency band you
    choose with sigma. Recommended sigma in [1, 10]; too small gives blurry predictions, too
    large gives salt-and-pepper artifacts.
    """

    def __init__(self, in_dim: int, n_features: int = 64, sigma: float = 2.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_features) * sigma)
        self.out_dim = 2 * n_features

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = 2.0 * math.pi * (z @ self.B)
        return torch.cat([torch.cos(p), torch.sin(p)], dim=-1)


class RWFLinear(nn.Module):
    """Linear layer with random weight factorisation - recipe step 3.

    w = exp(s) * v, with s ~ N(mu, sigma^2), training s and v separately.
    A drop-in replacement for nn.Linear. Consistently helpful, essentially free.
    Source: https://arxiv.org/abs/2210.01274
    """

    def __init__(self, in_dim: int, out_dim: int, mu: float = 1.0, sigma: float = 0.1):
        super().__init__()
        w = torch.empty(out_dim, in_dim)
        # Same Glorot-with-tanh-gain init as the plain layer, so RWF changes the OPTIMISATION
        # geometry and nothing else. Getting this wrong makes RWF look harmful in an ablation.
        nn.init.xavier_normal_(w, gain=nn.init.calculate_gain("tanh"))
        s = torch.randn(out_dim, 1) * sigma + mu
        self.s = nn.Parameter(s)
        self.v = nn.Parameter(w / torch.exp(s))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, torch.exp(self.s) * self.v, self.bias)


def _glorot_linear(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    # PyTorch's default is Kaiming, tuned for ReLU. On a tanh net that silently shrinks
    # activations layer by layer. calculate_gain('tanh') == 5/3.
    nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """Plain tanh MLP - recipe step 1. Width 128-512, depth 3-6 hidden layers.

    Do not scale this up out of instinct: Urban et al. (JCP 2025) find 2-3 hidden layers
    suffice once the optimiser is right, and PirateNets exists because deeper PINNs get WORSE.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 128,
        depth: int = 4,
        rwf: bool = False,
        activation: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        make = (lambda i, o: RWFLinear(i, o)) if rwf else _glorot_linear
        dims = [in_dim] + [width] * depth + [out_dim]
        layers: list[nn.Module] = []
        for k in range(len(dims) - 1):
            layers.append(make(dims[k], dims[k + 1]))
            if k < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PINN(nn.Module):
    """A PINN ansatz: optional Fourier embedding -> MLP -> optional hard-constraint transform.

    `hard` is a callable (inputs, raw_output) -> output. Use it to impose boundary or initial
    conditions EXACTLY (recipe step 4). Every constraint you hard-code is one loss term and one
    weight you no longer have to balance - the cheapest reliability win available.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 128,
        depth: int = 4,
        fourier: bool = False,
        fourier_features: int = 64,
        sigma: float = 2.0,
        rwf: bool = False,
        hard=None,
    ):
        super().__init__()
        self.embed = FourierFeatures(in_dim, fourier_features, sigma) if fourier else None
        body_in = self.embed.out_dim if self.embed is not None else in_dim
        self.body = MLP(body_in, out_dim, width, depth, rwf=rwf)
        self.hard = hard

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.embed(z) if self.embed is not None else z
        out = self.body(h)
        return self.hard(z, out) if self.hard is not None else out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
