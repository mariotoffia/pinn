"""Collocation point sampling - recipe step 8.

Uniform random is a weak default. Latin hypercube (Raissi's choice) and Sobol give much better
coverage for the same count, and residual-based adaptive refinement (RAR) is often worth more
than a bigger network.

RESAMPLE EVERY ITERATION during the Adam phase. A fixed collocation set overfits the residual:
you drive the loss to zero on your points while the true residual elsewhere stays large.
FREEZE the set during the L-BFGS phase, or you poison the quasi-Newton curvature estimate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .device import DEVICE


def _to_torch(a: np.ndarray, requires_grad: bool) -> torch.Tensor:
    t = torch.as_tensor(a, dtype=torch.get_default_dtype(), device=DEVICE)
    return t.requires_grad_(requires_grad)


def uniform(n: int, bounds: list[tuple[float, float]], *, requires_grad: bool = True):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pts = lo + (hi - lo) * np.random.rand(n, len(bounds))
    return _to_torch(pts, requires_grad)


def latin_hypercube(n: int, bounds: list[tuple[float, float]], *, requires_grad: bool = True):
    """Stratified: each dimension is split into n equal bins, one sample per bin, shuffled."""
    d = len(bounds)
    cut = (np.arange(n)[:, None] + np.random.rand(n, d)) / n
    for j in range(d):
        np.random.shuffle(cut[:, j])
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return _to_torch(lo + (hi - lo) * cut, requires_grad)


def sobol(n: int, bounds: list[tuple[float, float]], *, requires_grad: bool = True, seed: int = 0):
    """Low-discrepancy quasi-random. Best coverage of the three; n is rounded up to a power of 2."""
    from scipy.stats import qmc

    m = max(1, int(np.ceil(np.log2(max(n, 1)))))
    pts = qmc.Sobol(d=len(bounds), scramble=True, seed=seed).random_base2(m)[:n]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return _to_torch(lo + (hi - lo) * pts, requires_grad)


def rar_refine(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: list[tuple[float, float]],
    n_add: int = 256,
    n_candidates: int = 20_000,
) -> torch.Tensor:
    """Residual-based adaptive refinement: add points where the residual is largest.

    Draw a large candidate pool, evaluate |r| on it, keep the worst `n_add`. This is what makes
    Burgers converge at viscosities where uniform sampling fails, and it is usually cheaper than
    widening the network. Compare DeepXDE's Burgers.py with Burgers_RAR.py - the diff is the lesson.
    """
    cand = uniform(n_candidates, bounds, requires_grad=True)
    r = residual_fn(cand).abs().detach().flatten()
    idx = torch.topk(r, k=min(n_add, r.numel())).indices
    return cand[idx].detach().requires_grad_(True)
