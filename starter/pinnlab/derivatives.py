"""Autograd helpers for PDE residuals.

Two idioms exist and you should know both (chapter 03):

  Idiom A - `torch.autograd.grad` on a whole batch. Universal, used by nearly all PINN papers,
            and hides what is going on behind a `.sum()` trick.
  Idiom B - `torch.func` transforms on a SINGLE point, then `vmap`. You write the physics the way
            the PDE is written on paper, and vectorise mechanically.

This module gives you Idiom A (which is what the scripts use, because it reads closest to the
papers) plus a `laplacian` built on Idiom B so you can compare.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.func import jacfwd, jacrev, vmap


def d(y: torch.Tensor, x: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    """dy/dx for a batch, where y is (N,1) and x is (N,k) with requires_grad=True.

    The `.sum()` is not an approximation: because output row i depends only on input row i,
    d/dx_i sum_j y(x_j) == dy(x_i)/dx_i. It just lets one reverse sweep serve the whole batch.

    `create_graph=True` keeps the result differentiable - forget it and the parameter gradient
    of a residual loss is silently wrong.
    """
    (g,) = torch.autograd.grad(
        y.sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=False
    )
    return g


def grad_wrt(y: torch.Tensor, x: torch.Tensor, i: int, **kw) -> torch.Tensor:
    """Partial derivative of y with respect to input column i, shape (N,1)."""
    return d(y, x, **kw)[:, i : i + 1]


def laplacian(fn: Callable[[torch.Tensor], torch.Tensor], pts: torch.Tensor) -> torch.Tensor:
    """Sum of second derivatives, computed with torch.func (Idiom B).

    `fn` maps a single point of shape (k,) to a scalar. `jacfwd(jacrev(...))` is the Hessian;
    its trace is the Laplacian. Forward-over-reverse is the efficient order when the input
    dimension is small - which, in a PINN, it always is.
    """

    def _lap(z: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(jacfwd(jacrev(fn))(z)).sum()

    return vmap(_lap)(pts).reshape(-1, 1)


def finite_difference_second(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, h: float = 1e-5
) -> torch.Tensor:
    """Central second difference - your independent check on autograd.

    In float64 this should agree with `d(d(u,x),x)` to roughly 6 digits. If it does not, your
    graph is wrong, not the finite difference.
    """
    return (fn(x + h) - 2.0 * fn(x) + fn(x - h)) / (h * h)
