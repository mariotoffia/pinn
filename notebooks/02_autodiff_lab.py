# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy>=1.26",
#     "torch>=2.4",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Lab 02 — grad(grad(u)): the hinge exercise

    **What this lab does.** You compute `∂u/∂x` and `∂²u/∂x²` of a real tanh network by
    automatic differentiation, assemble the **Burgers residual** at thousands of random
    points, and verify everything two independent ways (finite differences and
    `gradgradcheck`). This is the exercise Chapter 03 calls "the one that makes the whole
    path click."

    **How this lab works.** Every cell below already contains a working solution, so the
    notebook always runs — *you cannot get stuck*. The tasks ask you to first try writing
    the marked lines yourself: open the hint, write your version, and compare. marimo
    re-runs everything reactively when you move a slider.

    **Foundations needed:** Chapter 03 §3.1–3.4 (the mental shift, the two idioms).

    **Success looks like:** finite differences agree with autodiff to ~1e-6, the two
    idioms agree to ~1e-15, and `gradgradcheck` returns `True`.
    """
    )
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(512, 8192, step=512, value=4096, label="collocation points N")
    log_h = mo.ui.slider(-7, -3, step=0.5, value=-5.0, label="log10(h) for the FD check")
    mo.hstack([n_points, log_h], justify="start", gap=2)
    return log_h, n_points


@app.cell
def _():
    import math

    import numpy as np
    import torch

    torch.set_default_dtype(torch.float64)   # non-negotiable for PINNs (Chapter 04)
    torch.manual_seed(0)
    NU = 0.01 / math.pi
    return NU, math, np, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The network

    A plain 3-hidden-layer tanh MLP mapping `(x, t) → u`. Glorot init with the tanh gain —
    the right default for PINNs (Chapter 05 §5.3).
    """
    )
    return


@app.cell
def _(torch):
    def make_net(width=64, seed=0):
        torch.manual_seed(seed)
        layers = []
        dims = [2, width, width, width, 1]
        for i in range(len(dims) - 1):
            lin = torch.nn.Linear(dims[i], dims[i + 1])
            torch.nn.init.xavier_normal_(lin.weight, gain=5.0 / 3.0)
            torch.nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(dims) - 2:
                layers.append(torch.nn.Tanh())
        return torch.nn.Sequential(*layers)

    net = make_net()
    n_params = sum(p.numel() for p in net.parameters())
    return make_net, n_params, net


@app.cell
def _(mo, n_params):
    mo.md(
        rf"""
    The network has **{n_params:,} parameters**. Tiny — and that is typical: PINNs are small.

    ## Task 1 — Idiom A: batched `autograd.grad`

    Compute `u`, `u_x`, `u_xx`, `u_t` at N random points and assemble the Burgers residual

    `r = u_t + u·u_x − ν·u_xx`

    **Try it yourself first** (open the hints below), then compare with the solution cell
    that follows — the notebook runs it either way.
    """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Hint 1 — the setup": mo.md(
                "Sample `x` and `t` as separate `(N,1)` tensors with `requires_grad_(True)`, "
                "concatenate them for the forward pass. You need them separate so you can "
                "differentiate with respect to each."
            ),
            "Hint 2 — the derivative call": mo.md(
                "`torch.autograd.grad(u.sum(), x, create_graph=True)[0]` gives `∂u/∂x` for the "
                "whole batch. The `.sum()` trick works because each output row depends only on "
                "its own input row. **`create_graph=True` is what keeps the result differentiable "
                "again** — forget it and `u_xx` is silently wrong."
            ),
            "Hint 3 — the second derivative": mo.md(
                "Apply the same call to `u_x` instead of `u`. That is all nesting means."
            ),
        }
    )
    return


@app.cell
def _(NU, n_points, net, torch):
    # --- SOLUTION, Task 1 (replace with your own attempt and re-run) -------------
    _N = int(n_points.value)
    x = (torch.rand(_N, 1) * 2 - 1).requires_grad_(True)     # x in [-1, 1]
    t = torch.rand(_N, 1).requires_grad_(True)               # t in [0, 1]

    u = net(torch.cat([x, t], dim=1))
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

    residual = u_t + u * u_x - NU * u_xx
    return residual, t, u, u_x, u_xx, x


@app.cell
def _(mo, residual, torch):
    _loss = (residual**2).mean()
    _loss.backward(retain_graph=True)
    mo.md(
        rf"""
    **Residual assembled.** mean |r| = `{residual.abs().mean().item():.4f}`,
    mean r² (the PINN loss) = `{(residual**2).mean().item():.4f}` — and after
    `.backward()`, every parameter has a finite gradient: this untrained network already
    *is* the core of a PINN. Training it to push r → 0 is Chapters 08–10.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 2 — prove the second derivative is right

    Autodiff is exact; your *code* might not be. The check that settles it: a central
    finite difference `(u(x+h) − 2u(x) + u(x−h)) / h²` should match `u_xx` to about six
    digits in float64 at `h = 1e-5`. Drag the `log10(h)` slider and watch the error: too
    large h → truncation error; too small h → float64 round-off. The sweet spot is the
    whole point of the exercise.
    """
    )
    return


@app.cell
def _(log_h, mo, net, t, torch, u_xx, x):
    _h = 10.0 ** float(log_h.value)
    with torch.no_grad():
        _up = net(torch.cat([x + _h, t], dim=1))
        _u0 = net(torch.cat([x, t], dim=1))
        _um = net(torch.cat([x - _h, t], dim=1))
        _fd = (_up - 2 * _u0 + _um) / (_h * _h)
    _rel = ((_fd - u_xx).abs() / (u_xx.abs() + 1e-12)).median().item()
    mo.md(
        rf"""
    With `h = {_h:.1e}`: **median relative disagreement = `{_rel:.2e}`**.

    At `h = 1e-5` you should see ~`1e-6`. If you see ~`1e-1` at every h, your graph is
    wrong — the classic cause is a missing `create_graph=True`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 3 — Idiom B: `torch.func` on one point, then `vmap`

    Write the physics for a *single* point — the way the PDE is written on paper — and
    vectorise mechanically. This idiom composes better and is the modern style
    (Chapter 03 §3.4). The check: A and B must agree to machine precision.
    """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Hint — the shape of it": mo.md(
                "`u_fn(params, z)` takes one point `z` of shape `(2,)` and returns a scalar. "
                "Then `jacrev(u_fn, argnums=1)` is its gradient, `jacfwd(jacrev(...))` its "
                "Hessian, and `vmap(..., in_dims=(None, 0))` maps it over the batch. "
                "Note **`in_dims`**, not JAX's `in_axes` — everyone hits this once."
            ),
        }
    )
    return


@app.cell
def _(net, t, torch, u_xx, x):
    # --- SOLUTION, Task 3 ---------------------------------------------------------
    from torch.func import functional_call, jacfwd, jacrev, vmap

    _params = dict(net.named_parameters())

    def _u_fn(params, z):                 # z: (2,) -> scalar. ONE point.
        return functional_call(net, params, (z.unsqueeze(0),)).squeeze()

    def _u_xx_one(params, z):             # d²u/dx² = H[0, 0]
        H = jacfwd(jacrev(_u_fn, argnums=1), argnums=1)(params, z)
        return H[0, 0]

    _pts = torch.cat([x, t], dim=1).detach()
    u_xx_B = vmap(_u_xx_one, in_dims=(None, 0))(_params, _pts)
    idiom_gap = (u_xx_B - u_xx.squeeze().detach()).abs().max().item()
    return idiom_gap, u_xx_B


@app.cell
def _(idiom_gap, mo):
    mo.md(
        rf"""
    **Idiom A vs Idiom B: max |difference| = `{idiom_gap:.2e}`** — the same exact
    derivative, computed two ways. (The starter kit measures 2.3e-16 for this comparison.)
    """
    )
    return


@app.cell
def _(make_net, mo, torch):
    # Task 4 — the rigorous stamp: gradgradcheck on a tiny net (needs float64)
    _tiny = make_net(width=8, seed=1)
    _z = torch.rand(3, 2, requires_grad=True)
    _ok = torch.autograd.gradgradcheck(
        lambda inp: _tiny(inp).sum(), (_z,), eps=1e-6, atol=1e-8, rtol=1e-6
    )
    mo.md(
        rf"""
    ## Task 4 — the rigorous stamp

    `torch.autograd.gradgradcheck` on a small network: **{_ok}**.

    This is the machine-checkable proof that second derivatives flow correctly through
    your network — and it *requires float64*, which is one of the two reasons this whole
    path runs in double precision (Chapter 04 §4.3).

    ---

    **Where next.** You have now done, interactively, what
    `starter/scripts/02_autodiff_playground.py` does as a script. Do the full §3.6
    exercise from a blank file, then move to Chapter 04 to set up the environment
    properly. Colab lane for this topic: the d2l.ai autograd section (Open-in-Colab
    button on the page) and the JAX Autodiff Cookbook, both linked from Chapter 03.
    """
    )
    return


if __name__ == "__main__":
    app.run()
