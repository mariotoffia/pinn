# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib>=3.8",
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
    # Lab 04 — Your first PINN: the damped oscillator, soft vs hard

    **What this lab does.** Trains a physics-informed network for the damped harmonic
    oscillator, written in its non-dimensional form (Chapter 10, Step 0):

    `ü + 2ζ·u̇ + u = 0,  u(0) = 1,  u̇(0) = 0,  ζ = 0.1,  τ ∈ [0, 4π]`  (two periods)

    You get three of the recipe's steps in one small experiment:

    - **Step 0** — the time axis is rescaled to network input `s ∈ [−1, 1]`, and the
      derivative factors are chain-ruled in. Feed a network `τ ∈ [0, 12.6]` raw and
      watch it struggle; this lab does it right.
    - **Step 4** — a switch between **soft** initial conditions (λ-weighted penalties)
      and **hard** ones, built in as `u = 1 + (s+1)²·N(s)`.
    - **Steps 7–8** — the **Adam → L-BFGS** sequence, with the collocation set frozen
      for the L-BFGS phase. Set the L-BFGS iterations to 0 and watch the error jump
      by two orders of magnitude: that is what "L-BFGS is the PINN endgame" means.

    **How this lab works.** Every cell already runs a working solution — you cannot get
    stuck. Try writing the marked pieces yourself first (hints below), then compare.

    **Foundations needed:** the Chapter 08 guided tour, §8.5–8.7, and Lab 02 for the
    double `autograd.grad`.

    **Success looks like** (defaults, seed 0, measured while writing this lab):
    soft λ=1 → **2.9e-3** · soft λ=100 → **3.0e-1** (the hand-tuned big weight is ~100×
    WORSE) · hard → **2.5e-3**. The ordering — hard ≤ λ=1 ≪ λ=100 — is the lesson. The
    starter kit's longer runs sharpen the same ordering to 1.6e-4 / 4.8e-4 / 1.1e-2.
    """
    )
    return


@app.cell
def _(mo):
    mode = mo.ui.radio(options=["soft", "hard"], value="soft", label="initial-condition style")
    log_lam = mo.ui.slider(-1.0, 2.0, step=0.25, value=0.0, label="log10 λ (soft mode)")
    adam_steps = mo.ui.slider(300, 3000, step=100, value=1200, label="Adam steps")
    lbfgs_iters = mo.ui.slider(0, 400, step=50, value=200, label="L-BFGS iterations (0 = off)")
    mo.vstack([mo.hstack([mode, log_lam], justify="start", gap=2),
               mo.hstack([adam_steps, lbfgs_iters], justify="start", gap=2)])
    return adam_steps, lbfgs_iters, log_lam, mode


@app.cell
def _():
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    torch.set_default_dtype(torch.float64)
    ZETA = 0.1
    T_END = 4 * math.pi                    # two periods
    A = 2.0 / T_END                        # ds/dtau for s = 2*tau/T - 1  (Step 0!)
    return A, T_END, ZETA, math, np, plt, torch


@app.cell
def _(T_END, ZETA, math, torch):
    _wd = math.sqrt(1 - ZETA * ZETA)

    def exact_of_s(s):
        tau = (s + 1.0) * T_END / 2.0
        return torch.exp(-ZETA * tau) * (torch.cos(_wd * tau) + (ZETA / _wd) * torch.sin(_wd * tau))
    return (exact_of_s,)


@app.cell
def _(mo):
    mo.accordion(
        {
            "Task A — the residual, with the Step-0 factors": mo.md(
                "The network sees `s`, but the ODE lives in `τ`. With `a = ds/dτ`: "
                "`du/dτ = a·u_s` and `d²u/dτ² = a²·u_ss`. So the residual is "
                "`r = a²·u_ss + 2ζ·a·u_s + u`. Two `autograd.grad` calls with "
                "`create_graph=True`, then this line."
            ),
            "Task B — the hard-constraint wrapper": mo.md(
                "Find a form with `u = 1` and `du/dτ = 0` at `s = −1`, for any network N. "
                "Answer: `u = 1 + (s+1)²·N(s)` — value 1 and zero slope at s = −1, by "
                "construction. Two loss terms and their λ's disappear."
            ),
            "Why the L-BFGS phase uses a FROZEN point set": mo.md(
                "L-BFGS builds curvature estimates from successive gradients. Resample the "
                "points between its steps and those gradients describe different functions — "
                "the estimate is poisoned. Freeze during L-BFGS; resample during Adam. "
                "(Chapter 05 §5.2, pitfall 3; Chapter 10 Step 8.)"
            ),
        }
    )
    return


@app.cell
def _(A, ZETA, adam_steps, lbfgs_iters, log_lam, mode, torch):
    # --- SOLUTION ----------------------------------------------------------------
    torch.manual_seed(0)
    _layers = []
    _dims = [1, 64, 64, 64, 1]
    for _i in range(len(_dims) - 1):
        _lin = torch.nn.Linear(_dims[_i], _dims[_i + 1])
        torch.nn.init.xavier_normal_(_lin.weight, gain=5.0 / 3.0)
        torch.nn.init.zeros_(_lin.bias)
        _layers.append(_lin)
        if _i < len(_dims) - 2:
            _layers.append(torch.nn.Tanh())
    _net = torch.nn.Sequential(*_layers)

    hard = mode.value == "hard"
    lam = 10.0 ** float(log_lam.value)

    def u_of(s):
        if hard:
            return 1.0 + (s + 1.0) ** 2 * _net(s)     # u(0)=1, u'(0)=0 by construction
        return _net(s)

    def full_loss(s_batch):
        _s = s_batch.clone().requires_grad_(True)
        _u = u_of(_s)
        _us = torch.autograd.grad(_u.sum(), _s, create_graph=True)[0]
        _uss = torch.autograd.grad(_us.sum(), _s, create_graph=True)[0]
        _r = A * A * _uss + 2 * ZETA * A * _us + _u   # Task A
        _loss = (_r**2).mean()
        if not hard:
            _s0 = torch.full((1, 1), -1.0, requires_grad=True)
            _u0 = u_of(_s0)
            _u0s = torch.autograd.grad(_u0.sum(), _s0, create_graph=True)[0]
            _loss = _loss + lam * ((_u0 - 1.0) ** 2).mean() + lam * ((A * _u0s) ** 2).mean()
        return _loss

    # Phase 1: Adam, resampling every step (Step 8)
    _opt = torch.optim.Adam(_net.parameters(), lr=2e-3)
    for _step in range(int(adam_steps.value)):
        _l = full_loss(torch.rand(200, 1) * 2 - 1)
        _opt.zero_grad()
        _l.backward()
        _opt.step()
    adam_final = _l.item()

    # Phase 2: L-BFGS on a FROZEN set (Steps 7-8)
    if int(lbfgs_iters.value) > 0:
        _sfix = torch.linspace(-1, 1, 256).unsqueeze(1)
        _lb = torch.optim.LBFGS(_net.parameters(), max_iter=int(lbfgs_iters.value),
                                line_search_fn="strong_wolfe")

        def _closure():
            _lb.zero_grad()
            _c = full_loss(_sfix)
            _c.backward()
            return _c

        _lb.step(_closure)
    return adam_final, full_loss, hard, lam, u_of


@app.cell
def _(exact_of_s, torch, u_of):
    _sg = torch.linspace(-1, 1, 400).unsqueeze(1)
    with torch.no_grad():
        u_pred = u_of(_sg)
    u_ex = exact_of_s(_sg)
    rel_l2 = (torch.linalg.norm(u_pred - u_ex) / torch.linalg.norm(u_ex)).item()
    s_grid = _sg
    return rel_l2, s_grid, u_ex, u_pred


@app.cell
def _(T_END, hard, lam, lbfgs_iters, plt, rel_l2, s_grid, u_ex, u_pred):
    _tau = (s_grid.squeeze() + 1) * T_END / 2
    _fig, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.plot(_tau, u_ex.squeeze(), "--", lw=1, color="gray", label="exact")
    _ax.plot(_tau, u_pred.squeeze(), lw=1.8, color="#b4541e", label="PINN")
    _title = "HARD IC (u = 1 + (s+1)²·N)" if hard else f"SOFT IC, λ = {lam:g}"
    _lb = f"L-BFGS ×{int(lbfgs_iters.value)}" if int(lbfgs_iters.value) > 0 else "no L-BFGS"
    _ax.set_title(f"{_title} · {_lb} · relative L2 error = {rel_l2:.2e}")
    _ax.set_xlabel("τ")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(hard, lam, lbfgs_iters, mo, rel_l2):
    if int(lbfgs_iters.value) == 0:
        _note = ("**L-BFGS is off** — compare with the default: Adam alone typically lands around "
                 "1e-1, and the L-BFGS polish buys ~two orders of magnitude on this smooth, "
                 "full-batch problem. That is Step 7 of the recipe, felt.")
    elif hard:
        _note = ("**Hard constraints:** the λ knob is gone and the error should match or beat the "
                 "best soft run. Lagaris 1997 — Chapter 10, Step 4.")
    elif lam >= 50:
        _note = ("**λ is huge:** the optimiser babysits two endpoint penalties while the physics "
                 "starves — expect ~100× worse than λ=1. Hand-tuning loss weights by instinct "
                 "does not work; that is why Chapter 10 Step 6 balances them from gradient "
                 "statistics instead.")
    elif lam <= 0.2:
        _note = ("**λ is tiny:** the equation is satisfied but the start point drifts — a valid "
                 "solution of the ODE, just not of *your* problem. The uniqueness lesson from the "
                 "Chapter 08 tour, reproduced.")
    else:
        _note = "**Balanced λ:** good — now flip to hard mode and delete the knob entirely."
    mo.md(
        rf"""
    Current run: **rel-L2 = {rel_l2:.2e}**. {_note}

    ---

    **Where next.** The scripted version with the measured reference numbers is
    `starter/scripts/04_pinn_oscillator.py`. The Colab lane is Ben Moseley's
    harmonic-oscillator PINN workshop (Open-in-Colab badge in its repo) — same physics
    plus the inverse problem. Then Chapter 08 §8.9: Burgers, written out in full.
    """
    )
    return


if __name__ == "__main__":
    app.run()
