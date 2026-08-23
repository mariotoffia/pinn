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
    # Lab 03 — Spectral bias, and the Fourier-feature fix

    **What this lab does.** A real PyTorch MLP fits `sin(x) + a·sin(kx)` — one slow wave
    carrying one fast ripple. You watch which frequency the network learns first, then
    switch on a **random Fourier feature mapping** and watch the fast mode arrive. Sliders
    control the fast frequency `k`, its amplitude `a`, the feature scale **σ**, and the
    training budget — so you can find the too-small-σ and too-large-σ failure modes
    yourself (the reason Chapter 10 recommends σ ∈ [1, 10]).

    **How this lab works.** Everything runs as-is — you cannot get stuck. The one task
    asks you to write the Fourier mapping yourself; the working solution is already in
    the cell, with hints in the accordion.

    **Foundations needed:** Chapter 05 §5.6 (spectral bias), the Chapter 05 guided tour.

    **Success looks like:** without features, the recovered fast-mode amplitude stalls far
    below its true value; with features (σ ≈ 3–8), it converges to the target. The starter
    kit's measured version of the same fact: 0.05 recovered out of 0.30 without features —
    0.3000 with them.
    """
    )
    return


@app.cell
def _(mo):
    k_fast = mo.ui.slider(5, 30, step=1, value=15, label="fast frequency k")
    amp = mo.ui.slider(0.1, 0.5, step=0.05, value=0.3, label="fast amplitude a")
    use_ff = mo.ui.switch(value=False, label="Fourier features")
    sigma = mo.ui.slider(0.5, 20, step=0.5, value=6.0, label="feature scale σ")
    steps = mo.ui.slider(200, 3000, step=200, value=1000, label="Adam steps")
    mo.vstack([mo.hstack([k_fast, amp], justify="start", gap=2),
               mo.hstack([use_ff, sigma, steps], justify="start", gap=2)])
    return amp, k_fast, sigma, steps, use_ff


@app.cell
def _():
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    torch.set_default_dtype(torch.float64)
    return math, np, plt, torch


@app.cell
def _(amp, k_fast, math, torch):
    N = 256
    xs = torch.linspace(-math.pi, math.pi, N).unsqueeze(1)
    target = torch.sin(xs) + float(amp.value) * torch.sin(int(k_fast.value) * xs)
    return N, target, xs


@app.cell
def _(mo):
    mo.accordion(
        {
            "Task — write the Fourier mapping yourself": mo.md(
                "The mapping is `γ(x) = [cos(Bx), sin(Bx)]` with `B ~ N(0, σ²)` fixed (not "
                "trained). Input dim 1 → feature dim 2·n_features. Two lines of PyTorch."
            ),
            "Hint": mo.md(
                "`B = torch.randn(1, n_features) * sigma` once, then "
                "`torch.cat([torch.cos(x @ B), torch.sin(x @ B)], dim=1)` in the forward pass. "
                "B is a buffer, not a parameter — the whole point is that it is random and fixed."
            ),
        }
    )
    return


@app.cell
def _(sigma, torch, use_ff, xs):
    # --- SOLUTION: the feature mapping ------------------------------------------
    torch.manual_seed(3)
    NFEAT = 32
    B = torch.randn(1, NFEAT) * float(sigma.value)

    def featurize(x):
        if not use_ff.value:
            return x / 3.15                       # plain input, roughly [-1, 1]
        p = x @ B
        return torch.cat([torch.cos(p), torch.sin(p)], dim=1)

    in_dim = featurize(xs[:2]).shape[1]
    return featurize, in_dim


@app.cell
def _(featurize, in_dim, steps, target, torch, xs):
    torch.manual_seed(0)
    _layers = []
    _dims = [in_dim, 64, 64, 1]
    for _i in range(len(_dims) - 1):
        _lin = torch.nn.Linear(_dims[_i], _dims[_i + 1])
        torch.nn.init.xavier_normal_(_lin.weight, gain=5.0 / 3.0)
        torch.nn.init.zeros_(_lin.bias)
        _layers.append(_lin)
        if _i < len(_dims) - 2:
            _layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*_layers)

    _opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    _feat = featurize(xs)
    for _s in range(int(steps.value)):
        _opt.zero_grad()
        _loss = ((model(_feat) - target) ** 2).mean()
        _loss.backward()
        _opt.step()
    final_loss = _loss.item()
    return final_loss, model


@app.cell
def _(N, amp, featurize, k_fast, model, target, torch, xs):
    with torch.no_grad():
        pred = model(featurize(xs))
        a_slow = (2 / N) * (pred.squeeze() * torch.sin(xs.squeeze())).sum().item()
        a_fast = (2 / N) * (pred.squeeze() * torch.sin(int(k_fast.value) * xs.squeeze())).sum().item()
    a_true = float(amp.value)
    return a_fast, a_slow, a_true, pred


@app.cell
def _(a_fast, a_slow, a_true, final_loss, plt, pred, target, xs):
    _fig, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.plot(xs.squeeze(), target.squeeze(), "--", lw=1, color="gray", label="target")
    _ax.plot(xs.squeeze(), pred.squeeze().detach(), lw=1.8, color="#b4541e", label="network")
    _ax.set_title(
        f"recovered amplitudes:  k=1 → {a_slow:.3f} (target 1.0)   ·   "
        f"fast mode → {a_fast:.3f} (target {a_true})   ·   MSE {final_loss:.1e}"
    )
    _ax.legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(a_fast, a_true, mo, use_ff):
    _frac = a_fast / a_true if a_true else 0.0
    if use_ff.value and _frac > 0.85:
        _verdict = "**The fast mode arrived.** This is Step 2 of the Chapter 10 recipe doing its job."
    elif use_ff.value:
        _verdict = ("**Features are on but the fast mode is still short.** Try σ closer to the "
                    "fast frequency, or more steps — and note how σ too large makes the fit noisy "
                    "instead: both failure modes are real, which is why σ ∈ [1, 10] is only a heuristic.")
    else:
        _verdict = ("**The fast mode is missing** while the slow one is already almost perfect — "
                    "spectral bias, live. Now switch Fourier features on and compare.")
    mo.md(
        _verdict
        + "\n\n---\n\n**Where next.** `starter/scripts/03_spectral_bias.py` is the scripted "
        "version with the reference figure in `starter/examples/`. The Colab lane is the "
        "official Fourier-features demo notebook (Tancik et al.) linked from Chapter 02 §2.5 "
        "and the Chapter 05 tour. Theory: §5.6, then the Tancik paper."
    )
    return


if __name__ == "__main__":
    app.run()
