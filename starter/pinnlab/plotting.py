"""Plotting helpers.

The one rule: ALWAYS plot prediction, reference and ERROR side by side. A PINN that looks right
and is 5% wrong is the standard failure mode, and only the error panel shows it. Use a diverging
colormap centred at zero for the error.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")                      # headless: scripts write files, they do not pop windows
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)


def save(fig, name: str) -> pathlib.Path:
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def three_panel(pred, ref, extent, name, titles=("PINN", "reference", "error"), cmap="viridis"):
    """pred / ref are 2D arrays (nt, nx); extent is (x0, x1, t0, t1)."""
    pred, ref = np.asarray(pred), np.asarray(ref)
    err = pred - ref
    vmin, vmax = float(min(pred.min(), ref.min())), float(max(pred.max(), ref.max()))
    lim = float(np.abs(err).max()) or 1e-16

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, data, title, kw in zip(
        axes,
        [pred, ref, err],
        titles,
        [dict(vmin=vmin, vmax=vmax, cmap=cmap),
         dict(vmin=vmin, vmax=vmax, cmap=cmap),
         dict(vmin=-lim, vmax=lim, cmap="RdBu_r")],   # diverging, centred at zero
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent, **kw)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
    return save(fig, name)


def curves(series: dict, name: str, xlabel="step", ylabel="loss", logy=True, title=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, label=label, lw=1.6)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, name)


def lines(x, series: dict, name: str, xlabel="x", ylabel="u", title=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, y in series.items():
        style = "--" if "ref" in label or "exact" in label else "-"
        ax.plot(x, y, style, label=label, lw=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return save(fig, name)
