"""03 - Spectral bias, made visible.

Networks fit LOW frequencies first. That is the F-principle, and it is the exact opposite of
classical iterative solvers (Jacobi and Gauss-Seidel damp HIGH frequencies fastest, which is what
makes multigrid work). It is also the single most-cited reason PINNs fail on multiscale or
high-frequency solutions.

Target: f(x) = sin(2*pi*x) + 0.3*sin(2*pi*15*x)   -- one easy mode, one hard mode.

Watch the high-frequency component arrive last, or never - and then watch a random Fourier
feature embedding fix it (recipe step 2).
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

from pinnlab import PINN, set_seed, setup           # noqa: E402
from pinnlab.plotting import curves, lines          # noqa: E402

LOW, HIGH, AMP = 1.0, 15.0, 0.3


def target(x):
    return torch.sin(2 * torch.pi * LOW * x) + AMP * torch.sin(2 * torch.pi * HIGH * x)


def band_energy(pred: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Project the prediction onto the two target modes. This is the number that tells the story."""
    lo = 2 * np.trapezoid(pred * np.sin(2 * np.pi * LOW * x), x)
    hi = 2 * np.trapezoid(pred * np.sin(2 * np.pi * HIGH * x), x)
    return float(lo), float(hi)


def train(fourier: bool, steps=8000, sigma=3.0):
    set_seed(0)
    net = PINN(in_dim=1, width=128, depth=3, fourier=fourier, fourier_features=64, sigma=sigma)
    x = torch.linspace(0, 1, 512, dtype=torch.float64).reshape(-1, 1)
    y = target(x)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 2000, 0.9)

    xs_np = x.numpy().ravel()
    hist_step, hist_loss, hist_lo, hist_hi = [], [], [], []
    for s in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if s % 250 == 0:
            with torch.no_grad():
                lo, hi = band_energy(net(x).numpy().ravel(), xs_np)
            hist_step.append(s)
            hist_loss.append(float(loss.detach()))
            hist_lo.append(abs(lo))
            hist_hi.append(abs(hi))
    with torch.no_grad():
        pred = net(x).numpy().ravel()
    return xs_np, pred, hist_step, hist_loss, hist_lo, hist_hi


def main():
    setup()
    print("Fitting sin(2*pi*x) + 0.3*sin(2*pi*15*x) with a plain MLP and a Fourier-feature MLP.")
    print("True mode amplitudes: low = 1.000, high = 0.300\n")

    results = {}
    for label, use_f in (("plain MLP", False), ("Fourier features (sigma=3)", True)):
        x, pred, st, lo_hist, lo, hi = train(use_f)
        results[label] = (x, pred, st, lo_hist, lo, hi)
        print(f"{label}")
        print(f"  final mse                : {lo_hist[-1]:.3e}")
        print(f"  recovered low  amplitude : {lo[-1]:.4f}  (true 1.000)")
        print(f"  recovered high amplitude : {hi[-1]:.4f}  (true 0.300)")
        first = next((s for s, h in zip(st, hi) if h > 0.15), None)
        print(f"  step at which the high mode reached half amplitude: "
              f"{first if first is not None else 'NEVER'}\n")

    x = results["plain MLP"][0]
    lines(
        x,
        {
            "target": target(torch.tensor(x).reshape(-1, 1)).numpy().ravel(),
            "plain MLP": results["plain MLP"][1],
            "Fourier features": results["Fourier features (sigma=3)"][1],
        },
        "03_spectral_bias_fit.png",
        ylabel="f(x)",
        title="A plain MLP smooths away the high-frequency mode",
    )
    curves(
        {
            f"{k} - low mode": (v[2], v[4]) for k, v in results.items()
        } | {
            f"{k} - HIGH mode": (v[2], v[5]) for k, v in results.items()
        },
        "03_spectral_bias_modes.png",
        ylabel="recovered amplitude",
        logy=False,
        title="Low frequencies are learned first (the F-principle)",
    )
    curves({k: (v[2], v[3]) for k, v in results.items()}, "03_spectral_bias_loss.png",
           title="Training loss")

    print("Keep 03_spectral_bias_modes.png. It explains half of chapter 09.")
    print("Next: scripts/04_pinn_oscillator.py - your first PINN.")


if __name__ == "__main__":
    main()
