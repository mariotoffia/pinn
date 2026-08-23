"""Exact / trusted reference solutions.

A PINN with no reference solution is a plot, not a result. These are the ground truths the
scripts measure against.
"""

from __future__ import annotations

import numpy as np


def heat1d_exact(x: np.ndarray, t: np.ndarray, nu: float = 0.05, n_modes: int = 1) -> np.ndarray:
    """1D heat equation u_t = nu*u_xx on [0,1], u(0,x)=sin(pi x), u(t,0)=u(t,1)=0.

    Separation of variables gives u = exp(-nu*(k*pi)^2*t) * sin(k*pi*x) per mode.
    With n_modes=1 this is a single decaying sine - the friendliest possible PDE, and exactly
    what you want for your first honest error metric.

    Returns an array of shape (len(t), len(x)).
    """
    X, T = np.meshgrid(x, t)
    out = np.zeros_like(X)
    for k in range(1, n_modes + 1):
        amp = 1.0 if k == 1 else 1.0 / k
        out += amp * np.exp(-nu * (k * np.pi) ** 2 * T) * np.sin(k * np.pi * X)
    return out


def burgers_exact(
    x: np.ndarray, t: np.ndarray, nu: float = 0.01 / np.pi, n_quad: int = 300
) -> np.ndarray:
    """Viscous Burgers u_t + u u_x = nu u_xx on [-1,1], u(0,x) = -sin(pi x).

    The Cole-Hopf transform linearises Burgers into the heat equation, giving a closed-form
    solution as a ratio of two integrals:

        u(x,t) = - INT sin(pi*(x-eta)) f(x-eta) exp(-eta^2/(4 nu t)) d_eta
                 / INT              f(x-eta) exp(-eta^2/(4 nu t)) d_eta ,
        f(y)   = exp(-cos(pi y) / (2 pi nu)).

    Substituting eta = 2*sqrt(nu*t)*s turns both integrals into Gauss-Hermite quadrature against
    exp(-s^2), which is exact for smooth integrands at modest order. This is the reference the
    original PINN paper is measured against - and generating it yourself beats trusting a
    bundled .npz you cannot check.

    Returns an array of shape (len(t), len(x)).
    """
    s, w = np.polynomial.hermite.hermgauss(n_quad)          # nodes/weights for exp(-s^2)
    out = np.empty((len(t), len(x)))

    for i, ti in enumerate(t):
        if ti <= 0.0:
            out[i] = -np.sin(np.pi * x)
            continue
        c = 2.0 * np.sqrt(nu * ti)
        y = x[:, None] - c * s[None, :]                     # (nx, n_quad)
        # log f, stabilised: subtract the row max before exponentiating (log-sum-exp trick)
        logf = -np.cos(np.pi * y) / (2.0 * np.pi * nu)
        logf -= logf.max(axis=1, keepdims=True)
        f = np.exp(logf)
        num = (w[None, :] * np.sin(np.pi * y) * f).sum(axis=1)
        den = (w[None, :] * f).sum(axis=1)
        out[i] = -num / den
    return out


def burgers_reference_fd(
    nx: int = 2001, nt: int = 20001, nu: float = 0.01 / np.pi, t_end: float = 1.0
):
    """Independent finite-difference solve, used only to CHECK `burgers_exact`.

    Conservative flux form with central differences on a fine grid. Slow and not needed for the
    PINN scripts - it exists so you can verify one method against a completely different one,
    which is the habit chapter 10 step 10 is trying to build.
    """
    x = np.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]
    dt = t_end / (nt - 1)
    u = -np.sin(np.pi * x)
    snapshots = {0.0: u.copy()}
    for n in range(1, nt):
        flux = 0.5 * u**2
        du = np.zeros_like(u)
        du[1:-1] = -(flux[2:] - flux[:-2]) / (2 * dx) + nu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        u = u + dt * du
        u[0] = u[-1] = 0.0
        snapshots[n * dt] = u.copy()
    return x, snapshots
