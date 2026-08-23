"""04 - Your first PINN: the damped harmonic oscillator.

    m u'' + mu u' + k u = 0,   u(0) = 1,   u'(0) = 0

with an exact solution to measure against. This is the "hello world" of PINNs (Ben Moseley's
notebook uses the same problem) because it has every ingredient - a differential operator, initial
conditions, an exact answer - and none of the distractions.

The point of THIS script is the comparison in recipe step 4:

    SOFT initial conditions  -> u_theta trained with a penalty term, and lambda_ic decides whether
                                you get the answer or a plausible artifact
    HARD initial conditions  -> u_theta(t) = 1 + t^2 * N(t), so u(0)=1 and u'(0)=0 hold IDENTICALLY
                                and both IC loss terms disappear

Every constraint you hard-code is one loss term and one weight you no longer have to balance.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from pinnlab import PINN, d, set_seed, setup                # noqa: E402
from pinnlab.plotting import curves, lines                  # noqa: E402
from pinnlab.train import adam_then_lbfgs, relative_l2      # noqa: E402

D, W0 = 2.0, 20.0            # damping, natural frequency.  m=1, mu=2D, k=W0^2
MU, K = 2 * D, W0 ** 2
T_END = 1.0


def exact(t):
    """u = e^{-D t} (cos(w t) + (D/w) sin(w t)),  w = sqrt(W0^2 - D^2)."""
    w = np.sqrt(W0 ** 2 - D ** 2)
    return np.exp(-D * t) * (np.cos(w * t) + (D / w) * np.sin(w * t))


def run(hard_ic: bool, lam_ic: float = 1.0, steps: int = 4000, n_col: int = 512):
    set_seed(0)

    # Hard-constraint ansatz: 1 + t^2 * N(t) satisfies u(0)=1 and u'(0)=0 identically.
    hard = (lambda z, out: 1.0 + z ** 2 * out) if hard_ic else None
    net = PINN(in_dim=1, width=64, depth=3, fourier=True, fourier_features=32, sigma=1.0,
               hard=hard)

    t_ic = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    t_eval = torch.linspace(0, T_END, 1000, dtype=torch.float64).reshape(-1, 1)
    u_ref = torch.tensor(exact(t_eval.numpy().ravel()), dtype=torch.float64).reshape(-1, 1)

    state = {"pts": None, "frozen": False}

    def sample():
        if state["frozen"]:
            return state["pts"]
        return (torch.rand(n_col, 1, dtype=torch.float64) * T_END).requires_grad_(True)

    def loss_fn():
        t = sample()
        u = net(t)
        u_t = d(u, t)
        u_tt = d(u_t, t)
        res = (u_tt + MU * u_t + K * u)
        # Non-dimensionalise the residual by K so it is O(1) alongside the IC terms - recipe step 0.
        l_res = (res / K).pow(2).mean()
        if hard_ic:
            return l_res, [l_res]
        u0 = net(t_ic)
        u0_t = d(u0, t_ic)
        l_ic = (u0 - 1.0).pow(2).mean() + u0_t.pow(2).mean()
        return l_res + lam_ic * l_ic, [l_res, l_ic]

    def freeze():
        # Recipe step 8: freeze a LARGER set for L-BFGS. Resampling between quasi-Newton
        # steps poisons the curvature estimate.
        state["pts"] = (torch.rand(4 * n_col, 1, dtype=torch.float64) * T_END).requires_grad_(True)
        state["frozen"] = True

    def on_log(step, total, terms):
        with torch.no_grad():
            return {"relL2": relative_l2(net(t_eval), u_ref)}

    log = adam_then_lbfgs(
        net.parameters(), loss_fn,
        adam_steps=steps, lbfgs_steps=800, lr=1e-3,
        log_every=1000, on_log=on_log, freeze_for_lbfgs=freeze, verbose=True,
    )
    with torch.no_grad():
        pred = net(t_eval).numpy().ravel()
    return pred, relative_l2(net(t_eval).detach(), u_ref), log


def main():
    setup()
    t = np.linspace(0, T_END, 1000)
    ref = exact(t)

    print("=" * 74)
    print("SOFT initial conditions, lambda_ic = 1  (the naive default)")
    print("=" * 74)
    soft1, err_soft1, log_s1 = run(hard_ic=False, lam_ic=1.0)

    print("\n" + "=" * 74)
    print("SOFT initial conditions, lambda_ic = 100  (hand-tuned)")
    print("=" * 74)
    soft100, err_soft100, log_s100 = run(hard_ic=False, lam_ic=100.0)

    print("\n" + "=" * 74)
    print("HARD initial conditions: u(t) = 1 + t^2 * N(t)   (no IC loss term at all)")
    print("=" * 74)
    hard, err_hard, log_h = run(hard_ic=True)

    lines(t, {"exact": ref, "soft IC (lam=1)": soft1, "soft IC (lam=100)": soft100,
              "hard IC": hard},
          "04_oscillator.png", xlabel="t", ylabel="u(t)",
          title="Damped harmonic oscillator PINN")
    curves({"soft IC (lam=1)": (log_s1.step, [e["relL2"] for e in log_s1.extra]),
            "soft IC (lam=100)": (log_s100.step, [e["relL2"] for e in log_s100.extra]),
            "hard IC": (log_h.step, [e["relL2"] for e in log_h.extra])},
           "04_oscillator_error.png", ylabel="relative L2 error",
           title="Hard constraints remove a weight you would otherwise have to tune")

    print("\n" + "=" * 74)
    print("RESULT  (relative L2 error against the exact solution)")
    print(f"  soft IC, lambda_ic = 1   : {err_soft1:.3e}")
    print(f"  soft IC, lambda_ic = 100 : {err_soft100:.3e}")
    print(f"  hard IC (no IC loss)     : {err_hard:.3e}")
    print("=" * 74)
    print("The lambda you did not have to choose is the one that cannot be wrong.")
    print("Next: scripts/05_pinn_heat1d.py - your first PDE.")


if __name__ == "__main__":
    main()
