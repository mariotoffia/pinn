"""05 - Your first PDE: the 1D heat equation, measured against an exact solution.

    u_t = nu u_xx,   x in [0,1],  t in [0,1]
    u(0,x) = sin(pi x),   u(t,0) = u(t,1) = 0
    exact:  u(t,x) = exp(-nu pi^2 t) sin(pi x)

This is the friendliest possible PDE - parabolic, smooth, one Fourier mode - which makes it the
right place to build the habit that matters: measure your PINN against something you trust, on a
FRESH dense grid, and always plot the error panel.

Run with --hard to impose the IC and both BCs exactly:

    u_theta(t,x) = sin(pi x) * [1 + t * N(t,x)]

which satisfies u(0,x)=sin(pi x) and u(t,0)=u(t,1)=0 identically, leaving a single loss term.
"""

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from pinnlab import GradNormBalancer, PINN, d, set_seed, setup      # noqa: E402
from pinnlab.plotting import curves, three_panel                    # noqa: E402
from pinnlab.reference import heat1d_exact                          # noqa: E402
from pinnlab.sampling import latin_hypercube                        # noqa: E402
from pinnlab.train import adam_then_lbfgs, relative_l2              # noqa: E402

NU = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", action="store_true", help="impose IC and BCs exactly")
    ap.add_argument("--fourier", action="store_true",
                    help="add Fourier features (this solution is a single LOW mode, so they "
                         "are not needed here - try it and see)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-col", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup()
    set_seed(args.seed)

    hard = (lambda z, out: torch.sin(torch.pi * z[:, 1:2]) * (1.0 + z[:, 0:1] * out)) if args.hard else None
    net = PINN(in_dim=2, width=128, depth=4, fourier=args.fourier, fourier_features=48,
               sigma=1.5, hard=hard)
    print(f"network: {net.n_params()} parameters, hard constraints: {args.hard}")

    bounds = [(0.0, 1.0), (0.0, 1.0)]        # (t, x)
    state = {"pts": None, "ic": None, "bc": None, "frozen": False}

    def sample():
        """Recipe step 8: resample every iteration during Adam, FREEZE during L-BFGS.

        Everything stochastic must be frozen, not just the interior points - a quasi-Newton
        method optimises a fixed function, and a resampled IC/BC term is enough to make its
        curvature estimate meaningless.
        """
        if state["frozen"]:
            return state["pts"], state["ic"], state["bc"]
        x0 = torch.rand(256, 1, dtype=torch.float64)
        tb = torch.rand(256, 1, dtype=torch.float64)
        return latin_hypercube(args.n_col, bounds), x0, tb

    def residual(z):
        u = net(z)
        g = d(u, z)
        u_t, u_x = g[:, 0:1], g[:, 1:2]
        u_xx = d(u_x, z)[:, 1:2]
        return u_t - NU * u_xx

    balancer = None if args.hard else GradNormBalancer(3, net.parameters(), every=500)

    def loss_fn():
        z, x0, tb = sample()
        l_res = residual(z).pow(2).mean()
        if args.hard:
            return l_res, [l_res]

        # initial condition:  u(0,x) = sin(pi x)
        z0 = torch.cat([torch.zeros_like(x0), x0], dim=1)
        l_ic = (net(z0) - torch.sin(torch.pi * x0)).pow(2).mean()

        # boundary conditions: u(t,0) = u(t,1) = 0
        zb = torch.cat([torch.cat([tb, tb]), torch.cat([torch.zeros_like(tb),
                                                        torch.ones_like(tb)])], dim=1)
        l_bc = net(zb).pow(2).mean()

        terms = [l_res, l_ic, l_bc]
        if not state["frozen"]:
            balancer.maybe_update(terms)     # weights are also frozen for L-BFGS
        return balancer.combine(terms), terms

    def freeze():
        state["pts"] = latin_hypercube(3 * args.n_col, bounds)
        state["ic"] = torch.rand(512, 1, dtype=torch.float64)
        state["bc"] = torch.rand(512, 1, dtype=torch.float64)
        state["frozen"] = True

    # A fresh, dense evaluation grid - NEVER the training points (recipe step 10a).
    xs = np.linspace(0, 1, 201)
    ts = np.linspace(0, 1, 101)
    ref = heat1d_exact(xs, ts, nu=NU)
    XX, TT = np.meshgrid(xs, ts)
    z_eval = torch.tensor(np.stack([TT.ravel(), XX.ravel()], axis=1), dtype=torch.float64)
    ref_t = torch.tensor(ref.ravel(), dtype=torch.float64).reshape(-1, 1)

    def on_log(step, total, terms):
        with torch.no_grad():
            return {"relL2": relative_l2(net(z_eval), ref_t)}

    log = adam_then_lbfgs(net.parameters(), loss_fn, adam_steps=args.steps, lbfgs_steps=600,
                          lr=1e-3, log_every=1000, on_log=on_log, freeze_for_lbfgs=freeze)

    with torch.no_grad():
        pred = net(z_eval).numpy().reshape(ref.shape)
    err = relative_l2(torch.tensor(pred), torch.tensor(ref))

    tag = "hard" if args.hard else "soft"
    three_panel(pred, ref, (0, 1, 0, 1), f"05_heat1d_{tag}.png",
                titles=("PINN", "exact", "error (diverging, centred at 0)"))
    curves({"relative L2": (log.step, [e["relL2"] for e in log.extra])},
           f"05_heat1d_{tag}_error.png", ylabel="relative L2 error")

    print("\n" + "=" * 74)
    print(f"relative L2 error on a fresh 101x201 grid : {err:.3e}")
    print(f"max pointwise error                       : {np.abs(pred - ref).max():.3e}")
    print(f"IC error at t=0                           : "
          f"{np.abs(pred[0] - np.sin(np.pi * xs)).max():.3e}")
    print(f"BC error at x=0 and x=1                   : "
          f"{max(np.abs(pred[:, 0]).max(), np.abs(pred[:, -1]).max()):.3e}")
    print(f"wall clock                                : {log.seconds:.1f} s")
    print("=" * 74)
    print("Check the IC and BC lines separately, not just the total. A PINN can drive the")
    print("residual to zero while quietly satisfying the WRONG initial condition.")
    print("\nNext: scripts/06_pinn_burgers.py - the canonical benchmark, with an ablation.")


if __name__ == "__main__":
    main()
