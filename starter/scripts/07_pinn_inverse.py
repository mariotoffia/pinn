"""07 - The inverse problem: recover a PDE coefficient from sparse, noisy data.

This is what PINNs are good at.

    u_t = nu u_xx  on [0,1]x[0,1],   nu unknown (truth: 0.05)
    given: 60 scattered, noisy measurements of u

Promote nu to a trainable parameter and minimise over (theta, nu) jointly. No adjoint solver, no
second optimisation loop, no extra machinery - you add one `nn.Parameter` and one data term.

Doing this with FEM means deriving and implementing an adjoint. That gap is the strongest
genuine argument for PINNs, and it is the reason chapter 14 says "learn PINNs for THIS."

The script sweeps the noise level so you can see where it breaks, which matters more than the
headline number at 1% noise.
"""

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from pinnlab import PINN, d, set_seed, setup                # noqa: E402
from pinnlab.plotting import curves, three_panel            # noqa: E402
from pinnlab.reference import heat1d_exact                  # noqa: E402
from pinnlab.sampling import latin_hypercube                # noqa: E402
from pinnlab.train import adam_then_lbfgs, relative_l2      # noqa: E402

NU_TRUE = 0.05
BOUNDS = [(0.0, 1.0), (0.0, 1.0)]        # (t, x)


def make_data(n_data: int, noise: float, rng):
    """Sparse scattered measurements of the exact solution, with multiplicative-scale noise."""
    t = rng.uniform(0, 1, n_data)
    x = rng.uniform(0, 1, n_data)
    u = np.exp(-NU_TRUE * np.pi**2 * t) * np.sin(np.pi * x)
    u_noisy = u + noise * np.abs(u).std() * rng.standard_normal(n_data)
    z = torch.tensor(np.stack([t, x], axis=1), dtype=torch.float64)
    return z, torch.tensor(u_noisy, dtype=torch.float64).reshape(-1, 1)


def run(noise: float, n_data: int = 60, steps: int = 3000, seed: int = 0, verbose: bool = True):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    z_data, u_data = make_data(n_data, noise, rng)

    net = PINN(in_dim=2, width=64, depth=4)

    # The unknown. Parameterised as log(nu) so it stays positive and the optimiser sees a
    # well-scaled variable - a small thing that prevents a lot of failed runs.
    log_nu = torch.nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float64))

    state = {"pts": None, "frozen": False}

    def sample():
        return state["pts"] if state["frozen"] else latin_hypercube(1024, BOUNDS)

    def loss_fn():
        z = sample()
        u = net(z)
        g = d(u, z)
        u_xx = d(g[:, 1:2], z)[:, 1:2]
        l_res = (g[:, 0:1] - torch.exp(log_nu) * u_xx).pow(2).mean()
        l_data = (net(z_data) - u_data).pow(2).mean()
        # The data term must dominate early, or nu drifts to whatever makes the residual easy
        # (nu -> 0 turns the PDE into u_t = 0, which is trivially satisfiable).
        return l_res + 100.0 * l_data, [l_res, l_data]

    def freeze():
        state["pts"] = latin_hypercube(4096, BOUNDS)
        state["frozen"] = True

    def on_log(step, total, terms):
        return {"nu": float(torch.exp(log_nu).detach())}

    params = list(net.parameters()) + [log_nu]
    log = adam_then_lbfgs(params, loss_fn, adam_steps=steps, lbfgs_steps=800, lr=2e-3,
                          log_every=1000, on_log=on_log, freeze_for_lbfgs=freeze, verbose=verbose)

    nu = float(torch.exp(log_nu).detach())

    xs = np.linspace(0, 1, 201)
    ts = np.linspace(0, 1, 101)
    ref = heat1d_exact(xs, ts, nu=NU_TRUE)
    XX, TT = np.meshgrid(xs, ts)
    z_eval = torch.tensor(np.stack([TT.ravel(), XX.ravel()], axis=1), dtype=torch.float64)
    with torch.no_grad():
        pred = net(z_eval).numpy().reshape(ref.shape)
    field_err = relative_l2(torch.tensor(pred), torch.tensor(ref))
    return nu, field_err, pred, ref, log, (z_data, u_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.01, help="noise level for the main run")
    ap.add_argument("--sweep", action="store_true", help="also sweep noise 0, 1%%, 5%%, 20%%")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup()

    print("=" * 74)
    print(f"MAIN RUN: 60 scattered measurements, {args.noise * 100:.0f}% noise, "
          f"true nu = {NU_TRUE}")
    print("=" * 74)
    nu, field_err, pred, ref, log, (zd, ud) = run(args.noise, steps=args.steps, seed=args.seed)

    three_panel(pred, ref, (0, 1, 0, 1), "07_inverse_field.png",
                titles=("PINN (nu discovered)", "exact (nu known)", "error"))
    curves({"discovered nu": (log.step, [e["nu"] for e in log.extra])},
           "07_inverse_nu.png", ylabel="nu", logy=False,
           title=f"Coefficient discovery (truth {NU_TRUE})")

    print("\n" + "=" * 74)
    print(f"discovered nu      : {nu:.6f}   (truth {NU_TRUE})")
    print(f"relative error     : {abs(nu - NU_TRUE) / NU_TRUE * 100:.3f}%")
    print(f"field relative L2  : {field_err:.3e}")
    print(f"wall clock         : {log.seconds:.1f} s")
    print("=" * 74)
    print("Sixty noisy points and one equation, and you recovered a physical constant AND the")
    print("field everywhere. No adjoint solver was written. That is the argument for PINNs.")

    if args.sweep:
        print("\n" + "=" * 74)
        print("NOISE SWEEP - where does it break?")
        print("=" * 74)
        rows = []
        for noise in (0.0, 0.01, 0.05, 0.20):
            nu_i, err_i, *_ = run(noise, steps=args.steps, seed=args.seed, verbose=False)
            rows.append((noise, nu_i, abs(nu_i - NU_TRUE) / NU_TRUE * 100, err_i))
            print(f"  noise {noise * 100:5.1f}%   nu = {nu_i:.6f}   "
                  f"error {rows[-1][2]:7.3f}%   field relL2 {err_i:.3e}", flush=True)
        print("\nReport this table, not just the clean number. An inverse method that only works")
        print("on noise-free data has not been tested on the problem it exists to solve.")

    print("\nThat is the starter kit. Next: chapter 12 for the DeepXDE gallery, and chapter 13")
    print("for operator learning - `pip install neuraloperator` and train at 64x64, evaluate at 256x256.")


if __name__ == "__main__":
    main()
