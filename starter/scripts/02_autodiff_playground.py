"""02 - The hinge exercise: build a PDE residual with autodiff.

By the end of this script you will have written the Burgers' equation residual and its parameter
gradient, before reading a single PINN paper. Everything else in the method is boundary
conditions, sampling and optimisation.

Covers:
  * du/dx and d2u/dx2 with `create_graph=True` (Idiom A, what the papers use)
  * the same thing with torch.func transforms (Idiom B, what composes better)
  * a finite-difference check and torch.autograd.gradgradcheck
  * the Burgers residual, and .backward() through all of it
  * the "zero gradient on the output bias" gotcha, reproduced deliberately
  * a timing comparison of the two idioms
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch                                                        # noqa: E402
from torch.func import functional_call, grad, jacfwd, jacrev, vmap  # noqa: E402

from pinnlab import MLP, d, setup                                   # noqa: E402
from pinnlab.derivatives import finite_difference_second            # noqa: E402

NU = 0.01 / torch.pi


def section(title):
    print(f"\n{'-' * 74}\n{title}\n{'-' * 74}")


def main():
    setup()                                # float64, CPU, seeded
    net = MLP(in_dim=2, out_dim=1, width=32, depth=3)
    print(f"network: {net.n_params() if hasattr(net, 'n_params') else sum(p.numel() for p in net.parameters())} parameters, dtype {torch.get_default_dtype()}")

    # ------------------------------------------------------------------ Idiom A
    section("Idiom A: torch.autograd.grad on a batch (what the papers use)")
    z = torch.rand(2048, 2, dtype=torch.float64) * 2 - 1        # (t, x) in [-1,1]^2
    z.requires_grad_(True)
    u = net(z)
    u_z = d(u, z)                                                # (N,2): [du/dt, du/dx]
    u_t, u_x = u_z[:, 0:1], u_z[:, 1:2]
    u_xx = d(u_x, z)[:, 1:2]

    print(f"  u    {tuple(u.shape)}    u_t {tuple(u_t.shape)}   u_x {tuple(u_x.shape)}   u_xx {tuple(u_xx.shape)}")
    print("  note the THREE graph traversals: u -> u_x -> u_xx, and .backward() will make a fourth")

    # ------------------------------------------------- check against finite differences
    section("Sanity check 1: finite differences (your independent oracle)")
    x0 = torch.rand(64, 1, dtype=torch.float64) * 2 - 1
    t0 = torch.rand(64, 1, dtype=torch.float64) * 2 - 1

    def u_of_x(xv):
        return net(torch.cat([t0, xv], dim=1))

    fd = finite_difference_second(u_of_x, x0, h=1e-5)
    zc = torch.cat([t0, x0], dim=1).requires_grad_(True)
    ad = d(d(net(zc), zc)[:, 1:2], zc)[:, 1:2]
    rel = (torch.linalg.norm(fd - ad) / torch.linalg.norm(ad)).item()
    print(f"  relative L2 |FD - autodiff| for u_xx : {rel:.3e}   (expect ~1e-6 in float64)")
    assert rel < 1e-4, "autodiff graph is wrong"

    # ------------------------------------------------------- gradgradcheck
    section("Sanity check 2: torch.autograd.gradgradcheck (the rigorous version)")
    small = MLP(in_dim=1, out_dim=1, width=8, depth=2)
    xs = torch.linspace(-0.5, 0.5, 6, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    ok = torch.autograd.gradgradcheck(lambda a: small(a), (xs,), eps=1e-6, atol=1e-5)
    print(f"  gradgradcheck passed: {ok}   (it REQUIRES float64 - this is why chapter 04 insists)")

    # ------------------------------------------------------------------ Idiom B
    section("Idiom B: torch.func on a single point, then vmap (what composes)")
    params = dict(net.named_parameters())

    def u_fn(p, zz):                       # zz: (2,) -> scalar. ONE point, like the paper.
        return functional_call(net, p, (zz.unsqueeze(0),)).squeeze()

    def u_xx_single(p, zz):
        H = jacfwd(jacrev(u_fn, argnums=1), argnums=1)(p, zz)     # (2,2) input-Hessian
        return H[1, 1]                                            # d2u/dx2

    zz = z.detach()
    u_xx_B = vmap(u_xx_single, in_dims=(None, 0))(params, zz).reshape(-1, 1)
    #                          ^^^^^^^ in_dims in torch.func; JAX spells it in_axes. You WILL hit this.
    rel = (torch.linalg.norm(u_xx_B - u_xx.detach()) / torch.linalg.norm(u_xx.detach())).item()
    print(f"  Idiom A vs Idiom B disagreement : {rel:.3e}   (should be at machine precision)")

    # --------------------------------------------------------- the Burgers residual
    section("The Burgers residual, and the parameter gradient through it")
    z = torch.rand(4096, 2, dtype=torch.float64)
    z[:, 1] = z[:, 1] * 2 - 1                                     # t in [0,1], x in [-1,1]
    z.requires_grad_(True)

    u = net(z)
    g = d(u, z)
    u_t, u_x = g[:, 0:1], g[:, 1:2]
    u_xx = d(u_x, z)[:, 1:2]

    r = u_t + u * u_x - NU * u_xx                                 # <-- THE PINN RESIDUAL
    loss = (r ** 2).mean()
    loss.backward()

    finite = all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    n_none = sum(1 for p in net.parameters() if p.grad is None)
    print(f"  mean residual^2 at initialisation : {float(loss.detach()):.6e}")
    print(f"  all parameter gradients finite    : {finite}   (None gradients: {n_none})")
    print("  You have now written a PINN residual. The rest is BCs, sampling and optimisation.")

    # ----------------------------------------------------------- the classic gotcha
    section("Gotcha: the output bias gets an EXACTLY zero gradient from a pure d2/dx2 loss")
    tiny = MLP(in_dim=1, out_dim=1, width=8, depth=2)
    xs = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    uu = tiny(xs)
    uxx = d(d(uu, xs), xs)
    lossxx = (uxx ** 2).mean()
    try:
        torch.autograd.grad(lossxx, list(tiny.parameters()), retain_graph=True)
        print("  (no error on this build)")
    except RuntimeError as exc:
        print(f"  RuntimeError as expected: {str(exc)[:96]}...")
    gs = torch.autograd.grad(lossxx, list(tiny.parameters()), allow_unused=True)
    zeros = [i for i, gg in enumerate(gs) if gg is None or float(gg.abs().max()) == 0.0]
    print(f"  with allow_unused=True, parameter indices with an exactly-zero gradient: {zeros}")
    print("  A constant bias vanishes under d2/dx2. It disappears once you add BC/IC terms -")
    print("  which is exactly why it ambushes you on your first residual-only test.")

    # --------------------------------------------------------------- timing
    section("Timing: Idiom A vs Idiom B for the same Laplacian, 4096 points")
    for name, fn in (("A: autograd.grad(batch)", "A"), ("B: vmap(jacfwd(jacrev))", "B")):
        zt = torch.rand(4096, 2, dtype=torch.float64).requires_grad_(True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(5):
            if fn == "A":
                uu = net(zt)
                gx = d(uu, zt)[:, 1:2]
                _ = d(gx, zt)[:, 1:2]
            else:
                _ = vmap(u_xx_single, in_dims=(None, 0))(params, zt.detach())
        print(f"  {name:26s} {(time.perf_counter() - t0) / 5 * 1e3:8.2f} ms/call")
    print("\n  Neither is 'the' answer. A reads like the papers; B reads like the PDE and")
    print("  exploits forward-over-reverse, which is the efficient order when input dim is small.")

    print("\nNext: scripts/03_spectral_bias.py - why a plain MLP will not learn your shock.")


if __name__ == "__main__":
    main()
