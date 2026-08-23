"""Time-marching rescue at beta=30 for the chapter 09 tour (real run).

Provenance: the numbers baked into src/tourdata.ts came from this script
(CPU, float64, seed 0). Re-run it to regenerate or extend them.

Split t in [0,1] into W windows. Solve each window with a PINN whose initial
condition is the previous window's prediction at the window boundary,
warm-starting the weights. Stitch and compare against sin(x - 30 t).
"""
import json, math, time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
TWO_PI = 2 * math.pi
BETA = 30.0
W = 10           # windows
STEPS = 1500     # Adam steps per window

def make_net(seed):
    torch.manual_seed(seed)
    dims = [2, 50, 50, 50, 50, 1]
    layers = []
    for i in range(len(dims) - 1):
        lin = torch.nn.Linear(dims[i], dims[i + 1])
        torch.nn.init.xavier_normal_(lin.weight, gain=5.0 / 3.0)
        torch.nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)

def train_window(net, t0, t1, ic_fn, steps, gen):
    """ic_fn(x) -> u at t0. Network input uses local time tau in [0,1]."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    dt = t1 - t0
    for k in range(steps):
        xf = torch.rand(2560, 1, generator=gen) * TWO_PI
        tauf = torch.rand(2560, 1, generator=gen)
        xi = torch.rand(256, 1, generator=gen) * TWO_PI
        taub = torch.rand(256, 1, generator=gen)
        xf_ = xf.clone().requires_grad_(True)
        tauf_ = tauf.clone().requires_grad_(True)
        u = net(torch.cat([xf_, tauf_], 1))
        ones = torch.ones_like(u)
        u_tau = torch.autograd.grad(u, tauf_, grad_outputs=ones, create_graph=True)[0]
        u_x = torch.autograd.grad(u, xf_, grad_outputs=ones, create_graph=True)[0]
        r = u_tau / dt + BETA * u_x          # d/dt = (1/dt) d/dtau
        loss_r = (r ** 2).mean()
        ui = net(torch.cat([xi, torch.zeros_like(xi)], 1))
        loss_ic = ((ui - ic_fn(xi)) ** 2).mean()
        u0 = net(torch.cat([torch.zeros_like(taub), taub], 1))
        u1 = net(torch.cat([torch.full_like(taub, TWO_PI), taub], 1))
        loss_bc = ((u0 - u1) ** 2).mean()
        loss = loss_r + 100.0 * loss_ic + 100.0 * loss_bc
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def main():
    t0c = time.time()
    gen = torch.Generator().manual_seed(0)
    edges = np.linspace(0, 1, W + 1)
    nets = []
    net = make_net(0)
    ic = lambda x: torch.sin(x)
    win_errs = []
    for w in range(W):
        a, b = float(edges[w]), float(edges[w + 1])
        net = train_window(net, a, b, ic, STEPS, gen)   # warm start: same net object
        nets.append([p.detach().clone() for p in net.parameters()])
        # window-local error at end of window
        x = torch.linspace(0, TWO_PI, 256)
        with torch.no_grad():
            up = net(torch.stack([x, torch.ones_like(x)], 1)).squeeze()
        ue = torch.sin(x - BETA * b)
        e = (torch.linalg.norm(up - ue) / torch.linalg.norm(ue)).item()
        win_errs.append(e)
        print(f"window {w+1}/{W} [{a:.2f},{b:.2f}]  end-of-window rel L2 = {e:.4f}  ({time.time()-t0c:.0f}s)")
        # next window's IC = this window's prediction at tau=1 (frozen snapshot)
        snap = [p.detach().clone() for p in net.parameters()]
        arch = make_net(0)
        with torch.no_grad():
            for p, q in zip(arch.parameters(), snap):
                p.copy_(q)
        arch.eval()
        ic = (lambda m: (lambda xx: m(torch.cat([xx, torch.ones_like(xx)], 1)).detach()))(arch)
    # stitched global error on a dense grid
    xg = torch.linspace(0, TWO_PI, 256)
    tg = torch.linspace(0, 1, 100)
    U = torch.zeros(256, 100)
    with torch.no_grad():
        for j, tv in enumerate(tg.tolist()):
            w = min(int(tv * W), W - 1)
            a, b = float(edges[w]), float(edges[w + 1])
            tau = (tv - a) / (b - a)
            arch = make_net(0)
            for p, q in zip(arch.parameters(), nets[w]):
                p.copy_(q)
            U[:, j] = arch(torch.stack([xg, torch.full_like(xg, tau)], 1)).squeeze()
    X, T = torch.meshgrid(xg, tg, indexing="ij")
    Uex = torch.sin(X - BETA * T)
    err = (torch.linalg.norm(U - Uex) / torch.linalg.norm(Uex)).item()
    print(f"time-marching global rel L2 at beta=30: {err:.4f}   ({time.time()-t0c:.0f}s)")
    # coarse field for the tour
    out = {"beta": 30, "windows": W, "steps_per_window": STEPS,
           "global_rel_l2": err, "end_of_window_errs": [round(e, 4) for e in win_errs],
           "field": {"x": [round(v, 4) for v in xg[::4].tolist()],
                     "t": [round(v, 4) for v in tg[::4].tolist()],
                     "u": [[round(U[i, j].item(), 3) for j in range(0, 100, 4)] for i in range(0, 256, 4)]}}
    with open("beta_march.json", "w") as f:
        json.dump(out, f)
    print("saved beta_march.json")

if __name__ == "__main__":
    main()
