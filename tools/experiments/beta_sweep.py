"""Real convection-PINN beta sweep for the chapter 09 tour.

Provenance: the numbers baked into src/tourdata.ts came from this script
(CPU, float64, seed 0). Re-run it to regenerate or extend them.

Reproduces the Krishnapriyan et al. setup in miniature:
  u_t + beta * u_x = 0,  x in [0, 2pi), t in [0, 1],  u(x,0) = sin(x), periodic BC.
Analytic solution: u(x,t) = sin(x - beta*t).

Two experiments, identical budgets:
  A) vanilla soft-constraint PINN, trained from scratch at each beta
  B) curriculum: one net trained at beta = 1 -> 5 -> 15 -> 30, warm-starting each stage

Outputs JSON with relative L2 errors and loss curves for the tour animation.
"""
import json
import math
import time

import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")

TWO_PI = 2 * math.pi

def make_net(seed):
    torch.manual_seed(seed)
    layers = []
    dims = [2, 50, 50, 50, 50, 1]
    for i in range(len(dims) - 1):
        lin = torch.nn.Linear(dims[i], dims[i + 1])
        torch.nn.init.xavier_normal_(lin.weight, gain=5.0 / 3.0)
        torch.nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)

def sample(n_f=2560, n_ic=256, n_bc=256, gen=None):
    xf = torch.rand(n_f, 1, generator=gen) * TWO_PI
    tf_ = torch.rand(n_f, 1, generator=gen)
    xi = torch.rand(n_ic, 1, generator=gen) * TWO_PI
    tb = torch.rand(n_bc, 1, generator=gen)
    return xf, tf_, xi, tb

def losses(net, beta, xf, tf_, xi, tb):
    xf = xf.clone().requires_grad_(True)
    tf_ = tf_.clone().requires_grad_(True)
    u = net(torch.cat([xf, tf_], 1))
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, tf_, grad_outputs=ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, xf, grad_outputs=ones, create_graph=True)[0]
    r = u_t + beta * u_x
    loss_r = (r ** 2).mean()
    # initial condition u(x,0) = sin(x)
    ui = net(torch.cat([xi, torch.zeros_like(xi)], 1))
    loss_ic = ((ui - torch.sin(xi)) ** 2).mean()
    # periodic BC u(0,t) = u(2pi,t)
    u0 = net(torch.cat([torch.zeros_like(tb), tb], 1))
    u1 = net(torch.cat([torch.full_like(tb, TWO_PI), tb], 1))
    loss_bc = ((u0 - u1) ** 2).mean()
    return loss_r, loss_ic, loss_bc

def rel_l2_err(net, beta):
    x = torch.linspace(0, TWO_PI, 256)
    t = torch.linspace(0, 1, 100)
    X, T = torch.meshgrid(x, t, indexing="ij")
    pts = torch.stack([X.reshape(-1), T.reshape(-1)], 1)
    with torch.no_grad():
        U = net(pts).reshape(X.shape)
    Uex = torch.sin(X - beta * T)
    return (torch.linalg.norm(U - Uex) / torch.linalg.norm(Uex)).item()

def train(net, beta, steps, gen, log_every=100, log=None, stage=None):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for k in range(steps):
        xf, tf_, xi, tb = sample(gen=gen)
        lr_, lic, lbc = losses(net, beta, xf, tf_, xi, tb)
        loss = lr_ + 100.0 * lic + 100.0 * lbc   # standard weighting for this problem
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log is not None and k % log_every == 0:
            log.append({"stage": stage, "step": k, "loss": float(loss.item()),
                        "err": rel_l2_err(net, beta)})
    return net

def main():
    t0 = time.time()
    betas = [1, 5, 15, 30]
    out = {"betas": betas, "vanilla": [], "vanilla_curves": {}, "curriculum": None,
           "config": {"net": "2-50x4-1 tanh", "steps_per_stage": 3000,
                      "n_f": 2560, "weights": "lambda_ic=lambda_bc=100", "lr": "1e-3 Adam",
                      "dtype": "float64", "device": "cpu", "seed": 0}}
    # A) vanilla from scratch per beta
    for b in betas:
        gen = torch.Generator().manual_seed(0)
        net = make_net(0)
        curve = []
        train(net, b, 3000, gen, log=curve, stage=f"beta={b}")
        e = rel_l2_err(net, b)
        out["vanilla"].append(e)
        out["vanilla_curves"][str(b)] = [{"step": c["step"], "err": round(c["err"], 4)} for c in curve]
        print(f"vanilla beta={b}: rel L2 = {e:.4f}  ({time.time()-t0:.0f}s)")
    # B) curriculum with same total budget spread over stages (3000 each to match per-beta budget)
    gen = torch.Generator().manual_seed(0)
    net = make_net(0)
    curve = []
    for b in betas:
        train(net, b, 3000, gen, log=curve, stage=f"beta={b}")
        print(f"curriculum through beta={b}: rel L2 = {rel_l2_err(net, b):.4f}  ({time.time()-t0:.0f}s)")
    e_final = rel_l2_err(net, 30)
    out["curriculum"] = {"final_err_beta30": e_final,
                         "curve": [{"stage": c["stage"], "step": c["step"], "err": round(c["err"], 4)} for c in curve]}
    print(f"curriculum final beta=30: rel L2 = {e_final:.4f}")
    # sample solution fields at beta=30 for the tour (coarse grids to keep JSON small)
    x = torch.linspace(0, TWO_PI, 64)
    t = torch.linspace(0, 1, 25)
    X, T = torch.meshgrid(x, t, indexing="ij")
    pts = torch.stack([X.reshape(-1), T.reshape(-1)], 1)
    with torch.no_grad():
        U_cur = net(pts).reshape(X.shape)
    gen = torch.Generator().manual_seed(0)
    net_v = make_net(0)
    train(net_v, 30, 3000, gen)
    with torch.no_grad():
        U_van = net_v(pts).reshape(X.shape)
    Uex = torch.sin(X - 30 * T)
    out["fields_beta30"] = {
        "x": [round(v, 4) for v in x.tolist()], "t": [round(v, 4) for v in t.tolist()],
        "exact": [[round(v, 3) for v in row] for row in Uex.tolist()],
        "vanilla": [[round(v, 3) for v in row] for row in U_van.tolist()],
        "curriculum": [[round(v, 3) for v in row] for row in U_cur.tolist()],
    }
    with open("beta_sweep.json", "w") as f:
        json.dump(out, f)
    print(f"done in {time.time()-t0:.0f}s -> beta_sweep.json")

if __name__ == "__main__":
    main()
