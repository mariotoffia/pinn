---
title: The PINN Recipe
subtitle: A 10-step checklist the field converged on — with defaults you can apply today
minutes: 20
---

# 10 — The PINN Recipe

Two source documents stand behind this chapter:

- **An Expert's Guide to Training Physics-informed Neural Networks
  (Wang, Sankaran, Wang & Perdikaris, Aug 2023)** — https://arxiv.org/abs/2308.08468 ·
  library **JAX-PI**: https://github.com/PredictiveIntelligenceLab/jaxpi
  **The most practically valuable paper in the field.** It bundles the whole
  fix-the-failure-modes literature into one pipeline, tests each component by switching it off
  ("ablation"), and releases a multi-GPU JAX implementation demonstrated on up to 256 GPUs.
- **PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks (JMLR 2024)** —
  https://arxiv.org/abs/2402.00326 · [PDF](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf)
  **The successor.** It explains the strange fact that **deeper PINNs get worse**, and traces
  it to standard initialisation interacting badly with the differentiated loss. PirateNets use
  adaptive skip connections, initialised so the network *starts shallow and learns its own
  depth*, plus a physics-informed initialisation of the final layer. Now the default
  architecture in JAX-PI.

Work through the checklist **in order.** Each step is cheap, and skipping the early ones makes
the later ones useless. **No single step carries the result — it is the whole stack.** The Expert's
Guide's Allen–Cahn ablation shows that removing any single component makes the result worse.

---

## Step 0 — Non-dimensionalise the PDE. Always. First.

Rescale the problem so that `x, t` and `u` are all of order 1.

Glorot/He initialisation *assumes* inputs in a moderate range. If `t ∈ [0, 1e6]` or `ν = 1e-6`,
your initialisation is wrong and your loss terms are numerically incomparable **before training
even starts**.

The Expert's Guide is blunt: this is **the highest-value single action, and it costs nothing.**
If a PINN diverges immediately, check this first.

---

## Step 1 — Architecture

- **Width 128–512, depth 3–6 hidden layers.** Too narrow or shallow lacks capacity; too wide or
  deep becomes impossible to optimise. **Do not scale up on instinct** — Urbán et al.
  (JCP 2025) find that 2–3 hidden layers are enough once the optimisation is right.
- **`tanh` activation.** Alternatives: sine (SIREN), GELU. **Never ReLU** — its second
  derivative is zero, which empties the physics term of meaning.
- **Glorot (Xavier) initialisation** for the dense layers. Note that PyTorch's default is
  ReLU-tuned Kaiming; for tanh, `torch.nn.init.calculate_gain('tanh')` = 5/3.
- If a plain MLP stalls, consider the **modified MLP** (the Wang/Teng/Perdikaris gated
  architecture) or **PirateNets**.

---

## Step 2 — Fourier feature embedding

Before the first layer, map the inputs through

$$\gamma(x) = \begin{bmatrix}\cos(Bx)\\ \sin(Bx)\end{bmatrix}, \qquad B_{ij} \sim \mathcal{N}(0, \sigma^2)$$

**This is the direct antidote to spectral bias** (Chapter 05 §5.6).

- **Recommended σ between 1 and 10.** Too small → blurry predictions; too large →
  salt-and-pepper noise.
- Ideally σ would match the frequency content of the solution — which you do not know for a
  forward problem. Hence the "moderately large" rule of thumb.
- **Use separate σ for space and time** when their scales differ.

```python
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, n_features=64, sigma=2.0):
        super().__init__()
        # NOT trainable: B is a fixed random projection
        self.register_buffer("B", torch.randn(in_dim, n_features) * sigma)
    def forward(self, z):
        p = 2 * torch.pi * z @ self.B
        return torch.cat([torch.cos(p), torch.sin(p)], dim=-1)
```

---

## Step 3 — Random weight factorisation (RWF)

Rewrite every neuron's weight vector as a learned scale times a learned direction:

$$w^{(k,l)} = \exp(s^{(k,l)}) \cdot v^{(k,l)}, \qquad s^{(k,l)} \sim \mathcal{N}(\mu, \sigma^2)$$

training `s` and `v` separately. **Recommended μ = 1.0, σ = 0.1** (the guide also mentions
μ = 0.5).

A drop-in replacement for `nn.Linear`. Consistently helpful, essentially free.
Source: https://arxiv.org/abs/2210.01274

---

## Step 4 — Enforce what you can exactly; penalise only the rest

**Every constraint you build into the network is one loss term and one weight you no longer
have to balance. This is the cheapest reliability win available.**

- **Periodic boundaries** — build them into the architecture: feed the network
  `v(x) = (cos ωx, sin ωx)` with `ω = 2π/P`. Periodicity then holds *exactly*, and the boundary
  loss term disappears. The Expert's Guide enforces exact periodicity in every experiment where
  it applies.
- **Dirichlet boundaries** — the Lagaris construction `u_θ(x) = A(x) + F(x)·N_θ(x)`, where `A`
  satisfies the boundary values and `F` vanishes on the boundary. For a 1D problem on [0,1]
  with `u(0)=a, u(1)=b`: `u_θ(x) = a(1−x) + bx + x(1−x)·N_θ(x)`.
- **Arbitrary geometry** — approximate distance functions (R-functions, mean-value
  potentials): **Sukumar & Srivastava, CMAME 389:114333** — https://arxiv.org/abs/2104.08426 —
  generalises Lagaris so Dirichlet/Neumann/Robin conditions hold exactly on complex domains.
  **The practical route to hard constraints.**
- **Constrained optimisation** — hPINN (**Lu, Pestourie, Yao, Wang, Verdugo & Johnson, 2021**) —
  https://arxiv.org/abs/2102.04626 — penalty and augmented-Lagrangian methods instead of plain
  soft penalties, demonstrated on topology optimisation in optics. Better-satisfied constraints
  *and* better objectives.

---

## Step 5 — Causal weighting for time-dependent problems

Split [0, T] into M ordered segments and weight segment i by

$$w_i = \exp\left(-\epsilon\sum_{k<i}\mathcal{L}^k_r(\theta)\right), \qquad \epsilon = 1.0 \text{ (default)}$$

**Stop-gradient on `w_i`** (the weights are treated as constants during backprop). Tune ε by
watching whether the weights ever wake up: too large and the later segments never train; too
small and you are back to the vanilla loss.

```python
w = torch.exp(-eps * torch.cumsum(
        torch.cat([torch.zeros(1), seg_losses[:-1]]), dim=0)).detach()
loss_r = (w * seg_losses).mean()
```

---

## Step 6 — Adaptive global loss balancing

Every `f = 1000` steps, recompute the weights from gradient sizes:

$$\hat\lambda_i = \frac{\sum_j \|\nabla_\theta \mathcal{L}_j\|}{\|\nabla_\theta \mathcal{L}_i\|}, \qquad \lambda^{\text{new}} = \alpha\lambda^{\text{old}} + (1-\alpha)\hat\lambda, \quad \alpha = 0.9$$

**Stop-gradient on λ.** The guide finds gradient-norm balancing and NTK balancing comparable,
and **recommends the gradient-based scheme first** — it is cheaper.

Per-point schemes are complementary, and can be layered on top:
- **SA-PINN** (a trainable weight per collocation point, trained adversarially) —
  https://arxiv.org/abs/2009.04544
- **RBA** (gradient-free running average of each point's residual size) —
  https://arxiv.org/abs/2307.00379 — cheaper than SA-PINN, and comparable or better in
  practice.

---

## Step 7 — Optimiser and schedule

- **Adam**, initial learning rate **1e-3**, **exponential decay** (the guide uses decay 0.9
  every 2000 steps).
- **No weight decay.** The guide explicitly warns it hurts accuracy on forward problems.
- **Then L-BFGS to finish**, with `line_search_fn="strong_wolfe"`, full batch, a **frozen
  collocation set**, in float64. Rathore et al. (ICML 2024) establish that **Adam → L-BFGS
  beats either alone.** Raissi's original used pure L-BFGS, and it worked for Burgers; for
  anything harder you need the Adam phase first.
- **The 2025–2026 frontier:** if you need better than ~1e-5, go second-order — SSBroyden
  ([Kiyani et al.](https://arxiv.org/abs/2501.16371)), NysNewton-CG
  ([Rathore et al.](https://arxiv.org/abs/2402.01868)), gradient alignment
  ([Wang et al.](https://arxiv.org/abs/2502.00604)), or DSGNAR
  ([Webb et al. 2026](https://arxiv.org/abs/2607.02194)). **This is where the accuracy ceiling
  is currently being broken.**

---

## Step 8 — Sampling

**Resample the collocation points randomly at every iteration.** Batch size ~4096 in the
guide's experiments.

A fixed full-batch collocation set lets the network **overfit the residual**: `ℒ_r` goes to
zero on your fixed points while the true residual between them stays large. Random resampling
both regularises and saves memory. The guide is emphatic: *"we strongly recommend using random
sampling in all PINN simulations."*

**Except during the L-BFGS phase** — freeze the set there, or you poison the curvature
estimate.

**Adaptive sampling is often worth more than a bigger network.** Residual-based adaptive
refinement (RAR) adds points where the residual is largest. DeepXDE ships it — compare
`Burgers.py` (55 lines) with `Burgers_RAR.py` (64 lines); **the diff between them is the
lesson.**

Uniform random sampling is a weak default. Prefer **Latin hypercube or Sobol** sequences, plus
RAR near shocks and steep gradients.

---

## Step 9 — Curriculum and time-marching for hard problems

Two complementary ladders:

- **Parameter curriculum** (Krishnapriyan): train at easy parameter values first, then restart
  from those weights at harder values. **Essential for high Reynolds / Péclet / convection.**
- **Time marching:** split [0, T] into windows; solve each window using the previous window's
  final state as the new initial condition, restarting from the previous weights. Apply causal
  weighting *inside* each window too — the guide notes causality violations still occur within
  a single window.

---

**Run Steps 4 and 9 as notebooks.** Moseley's FBPINNs repo (MIT) ships teaching notebooks
that are precisely these steps, live:
[hard constraints on the oscillator](https://colab.research.google.com/github/benmoseley/FBPINNs/blob/main/examples/2.%20Using%20hard%20constraints%20-%201D%20harmonic%20oscillator.ipynb "Step 4 of the recipe, live: Lagaris-style hard constraints on the 1D oscillator")
(Step 4) and
[subdomain scheduling on Burgers](https://colab.research.google.com/github/benmoseley/FBPINNs/blob/main/examples/4.%20Using%20subdomain%20scheduling%20-%20%281%2B1%29D%20Burgers%27%20equation.ipynb "Domain decomposition as an engineering cure for multiscale and propagation failures — JAX; a Colab CPU session handles it")
(Step 9's decomposition cousin). First cell installs from GitHub; JAX; runs on a Colab CPU.

---

## Step 10 — Validate like a numerical analyst, not like an ML engineer

**A low training loss is not evidence of correctness.** Check:

- **(a)** the residual on a **fresh, dense** set of points — not the training set;
- **(b)** the initial and boundary conditions explicitly, plotted;
- **(c)** conserved quantities (mass, energy, momentum) over time;
- **(d)** **several random seeds** — PINN seed-to-seed variance is large and rarely reported;
- **(e)** a coarse classical solution, or a **manufactured solution**, wherever you can get
  one. (Method of manufactured solutions: pick any `u` you like, substitute it into the PDE to
  get the source term `f` that makes it exact, then solve for `u`. Free ground truth for any
  operator you can differentiate.)

**Always plot prediction, reference and error side by side.** A PINN that looks right and is 5%
wrong is the standard failure mode — only the error panel shows it.

---

## The numbers to calibrate against

Table 1 of the Expert's Guide — the best relative L² errors after extensive hyperparameter
sweeps. This is the state of the art as of 2023, achieved by the people who invented these
techniques, on multi-GPU JAX:

| Problem | Relative L² error |
|---|---|
| Allen–Cahn | 5.37 × 10⁻⁵ |
| Advection | 6.88 × 10⁻⁴ |
| Stokes flow | 8.04 × 10⁻⁵ |
| **Kuramoto–Sivashinsky** | **1.61 × 10⁻¹** |
| **Lid-driven cavity (Re = 3200)** | **1.58 × 10⁻¹** |
| **Navier–Stokes in a torus** | **2.45 × 10⁻¹** |

**Read the bottom three carefully. On genuinely hard problems, the best-tuned PINN in the world
in 2023 had 16–25% relative error.** That is the calibration point. The 2026 optimiser
results (DSGNAR) suggest this ceiling is finally moving — verify on your own problem before
believing it.

---

## The recipe as code

```python
# Step 0: non-dimensionalise — do this on paper, before you write anything
# Step 1–3: architecture
net = FourierMLP(in_dim=2, width=256, depth=4, sigma=2.0, rwf=True)   # tanh, Glorot, RWF
# Step 4: hard-constrain what you can
u = lambda z: hard_bc(z, net(fourier(z)))
# Step 8: resample every iteration
for step in range(n_adam):
    pts = sample_collocation(n=4096)                 # Latin hypercube / Sobol
    losses = residual_loss(u, pts), bc_loss(u), ic_loss(u)
    if step % 1000 == 0:                             # Step 6
        lam = update_grad_norm_weights(lam, losses, alpha=0.9)
    if time_dependent:                               # Step 5
        losses = apply_causal_weights(losses, eps=1.0)
    loss = sum(l_i * w_i for l_i, w_i in zip(losses, lam))
    adam.step(loss)                                  # Step 7: lr 1e-3, exp decay, no weight decay
pts = sample_collocation(n=20000)                    # FREEZE for L-BFGS
lbfgs = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe")
lbfgs.step(closure)
validate_like_a_numerical_analyst(u)                 # Step 10
```

`starter/pinnlab/` implements every one of these steps as a small, readable module.
`06_pinn_burgers.py` switches them on and off with flags, so you can reproduce the ablation
yourself.
