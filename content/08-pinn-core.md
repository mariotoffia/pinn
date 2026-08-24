---
title: PINN Core
subtitle: What a PINN is, precisely — with Burgers written out in full
minutes: 26
---

# 08 — PINN Core

This chapter is **content, not a link list.** Read it slowly. Everything in Chapters 09–14 is a
reaction to what is written here.

---

## 8.1 The pre-history (PINNs were not invented in 2019)

The core idea — minimise the equation's error at sample points — is from **1994**. The
trial-function / hard-constraint idea is from **1997**. The approximation theory that justifies
both is from **1989–1990**. What 2017–2019 added was reverse-mode autodiff at scale, GPUs, and a
compelling story.

- **Dissanayake & Phan-Thien (1994)** — https://doi.org/10.1002/cnm.1640100303 — the first
  clear statement of the PINN loss: represent the solution with an MLP, form the PDE residual
  at interior points and the boundary residual at boundary points, minimise the sum of squares.
  **The PINN objective, 23 years early.** It lacked only autodiff and compute.
- **Lagaris, Likas & Fotiadis (1997/1998)** — https://arxiv.org/abs/physics/9705023 ·
  [open PDF](https://www.cs.uoi.gr/~lagaris/papers/TNN-LLF.pdf) · IEEE TNN 9(5):987–1000 —
  introduced the **trial solution** `u_t(x) = A(x) + F(x)·N_θ(x)`, where `A` satisfies the
  boundary/initial conditions exactly and `F` vanishes on the boundary — so **the boundary
  conditions hold by construction**, not as a penalty. The ancestor of every hard-constraint
  PINN, and arguably still the better idea: it deletes an entire loss term and its weight.
- **Hornik, Stinchcombe & White (1989)** — https://doi.org/10.1016/0893-6080%2889%2990020-8 —
  the standard universal-approximation result: an MLP can approximate any reasonable function.
  Necessary for PINNs, but **not sufficient**.
- **Hornik, Stinchcombe & White (1990)** — https://doi.org/10.1016/0893-6080%2890%2990005-6 —
  **this is the one that matters.** Approximation in **Sobolev norms**: a network can
  approximate a function *and its derivatives up to order k* at the same time, provided the
  activation is smooth enough. Without this result, minimising a residual containing `∂²u/∂x²`
  would have no theoretical licence at all. It also explains why ReLU is disqualified: its
  second derivative is zero almost everywhere.
- **Pinkus, Acta Numerica 1999** — https://doi.org/10.1017/S0962492900002919 — the rigorous
  quantitative treatment. The reference to reach for when someone waves "universal
  approximation" at you as though it settled anything.

> **One-line takeaway: universal approximation says a good set of weights *exists*. Nothing in
> this theory says gradient descent will *find* it. Almost every failure in Chapter 09 lives in
> that gap.**

---

## 8.2 The canonical papers

- **Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear PDEs** —
  https://arxiv.org/abs/1711.10561 — the **forward** problem. Introduces both the
  *continuous-time* model (sample points across space and time) and the *discrete-time* model
  (implicit Runge–Kutta steps baked into the network). Take from it: the loss decomposition,
  the Burgers/Schrödinger/Allen–Cahn benchmarks, and the sensitivity tables over the number of
  data points, collocation points, depth and width.
- **Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear PDEs** —
  https://arxiv.org/abs/1711.10566 — the **inverse** problem: unknown PDE coefficients λ become
  trainable parameters, optimised together with the network weights. **This is the part of the
  PINN story that has held up best.**
- **PINNs: A deep learning framework for solving forward and inverse problems involving
  nonlinear PDEs (Raissi, Perdikaris & Karniadakis, JCP 2019)** —
  https://doi.org/10.1016/j.jcp.2018.10.045 — the consolidated, peer-reviewed merge of Parts I
  and II. **This is *the* citation** — around 15,000 citations, the single most influential
  paper in scientific machine learning. Unusually clear and short; read it end to end once.
- **maziarraissi/PINNs** — https://github.com/maziarraissi/PINNs ·
  [site](https://maziarraissi.github.io/PINNs/) — the original reference implementation,
  TensorFlow 1.x. **Explicitly no longer maintained**; the README itself points readers to
  modern implementations. **Read it as a historical artifact; never run it.** It is ~200 lines,
  and it shows exactly how little machinery a PINN needs.
- **Physics-informed machine learning (Karniadakis et al., Nature Reviews Physics 2021)** —
  https://www.nature.com/articles/s42254-021-00314-5 — the field-defining survey. Its lasting
  contribution is the taxonomy of **three ways to inject physics into ML**: *observational
  bias* (data), *inductive bias* (architecture — symmetry, hard constraints), and *learning
  bias* (soft penalties — PINNs proper). It frames PINNs as **one point in a design space, not
  the design space.** Take that taxonomy with you; it organises everything else you will read.
- **Physics-Informed Neural Networks and Extensions (Raissi et al., Aug 2024)** —
  https://arxiv.org/abs/2408.16806 — a short retrospective by the original authors, seven years
  on. Valuable precisely because it is written by the people with the most reason to
  overstate — and it is noticeably more measured than the 2019 paper.

---

## 8.3 The problem being solved

Take a general time-dependent PDE in **strong form**, on a spatial domain Ω ⊂ ℝᵈ over the time
interval [0, T]:

$$\partial_t u(x,t) + \mathcal{N}[u](x,t) = 0, \qquad (x,t) \in \Omega \times (0,T]$$
$$\mathcal{B}[u](x,t) = g(x,t), \qquad (x,t) \in \partial\Omega \times (0,T]$$
$$u(x,0) = u_0(x), \qquad x \in \Omega$$

Here `𝒩` is a (generally nonlinear) differential operator in `x` — the "physics" part of the
equation. `ℬ` is a boundary operator: the identity for Dirichlet conditions, the normal
derivative for Neumann, and so on.

**"Strong form" carries real weight here.** A PINN evaluates `𝒩[u]` **pointwise**, so the
network must be classically differentiable, to the order of the PDE, everywhere. That is a
genuine tightening of what FEM requires — and one reason PINNs struggle on sharp interfaces and
non-smooth solutions.

---

## 8.4 The network as a stand-in for the solution

Replace `u` by a smooth parametric function — a plain MLP:

$$u_\theta(z) = W^{(L+1)}\,\sigma\!\big(W^{(L)}\sigma(\cdots\sigma(W^{(1)}z + b^{(1)})\cdots) + b^{(L)}\big) + b^{(L+1)}, \quad z = (x,t)$$

with weights `θ = {W⁽ˡ⁾, b⁽ˡ⁾}` and activation `σ = tanh`.

**This is not a discretisation.** `u_θ` is a single smooth function defined on the whole domain,
differentiable as many times as you like. There is no mesh, no basis, no stencil. That is the
entire "mesh-free" claim — and it is true. But note the flip side: "mesh-free" also means *no
mesh to refine, no error estimator, and no convergence order to report.*

**The activation choice is a hard requirement, not a taste.** To evaluate a second-order
operator you need `σ` twice differentiable with a non-degenerate second derivative. ReLU has
`σ'' ≡ 0` almost everywhere — so any residual containing `u_xx` collapses to zero, and the
physics term teaches the network nothing.

---

## 8.5 The residual — the entire trick

$$r_\theta(x,t) := \partial_t u_\theta(x,t) + \mathcal{N}[u_\theta](x,t)$$

If `u_θ` were the exact solution, `r_θ` would be zero everywhere. Since it is not, the size of
`r_θ` measures how badly the network violates the physics — and it can be computed **anywhere,
at any point, with no data whatsoever.**

**The PDE becomes an infinite supply of free training targets.** That is the idea. Everything
else is engineering.

---

## 8.6 Collocation points

We cannot force `r_θ = 0` everywhere, so we sample points and enforce it there. These sample
points are called **collocation points**:

- `{(xʳᵢ, tʳᵢ)}` for `i = 1..N_r` inside Ω × (0,T] — the collocation (interior) points
- `{(xᵇᵢ, tᵇᵢ)}` for `i = 1..N_b` on the boundary ∂Ω × (0,T]
- `{x⁰ᵢ}` for `i = 1..N_0` in Ω at `t = 0` — initial-condition points
- `{(xᵈᵢ, tᵈᵢ, uᵈᵢ)}` for `i = 1..N_d` — any actual measurements you have (possibly none)

Sampling is typically **Latin hypercube** (Raissi's choice), uniform random, or Sobol. Note what
collocation points are *not*: they are not a mesh, they carry no connectivity, and **they can be
resampled at every iteration.** Chapter 10 argues you *should* resample — a fixed collocation
set lets the network overfit the residual.

---

## 8.7 The combined loss

$$\mathcal{L}(\theta) = \lambda_r\mathcal{L}_r + \lambda_{bc}\mathcal{L}_{bc} + \lambda_{ic}\mathcal{L}_{ic} + \lambda_d\mathcal{L}_d$$

with

$$\mathcal{L}_r = \frac{1}{N_r}\sum_i \big|r_\theta(x^r_i,t^r_i)\big|^2 \qquad \text{(physics)}$$
$$\mathcal{L}_{bc} = \frac{1}{N_b}\sum_i \big|\mathcal{B}[u_\theta](x^b_i,t^b_i) - g(x^b_i,t^b_i)\big|^2 \qquad \text{(boundary)}$$
$$\mathcal{L}_{ic} = \frac{1}{N_0}\sum_i \big|u_\theta(x^0_i,0) - u_0(x^0_i)\big|^2 \qquad \text{(initial)}$$
$$\mathcal{L}_d = \frac{1}{N_d}\sum_i \big|u_\theta(x^d_i,t^d_i) - u^d_i\big|^2 \qquad \text{(data)}$$

Then `θ* = argmin_θ ℒ(θ)`, found by gradient descent.

**Notice three things — all of Chapter 09 follows from them.**

**(i) This is soft-constrained multi-objective optimisation.** The initial and boundary
conditions are *penalties*, not constraints. Nothing forces `u_θ(x,0) = u₀(x)`. The network can
drive `ℒ_r` to near zero by finding a different, perfectly valid solution of the same PDE with
the wrong initial condition — `u ≡ 0` solves many PDEs. The λ weights decide which objective
wins, and **there is no principled way to set them in advance.** Raissi's original used
`λ_r = λ_bc = λ_ic = 1`. That happens to work for Burgers, and fails badly elsewhere.

**(ii) The terms have different physical units and wildly different gradient sizes.** `ℒ_r`
carries the units of the PDE operator; `ℒ_ic` carries the units of `u`. For a diffusion problem
with `ν = 1e-6`, the terms can differ by many orders of magnitude *before training starts*.
This is why non-dimensionalisation is a correctness requirement, not housekeeping.

**(iii) With `ℒ_d = 0` you have a pure forward solver; with few collocation points and lots of
data you have ordinary supervised regression. PINNs slide continuously between the two.** That
continuum is the conceptual contribution.

---

## 8.8 How autodiff supplies the derivatives

This is the part that engineers coming from standard deep learning most often get backwards.
Chapter 03 has the mechanics; here is what it costs.

- `∂u_θ/∂t` — one sweep through the network with respect to the `t` input. **Exact to machine
  precision.** Not a finite difference. No truncation error, no grid, no stability condition.
- `∂u_θ/∂x` — the same, with respect to `x`.
- `∂²u_θ/∂x²` — differentiate the *graph of* `∂u_θ/∂x` again. This nests autodiff: the graph
  for `u_xx` is roughly twice the size of the graph for `u_x`.

Then `r_θ = u_t + u·u_x − ν·u_xx` is assembled, squared, averaged — and only *then* do you call
`.backward()` on the whole thing. **You are backpropagating through a graph that already
contains two levels of differentiation.**

Practical consequences:
- **Cost grows with the PDE's order.** Fourth-order equations (biharmonic, Cahn–Hilliard,
  Kuramoto–Sivashinsky) cost noticeably more per point, and the nested graph's memory is
  usually the first thing to run out.
- **Forward mode is the cheap direction for the input derivatives** (few inputs); reverse mode
  is right for the weight gradient. `jacfwd(jacrev(...))` exploits this; chained
  `autograd.grad` calls do not.
- **Taylor-mode / higher-order AD** gives real speedups for high-order operators.
- **Everything is exact.** All the error is the network's approximation error plus the
  optimiser's failure to find the best weights. This is the strongest true claim PINNs have.

---

## 8.9 The canonical example: Burgers' equation, written out

Raissi et al.'s first example. One-dimensional viscous Burgers with `ν = 0.01/π`:

$$u_t + u\,u_x - \frac{0.01}{\pi}u_{xx} = 0, \quad x \in [-1,1],\ t \in (0,1]$$
$$u(0,x) = -\sin(\pi x), \qquad u(t,-1) = u(t,1) = 0$$

The solution develops a **sharp internal layer near x = 0 around t ≈ 0.4** — a near-shock that
classical methods need careful adaptive refinement to resolve. That is exactly why it was
chosen.

The residual:

$$f_\theta(t,x) := \partial_t u_\theta + u_\theta\,\partial_x u_\theta - \frac{0.01}{\pi}\partial_x^2 u_\theta$$

In the original code this is literally four lines:

```python
u    = neural_net(tf.concat([t, x], 1), weights, biases)
u_t  = tf.gradients(u, t)[0]
u_x  = tf.gradients(u, x)[0]
u_xx = tf.gradients(u_x, x)[0]
f    = u_t + u*u_x - (0.01/tf.pi)*u_xx
```

The loss, exactly as in the paper (equal weights; initial and boundary data merged into one
term):

$$\mathcal{L}(\theta) = \underbrace{\frac{1}{N_u}\sum_{i=1}^{N_u}\big|u_\theta(t^u_i,x^u_i) - u^i\big|^2}_{\mathcal{L}_u:\ \text{IC} \cup \text{BC}} + \underbrace{\frac{1}{N_f}\sum_{i=1}^{N_f}\big|f_\theta(t^f_i,x^f_i)\big|^2}_{\mathcal{L}_f:\ \text{physics}}$$

with `N_u = 100` points from the initial and boundary data, and `N_f = 10,000` collocation
points from a Latin hypercube over [0,1] × [-1,1].

**Reported configuration and result:** 9 layers, 20 neurons per hidden layer, `tanh`, **3021
parameters in total**, full-batch **L-BFGS**. Relative L² error: **6.7 × 10⁻⁴**.

One hundred data points plus the equation itself — and you recover a solution with a near-shock
to four correct digits, with no mesh. **That result is real, and it is why the field exploded.**
Hold it in mind next to Chapter 09, where the very same recipe applied to a *linear* convection
equation with β = 30 fails outright.

`starter/scripts/06_pinn_burgers.py` implements all of this in ~200 readable lines, with
Adam → L-BFGS, optional Fourier features and optional causal weighting, and reports the
relative L² error against a reference solution.

---

## 8.10 Forward vs inverse

**Forward.** The operator `𝒩` is fully known and there is little or no data term. You minimise
over θ alone. In other words: you are using a nonconvex optimiser to solve an equation that a
classical method would solve by assembling and inverting one (usually sparse, usually
well-behaved) linear system. **Expect to lose this fight** on any problem where a classical
method applies.

**Inverse.** The operator contains unknown physical parameters: `𝒩 = 𝒩_λ`. Promote λ to
trainable variables and minimise over (θ, λ) together. Now the data term is essential — the data
is what pins down λ. For Burgers, Raissi et al. write

$$f_\theta = u_t + \lambda_1 u\,u_x - \lambda_2 u_{xx}, \qquad \text{truth: } \lambda_1 = 1,\ \lambda_2 = 0.0031831$$

and recover, from 2000 scattered measurements:

| | λ₁ | λ₂ |
|---|---|---|
| Clean data | 0.99915 | 0.0031794 |
| 1% Gaussian noise | 1.00042 | 0.0032098 |

For the **Navier–Stokes cylinder wake**, from velocity data alone, they recover λ₁ to 0.078%
and λ₂ to 4.67% (clean data), 0.17% / 5.70% at 1% noise — **and reconstruct the pressure field,
which was never measured at all**, up to the additive constant pressure is only defined to.

**That last point is the headline of the whole PINN programme.** Recovering an unobserved
field and a physical constant from sparse, noisy, patchy measurements of a *different* field —
in a few dozen lines of code — is something no off-the-shelf classical tool does comfortably.
**Inverse problems and data assimilation are where PINNs earn their keep.**

---

## 8.11 What "physics-informed" actually buys you

Stated precisely, with the caveats attached:

1. **Data efficiency.** The residual term is an infinitely rich regulariser defined by the
   equation, not by data. A 3021-parameter network trained on 100 points does not overfit.
   *Caveat: it regularises toward the set of all PDE solutions, which may contain many —
   only the soft initial/boundary penalties pick out yours.*
2. **Mesh-free, and the cost barely grows with dimension.** Arbitrary geometry via point
   sampling. Grossmann et al. measured **no increase in cost going from 2D to 3D Poisson.**
   *Caveat: this is a Monte-Carlo-sampling property, not magic; the variance still hurts and
   the constant factor is large.*
3. **Native inverse and data-assimilation capability.** Unknown coefficients, unknown sources,
   unobserved fields, sparse noisy data — all handled by adding terms to one loss. **No adjoint
   solver to derive, no second optimisation loop.** This is the strongest case for PINNs, and
   it is a genuinely new capability, not a faster version of an old one.
4. **A smooth, differentiable solution object.** The output is a closed-form function you can
   evaluate, differentiate and compose anywhere, at constant cost per point, forever.
   Grossmann et al. found PINN *evaluation* 2–3 orders of magnitude faster than interpolating
   an FEM solution onto a new grid.
5. **Composability.** Extra physics — conservation laws, symmetries, another measurement type —
   is one more loss term. *Caveat: and one more weight to balance.*

**What it does not buy you:** competitive forward-solve time or accuracy on problems classical
methods handle. Convergence guarantees. Error estimators. Reliability. Reproducibility across
random seeds.

---

## 8.12 Run a first PINN in the browser — three hosted lanes

The local lane for this chapter is the marimo lab (`pinn lab 04`) and the guided tour above.
Prefer a hosted notebook — or want the same story told in another voice? These three are
verified and free:

- **ETH Zürich — Tutorial 03: PINN Training** — `notebook` `free` —
  [open in Colab](https://colab.research.google.com/github/camlab-ethz/AI_Science_Engineering/blob/main/Tutorial%2003%20-%20PINN%20Training.ipynb)
  — from *AI in the Sciences and Engineering* (Mishra & Moseley — the same Moseley as this
  chapter's workshop lab). PyTorch; the full course entry is in Chapter 12 §12.7. The repo has
  no license file: run the notebook in place, do not copy it.
- **Purdue ME 539 — hands-on 26.1 and 26.2** — `notebook` `free` `GPL-3.0` —
  [26.1](https://colab.research.google.com/github/PredictiveScienceLab/data-analytics-se/blob/master/lecturebook/lecture26/hands-on-26.1.ipynb) ·
  [26.2](https://colab.research.google.com/github/PredictiveScienceLab/data-analytics-se/blob/master/lecturebook/lecture26/hands-on-26.2.ipynb)
  — physics-informed regularisation for an ODE and a PDE, a Lagaris-style trial function (what
  §8.7 calls a hard constraint), and an Adam-vs-L-BFGS comparison. A live course, running
  Fall 2026.
- **Cornell Virtual Workshop notebooks (chishiki-ai)** — `notebook` `free` — the companion
  notebooks of §12.7's gentlest on-ramp, in plain PyTorch, updated August 2026:
  [02a — first PINN](https://colab.research.google.com/github/chishiki-ai/sciml-course/blob/main/SciML/02a_pinn.ipynb) ·
  [02b — Poisson](https://colab.research.google.com/github/chishiki-ai/sciml-course/blob/main/SciML/02b_poisson.ipynb) ·
  [02c — inverse heat](https://colab.research.google.com/github/chishiki-ai/sciml-course/blob/main/SciML/02c_inverse_heat.ipynb).
  (Notebook repo carries no license: run in place.)

→ **Chapter 09** is that list, in detail.
