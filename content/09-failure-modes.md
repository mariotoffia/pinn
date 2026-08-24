---
title: Why PINNs Fail
subtitle: The most important chapter on this path
minutes: 22
---

# 09 — Why PINNs Fail

The literature here is unusually good. **The field has been honest with itself in the technical
papers**, even when the abstracts and press releases were not. Read this chapter immediately
after Chapter 08, before you believe anything.

---

## 9.1 The foundational failure paper

- **Characterizing possible failure modes in physics-informed neural networks
  (Krishnapriyan, Gholami, Zhe, Kirby & Mahoney, NeurIPS 2021)** — `paper` `free` —
  https://arxiv.org/abs/2109.01050 ·
  [proceedings](https://proceedings.neurips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html) ·
  code https://github.com/a1k12/characterizing-pinns-failure-modes

**Read this second, right after the 2019 JCP paper.** The authors take deliberately simple
**linear** problems — 1D convection `u_t + β·u_x = 0`, reaction, reaction–diffusion — and show
that PINNs fail badly as the speed β or the reaction rate ρ grows. This happens even though the
*solution* stays simple, and the network has plenty of capacity to represent it.

**The decisive experiment:** they prove the failure is **not** about capacity. A network trained
by ordinary regression on the true solution fits it easily. The failure is **optimisation**: the
soft residual penalty makes the loss landscape ill-conditioned.

Their fixes: **curriculum training** (train at a small, easy β first, then restart from those
weights at progressively larger β) and **sequence-to-sequence learning** (predict one time
segment at a time instead of the whole space-time block at once). Both give **1–2 orders of
magnitude lower error.**

The march-forward-in-time idea returns again and again in later work. **It is arguably the
single most reliable practical fix in the field.**

---

## 9.2 Loss balancing: the λ's are not a detail

- **Understanding and mitigating gradient pathologies in PINNs (Wang, Teng & Perdikaris, 2020)** —
  https://arxiv.org/abs/2001.04536 · SIAM J. Sci. Comput. 43(5):A3055 ·
  [code](https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs) — diagnoses a
  **stiffness in the gradient flow**: the gradient sizes of the residual term and the boundary
  term can differ by orders of magnitude, so gradient descent effectively ignores one of the
  objectives. Proposes **learning-rate annealing** (auto-rebalancing the weights from gradient
  statistics) plus a gated network architecture. This is where the modern practice of adaptive
  loss weights begins.
- **When and why PINNs fail to train: A neural tangent kernel perspective (Wang, Yu & Perdikaris)** —
  https://arxiv.org/abs/2007.14527 · JCP 449:110768 — the theoretical companion. In the
  infinite-width limit, PINN training is governed by a **PINN NTK** — a kernel that splits into
  boundary and residual blocks whose eigenvalues differ enormously. The result: different parts
  of the problem are learned at wildly different speeds, which is exactly the symptom seen in
  practice. Leads to NTK-based adaptive weighting. **Take away this idea: "which parts of the
  solution get learned, and how fast" is set by an eigenvalue spectrum you can actually
  compute.**
- **Self-Adaptive PINNs using a Soft Attention Mechanism (McClenny & Braga-Neto)** —
  https://arxiv.org/abs/2009.04544 · JCP 474:111722 · [code](https://github.com/levimcclenny/SA-PINNs) —
  instead of a few global λ's, attach a **trainable weight to every single collocation point**,
  and train those weights by gradient *ascent* while the network descends — a minimax game. The
  weights organise themselves into a soft attention mask that concentrates on the difficult
  regions (shock fronts, boundary layers). Elegant, effective — and memory-hungry (one extra
  parameter per collocation point).
- **Residual-based attention and connection to information bottleneck theory in PINNs
  (Anagnostopoulos, Toscano, Stergiopulos & Karniadakis, 2023)** — https://arxiv.org/abs/2307.00379 ·
  CMAME 421:116805 — a **gradient-free** per-point weighting scheme: each point's weight is a
  running average of its recent residual size. Cheaper than SA-PINN, comparable or better in
  practice, and now a common default. (The information-bottleneck story is speculative; the
  weighting scheme is not.)

---

## 9.3 Causality: the network cheats on time

- **Respecting causality is all you need for training physics-informed neural networks
  (Wang, Sankaran & Perdikaris, 2022)** — https://arxiv.org/abs/2203.07404 ·
  [code](https://github.com/PredictiveIntelligenceLab/CausalPINNs)

**The single sharpest diagnosis in the literature.** A standard PINN minimises the residual over
the whole space-time domain *at once* — so the network is implicitly trying to satisfy the
equation at t = 1 before it has learned the solution at t = 0.1. **That is anti-causal — it
learns the future before the past — and it is why PINNs cannot handle chaotic or turbulent
dynamics.**

The fix is almost embarrassingly simple. Split the time axis into M ordered segments, and weight
segment i by

$$w_i = \exp\left(-\epsilon\sum_{k=1}^{i-1}\mathcal{L}_r^k(\theta)\right), \qquad \mathcal{L}_r = \frac{1}{M}\sum_i w_i\,\mathcal{L}_r^i$$

Segment i only gets meaningful weight once all earlier segments already have a small residual —
the loss itself now *enforces* that time is learned front to back. With this, the authors solve
chaotic systems (Kuramoto–Sivashinsky, Navier–Stokes) that vanilla PINNs simply cannot touch.

**Copy the formula. It costs three lines of code, and it often changes everything.**

---

## 9.4 Spectral bias and multiscale failure

- **On the Spectral Bias of Neural Networks (Rahaman et al., ICML 2019)** — https://arxiv.org/abs/1806.08734
- **Frequency Principle (Xu et al.)** — https://arxiv.org/abs/1901.06523 ·
  [overview](https://arxiv.org/abs/2201.07395)

**The critical observation:** networks learn low frequencies (smooth shapes) first — which is
the *exact opposite* of classical iterative solvers. Jacobi and Gauss–Seidel damp
**high**-frequency error fastest; that is the fact multigrid is built on. Networks damp **low**
frequencies fastest. So a century of intuition about iterative PDE solvers flips upside down —
and multiscale problems, which classical methods conquer by hierarchy, become the **worst
case** for PINNs.

- **On the eigenvector bias of Fourier feature networks (Wang, Wang & Perdikaris)** —
  https://arxiv.org/abs/2012.10047 · CMAME 384:113938 — connects spectral bias to the NTK
  eigenvalue spectrum, and shows that **random Fourier feature embeddings** shift the network's
  preferred frequencies to whatever band you choose via the parameter σ. The theory behind the
  most useful architectural fix.
- **Fourier Features (Tancik et al., NeurIPS 2020)** — https://arxiv.org/abs/2006.10739 — the
  source of `γ(x) = [cos(Bx), sin(Bx)]ᵀ`, `Bᵢⱼ ~ N(0, σ²)`. Not a PINN paper — but every modern
  PINN uses it.

---

## 9.5 The empirical pattern

| Regime | Why it breaks |
|---|---|
| **Fast convection** (`u_t + β·u_x = 0`, large β) | Krishnapriyan: the loss landscape degrades as β grows; the error explodes even though the solution stays simple |
| **Fast reaction / stiff terms** | Widely separated timescales ⇒ widely separated NTK eigenvalues ⇒ the fast mode hogs the gradient; the slow mode is never learned |
| **Long time horizons** | The causality violation compounds; the network learns a time-averaged fiction. Fix: march forward in time windows |
| **Sharp interfaces / near-shocks** (Allen–Cahn, small ε) | The strong form is nearly singular at the interface; spectral bias fights the thin high-frequency layer |
| **Multiscale / turbulent** | Spectral bias + causality + loss imbalance, all at once |
| **Chaotic** (Kuramoto–Sivashinsky, Lorenz) | Residual minimisation has no way to track diverging trajectories; even with causal weighting, the best published result is ~16% relative L² on KS |

- **The curse of dimensionality: what lies beyond the capabilities of PINNs (2025)** —
  https://arxiv.org/abs/2511.08561 — a clean minimal counterexample: on simple RC low-pass
  filter circuits, PINNs predict the forward dynamics accurately but **fail to recover unique
  physical parameters** when the inverse problem is ill-posed. **PINN inverse success depends
  on identifiability — whether the data can single out the parameters at all — and the loss
  does not check that for you.**

---

## 9.6 The optimiser is the bottleneck — where 2024–2026 lives

- **Numerical analysis of PINNs and related models in physics-informed machine learning
  (De Ryck & Mishra, Acta Numerica 2024)** — https://arxiv.org/abs/2402.10926 · Acta Numerica 33:633–713
  — **the rigorous reference.** A unified split of the total error into *approximation* +
  *generalisation* + *optimisation/training* error, with theorems for each across PDE classes,
  dimensions and smoothness levels. **Headline conclusion: the training error is the
  bottleneck — and it is the piece with the weakest theory.** If you want to know what is
  actually *proven* about PINNs, rather than observed, this is the document.
- **Challenges in Training PINNs: A Loss Landscape Perspective (Rathore et al., ICML 2024)** —
  https://arxiv.org/abs/2402.01868 — traces the difficulty to **ill-conditioning caused by the
  differential operator inside the residual**, shows empirically that **Adam → L-BFGS beats
  either alone**, and introduces NysNewton-CG. The paper that made "the problem is second-order
  curvature, not architecture" the consensus view.
- **Unveiling the optimization process of PINNs (Urbán, Stefanou & Pons, JCP 2025)** —
  https://arxiv.org/abs/2405.04230 — swapping the optimiser buys **several orders of magnitude**
  in loss — far more than tinkering with the loss function — and **small networks (2–3 hidden
  layers) suffice once the optimisation is right.** An important correction to the "just make
  it bigger" instinct.
- **Optimizing the Optimizer for PINNs and KANs (Kiyani, Shukla, Urbán, Darbon & Karniadakis, 2025)** —
  https://arxiv.org/abs/2501.16371 — a systematic study of self-scaled quasi-Newton methods
  (SSBroyden and relatives) against Adam and standard L-BFGS. Concrete and directly usable.
- **Gradient Alignment in PINNs: A Second-Order Optimization Perspective (Wang et al., 2025)** —
  https://arxiv.org/abs/2502.00604 — frames the combined loss as multi-task learning with
  **directional conflict** between the loss gradients, proves first-order methods cannot
  resolve it, and gives second-order remedies. Ships in JAX-PI.
- **An Optimisation Framework for the Well-Conditioned Training of PINNs (Webb, Jerad & Cartis, July 2026)** —
  https://arxiv.org/abs/2607.02194 — the current state of the art. **DSGNAR** (Doubly-Sketched
  Gauss-Newton with Adaptive Ratio) attacks the ill-conditioning head-on and reports relative
  ℓ² errors down to **~3 × 10⁻¹¹** across nonlinear, chaotic, multiscale, high-dimensional and
  Navier–Stokes problems. If the trend holds, *"PINNs cannot reach solver-grade precision"* is
  becoming a statement about optimisers, not about PINNs. **Verify on your own problem before
  believing it.**
- **[FP64 is All You Need (NeurIPS 2025)](https://arxiv.org/abs/2505.10949)** — belongs in this
  section too: several "failure modes" turn out to be float32 rounding limits, where L-BFGS
  stops early and reports success. See Chapter 04.

### An active dissent, worth knowing about

- **Visualizing the loss landscapes of physics-informed neural networks (Feb 2026)** —
  https://arxiv.org/abs/2602.05849 — applying loss-landscape visualisation to PINNs, the
  authors find landscapes that look *smooth, well-conditioned and convex near the solution* —
  explicitly *"challenging prevailing intuitions about the complexity of the loss landscapes of
  physics-informed networks."*

**Hold this in tension with Krishnapriyan and Rathore.** The field has not fully settled what
"ill-conditioned" means here — do not repeat the slogan without knowing it is contested.

---

### Run the failure — and one cure — yourself

- The **guided tour on this chapter** replays the measured β-sweep; the marimo and starter labs
  reproduce it locally. A hosted alternative:
  [the pbdl-book's PINN chapter](https://colab.research.google.com/github/tum-pbs/pbdl-book/blob/main/physicalloss-code.ipynb)
  trains a Burgers PINN in your browser — and the book then spends two chapters on why its
  differentiable-physics alternative trains better (§11.3 has the book entry). TensorFlow
  code: read it as a comparison, not a template.
- **FBPINNs example 4 — subdomain scheduling on (1+1)D Burgers** — `notebook` `free` `MIT` —
  [open in Colab](https://colab.research.google.com/github/benmoseley/FBPINNs/blob/main/examples/4.%20Using%20subdomain%20scheduling%20-%20%281%2B1%29D%20Burgers%27%20equation.ipynb)
  (first cell installs from GitHub) — domain decomposition as an engineering cure for exactly
  the multiscale and propagation failures above. JAX; a Colab CPU session handles it.

---

## 9.7 The honest comparison against classical methods

- **Can Physics-Informed Neural Networks beat the Finite Element Method?
  (Grossmann, Komorowska, Latz & Schönlieb, IMA J. Appl. Math. 89(1):143–174, 2024)** —
  https://arxiv.org/abs/2302.04107 · [open](https://pmc.ncbi.nlm.nih.gov/articles/PMC11197852/)

**The answer is no — and you should know exactly how "no".** The findings, which are more
nuanced than the title:

- *"In terms of solution time and accuracy, physics-informed neural networks have not been able
  to outperform the finite element method in our study."* FEM was faster at equal accuracy, or
  more accurate at equal time, across Poisson (1D/2D/3D), Allen–Cahn and Schrödinger.
- On **2D Poisson**, FEM beat PINNs on solution time by **1–3 orders of magnitude** at every
  accuracy level tested.
- On **Allen–Cahn**, counting PINN training time, **FEM was 5–6 orders of magnitude faster.**
  The authors say they were *"particularly surprised"* — they had *expected* PINNs to win here.
  They blame the near-singular interface and the *"possibly ill-conditioned strong form."*
- **The one PINN win:** *evaluating* the trained solution at new points was **2–3 orders of
  magnitude faster** than interpolating an FEM solution onto a new grid. But FEM's solve was so
  much faster that simply re-solving on a new grid would still beat train-then-evaluate.
- **The genuinely positive finding the title hides:** *"PINNs were good at the transition into
  higher dimensions: there is no increment in computational cost from the Poisson equation in
  2D and 3D."* **Flat cost in dimension is the real structural advantage** — and it points to
  high-dimensional PDEs, where FEM simply does not exist.
- Their own closing caveat: an adaptive or variational PINN might have done better on
  Allen–Cahn, and operator learning changes the economics entirely for *parametric* PDEs.
  Carry that caveat with you.

- **Experience report of PINNs in fluid simulations: pitfalls and frustration (Chuang & Barba, SciPy 2022)** —
  https://arxiv.org/abs/2205.14249 — **Taylor–Green vortex (Re=100): about 32 hours of PINN
  training to match a finite-difference simulation that ran in under 20 seconds** — a factor of
  roughly **6,000×** in the wrong direction. **Cylinder flow (Re=200): the PINN failed
  outright**, behaving "like a steady-flow solver" and missing the vortex shedding entirely.
  Follow-up: [Predictive Limitations of PINNs in Vortex Shedding (2023)](https://arxiv.org/abs/2306.00230).
  **The most useful negative result in the PINN literature**, precisely because it is an
  honest, reproducible engineering report. The Barba group's willingness to publish "we tried
  hard and it did not work" is worth more than a hundred papers reporting 1e-4 on 1D Burgers.
- **Examining the robustness of PINNs to noise for Inverse Problems (2025)** —
  https://arxiv.org/abs/2509.20191 — tests the *strongest* PINN claim, and finds PINNs **still
  beaten by traditional FEM-based approaches** — though PINNs require less specialist
  knowledge, and the gap narrows in higher dimensions and with more data. The authors name
  **training failures** as the barrier. Uncomfortable, recent, necessary: **even the inverse
  case must be argued, not assumed.**
- **PINNacle: A Comprehensive Benchmark of PINNs for Solving PDEs (NeurIPS 2024 D&B)** —
  https://arxiv.org/abs/2306.08827 · https://github.com/i207M/PINNacle — 20+ PDEs, 10+ PINN
  variants. Its main service: showing that **no variant dominates** — the ranking depends on
  the problem, and many published improvements do not transfer. **Skim it before you believe
  any paper's accuracy claim.**

---

## 9.8 The one-paragraph honest summary

> A PINN solves an optimisation problem whose minimiser is the PDE solution — using a method
> with no convergence guarantee, on a landscape that is at best awkward and at worst
> pathological, with hyperparameters (the λ's) that have no principled setting and that decide
> whether you get the answer or a convincing-looking artifact. When PINNs work, they work
> beautifully and with very little code. **When they fail, they fail silently** — the loss goes
> down, the plot looks smooth, and the answer is wrong. No residual-based error estimator
> reliably catches this, because a small residual is compatible with satisfying the wrong
> initial condition. **For any forward problem a classical solver can handle, use the classical
> solver.** Reach for a PINN when the problem is inverse, the data is sparse and patchy, the
> geometry is awkward, the dimension is high, or the physics is only partly known — that is,
> when the classical alternative does not exist.

→ **Chapter 10** turns everything the field learned from these failures into a checklist.
