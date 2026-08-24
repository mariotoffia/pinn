---
title: Neural Operators
subtitle: What comes after vanilla PINNs — learning the solution operator, not one solution
minutes: 23
---

# 13 — Neural Operators

## 13.1 The conceptual shift

**In one sentence.** A PINN trains a network `u_θ(x,t)` to *be* the solution of *one* PDE
instance — one set of coefficients, one forcing, one boundary condition — and must be retrained
from scratch for every new instance. An operator learner instead trains a network
`𝒢_θ: a ↦ u` that maps an *input function* (an initial condition, a coefficient field, a
geometry, boundary data) to the *solution function*. **Train once; evaluate for any new input
in milliseconds.**

**Why an engineer should care.** The economics flip. A PINN is a solver with a terrible
constant factor — you pay a full optimisation run per query. An operator is an *amortised*
surrogate: pay one large offline training cost, then get inference that is 100–10,000× faster
than the solver. That is the right shape for design-space exploration, optimisation loops,
Monte-Carlo uncertainty quantification, model-predictive control and real-time digital twins —
**anything where you solve the *same* PDE family thousands of times.**

**What it costs, stated bluntly.** Operator learning is *supervised* learning. You need
input→solution pairs — which means you need a working classical solver to generate them.
**You cannot escape the solver; you amortise it.** Three consequences newcomers routinely miss:

- If you can only afford 50 classical runs, you do not have an operator-learning problem. You
  have a reduced-order-modelling problem, and **classical ROM (POD/PCA + regression) will
  probably beat a neural operator.**
- **Accuracy is bounded by your training distribution.** Generalising to genuinely new physics
  regimes is *the* unsolved problem — §13.4 has the 2026 numbers.
- There is a theoretical wall: for operators with only generic smoothness structure, neural
  operators suffer an infinite-dimensional version of the curse of dimensionality.
  **Structure must be exploited; generic architectures cannot be efficient in general.**

**Core references**

- **Neural Operator: Learning Maps Between Function Spaces (Kovachki, Li, Liu, Azizzadenesheli,
  Bhattacharya, Stuart, Anandkumar, JMLR 2023)** — https://arxiv.org/abs/2108.08481 ·
  [PDF](https://jmlr.org/papers/volume24/21-1524/21-1524.pdf) — the unifying mathematical
  framework: what a neural operator is, discretisation invariance, universal approximation.
  **The single best theory reference.** Chapters 1–4 carry the real conceptual payload.
- **The Parametric Complexity of Operator Learning (Lanthaler & Stuart, 2023, rev. 2025)** —
  https://arxiv.org/abs/2306.15924 — proves a **"curse of parametric complexity" that afflicts
  both DeepONet and FNO**: for operators with only smoothness/Lipschitz structure, the number
  of parameters needed blows up exponentially with the accuracy. **The paper that punctures
  the "neural operators beat the curse of dimensionality" marketing.** It also shows the way
  out — exploit problem structure — which is where the serious research now goes.
- **Neural operators for accelerating scientific simulations and design (Nature Reviews Physics 6, 2024)**
  — https://www.nature.com/articles/s42254-024-00712-5 — the field's flagship review.
  **Read it, but discount it:** it is written by the FNO authors, claims "four to five orders
  of magnitude" speedups, and neither quantifies the accuracy trade-offs systematically nor
  mentions the Lanthaler–Stuart bound. Pair it with §13.4.

---

## 13.2 DeepONet

**The theorem it rests on.** **Chen & Chen (1995)**, IEEE TNN 6(4):911–917 —
https://ieeexplore.ieee.org/document/392253 — proved that a network of the form
`Σₖ bₖ(a(x₁),…,a(xₘ))·tₖ(y)` can approximate any continuous nonlinear operator arbitrarily
well. That factorised form *is* the DeepONet architecture. Lu et al. re-derived it as a
deep-learning architecture 26 years later. **DeepONet is an architecture built on 1995
mathematics — not a new theory.**

**Branch and trunk, properly.**

- The **branch net** takes the input function `a`, *sampled at a fixed set of m "sensor"
  points*, and outputs `p` coefficients.
- The **trunk net** takes a *single query coordinate* `y` and outputs `p` basis values.
- The prediction is their **inner product**: `𝒢_θ(a)(y) = Σₖ bₖ·tₖ(y) + b₀`.

**The right mental model: the trunk learns a basis (think of it as a learned
spectral/finite-element basis), and the branch learns the coefficients for expanding the
solution in that basis.** That is why POD-DeepONet works so well — you can *replace* the trunk
with POD modes computed from the training data, and learn only the branch. **If POD-DeepONet
ties vanilla DeepONet on your problem, your problem is low-rank, and you may not need deep
learning at all.** A useful diagnostic.

**Key limitation:** the branch needs the input function sampled at a **fixed sensor layout.**
DeepONet is discretisation-invariant in its *output*, but not its *input*. FNO is the opposite
trade.

- **Learning nonlinear operators via DeepONet (Lu, Jin, Pang, Zhang, Karniadakis, Nature Machine
  Intelligence 3:218–229, 2021)** — https://www.nature.com/articles/s42256-021-00302-5 ·
  [preprint](https://arxiv.org/abs/1910.03193) — the founding paper. **Conceptually excellent,
  empirically dated** — its benchmarks are toy problems; do not treat its numbers as evidence
  about 2026 performance.
- **Where to actually run it: DeepXDE** — implements `DeepONet`, `DeepONetCartesianProd`,
  physics-informed DeepONet, **POD-DeepONet**, and **MIONet** (multiple input functions). Runs
  fine on a laptop CPU for the tutorial problems.
- **`lululxvi/deeponet`** — 821 ★, 16 commits, one release in Dec 2020, **CC BY-NC-SA 4.0
  (non-commercial!)**, pins DeepXDE v0.11.2. **Effectively abandoned reproduction code. Do not
  build on it.** The non-commercial licence alone rules it out for industry.
- **Physics-informed DeepONets (Wang, Wang, Perdikaris, Science Advances 7(40), 2021)** —
  https://www.science.org/doi/10.1126/sciadv.abi8605 ·
  [open](https://pmc.ncbi.nlm.nih.gov/articles/PMC8480920/) · [preprint](https://arxiv.org/abs/2103.10974)
  — adds the PDE residual as a loss on the DeepONet's output, enabling training with **little
  or no paired solution data.** **The most important DeepONet follow-up, and the natural bridge
  between PINNs and operators.** *Caveat: it inherits every PINN pathology — stiff loss
  balancing, spectral bias, failure on convection-dominated and multiscale problems — on top
  of the difficulty of operator learning. It is harder to train than supervised DeepONet, not
  easier.*

**Build one, from scratch, in one notebook:**
[DeepONet in plain PyTorch](https://colab.research.google.com/github/JohnCSu/DeepONet_Pytorch_Demo/blob/main/DeepONet.ipynb)
(MIT) trains both a data-driven and a physics-informed DeepONet on the antiderivative
operator — 28 self-contained cells, no external data, opens straight in Colab. For the library
version, TorchPhysics' physics-informed DeepONet tutorial is linked in Chapter 11.

---

## 13.3 Fourier Neural Operator

**The idea.** A neural-operator layer looks like
`u_{t+1}(x) = σ(W·u_t(x) + ∫K(x,y)·u_t(y)dy)`. The integral is the expensive part. FNO makes
the kernel translation-invariant, so the integral becomes a **convolution** — and by the
convolution theorem, a convolution is just a **pointwise multiply in Fourier space**:
FFT → multiply the lowest `k` frequency modes by learned weights → inverse FFT. Cost
`O(N log N)`.

**The property that carries the weight: the learned weights live in frequency space, not on the
grid — so a model trained on a 64² grid can be evaluated on a 256² grid.** That zero-shot
super-resolution is the genuine, distinctive FNO selling point.

**The costs, rarely stated up front:** the FFT assumes a **uniform grid on a (nearly) periodic
rectangular domain.** Real engineering geometry is none of those things — every FNO variant
since (Geo-FNO, GINO, U-FNO, Factorized-FNO) exists to patch that hole. Keeping only the low
frequency modes is also an explicit **low-pass filter**, so FNO is structurally weak at shocks,
sharp fronts and fine-scale turbulence.

- **Video first, if you like explainers:**
  [Fourier Neural Operator for Parametric PDEs — Paper Explained (Yannic Kilcher)](https://www.youtube.com/watch?v=IaS72aHrJKE)
  — ~1 h, free. A clear walkthrough of the paper with diagrams and code. A painless way in
  before reading the original.
- **Neural Operator: Graph Kernel Network for PDEs (Li et al., Mar 2020)** —
  https://arxiv.org/abs/2003.03485 — the ancestor: learn the kernel integral with a graph
  network, mesh-agnostic. **Historically essential, practically superseded** (too slow). Read
  the introduction, skip the experiments.
- **Fourier Neural Operator for Parametric PDEs (Li et al., ICLR 2021)** —
  https://arxiv.org/abs/2010.08895 · [OpenReview](https://openreview.net/forum?id=c8P9NQVtmnO) —
  the paper that made operator learning mainstream. **Read it — then treat its Navier–Stokes
  results as a floor for modern baselines, not a target**: the comparisons are against
  2020-era baselines, at modest Reynolds number, on a periodic 2D domain.
- **Physics-Informed Neural Operator (PINO) (Li et al., 2021)** — https://arxiv.org/abs/2111.03794 ·
  [ACM/IMS JDS](https://dl.acm.org/doi/10.1145/3648506) — combines coarse-resolution data with
  a PDE residual computed *in Fourier space*, plus per-instance fine-tuning at test time.
  **The most intellectually satisfying synthesis in the field** — operator pretraining for
  speed, a physics loss for accuracy and resolution transfer. Under-used relative to its
  merit, mostly because it is fiddly to tune.
- **[neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)** —
  3.8k ★, **v2.0.0 (Oct 2025), MIT, part of the PyTorch Ecosystem, actively maintained.** FNO,
  **TFNO** (tensor-factorised, ~10% of the dense parameter count), UNO, GINO and more.
  [Docs](https://neuraloperator.github.io/dev/). **CPU: yes** — plain PyTorch, and it ships
  `load_darcy_flow_small()` (1000 train / 100 test) which trains on a laptop in minutes.
  **The default library. Start here.** The official first tutorial —
  [Training an FNO on Darcy flow](https://neuraloperator.github.io/dev/auto_examples/models/plot_FNO_darcy.html)
  — runs in ~17 s on a CPU with the bundled dataset and demonstrates zero-shot
  super-resolution; in Colab it is `pip install neuraloperator` plus that page's
  "Download Jupyter notebook" link. Prefer a stepped lesson instead? ETH's
  [Tutorial 05 — Fourier Neural Operator](https://colab.research.google.com/github/camlab-ethz/AI_Science_Engineering/blob/main/Tutorial%2005%20-%20Operator%20Learing%20-%20Fourier%20Neural%20Operator.ipynb)
  opens in Colab (the "Learing" typo is in the real filename).

### The weather lineage — and where it stands in 2026

The most-cited "physics-ML works!" success story, so it deserves precision.

- **FourCastNet (NVIDIA, 2022)** — https://arxiv.org/abs/2202.11214 — AFNO trained on the ERA5
  reanalysis. Historically important proof that a neural operator could do global weather.
  **Superseded.**
- **GraphCast (Lam et al., DeepMind, Science 382:1416, 2023)** —
  https://www.science.org/doi/10.1126/science.adi2336 · [repo](https://github.com/google-deepmind/graphcast)
  — a graph network on an icosahedral mesh; beat ECMWF's HRES on most deterministic targets.
  **The landmark result. Real, replicated, and it changed the field.**
- **GenCast (Price et al., DeepMind, Nature 637:84, 2024)** —
  https://www.nature.com/articles/s41586-024-08252-9 — diffusion-based *ensemble* forecasting;
  beat the ECMWF ensemble on **97.2% of 1,320 targets.** **More important than GraphCast**,
  because operational meteorology is probabilistic, and that is where the deterministic AI
  models were weakest. Successor line: [WeatherNext](https://github.com/google-deepmind/weathernext).
- **Aurora (Bodnar et al., Nature, May 2025)** — [repo](https://github.com/microsoft/aurora) —
  917 ★, MIT, **open weights** (small checkpoint ~500 MB) — one pretrained atmospheric
  backbone, fine-tuned to weather, **air pollution** and **ocean waves.** **The strongest real
  evidence that "a foundation model for physics" is more than a slogan** — but note that it
  succeeded in the one domain that already had a decade of petabyte-scale, homogeneous
  reanalysis data. **Not a template most engineering domains can copy.**
- **ECMWF AIFS — and the 2026 cull** — https://www.ecmwf.int/en/about/media-centre/aifs-blog/2026/farewell-external-ai-models
  — **the most important weather link on this page.** ECMWF's own AI model, AIFS, went
  **operational in February 2025.** Then, with the **IFS Cycle 50r1 upgrade in May 2026, ECMWF
  stopped running all four external AI models — Pangu-Weather, GraphCast, Aurora and
  FourCastNet.** The reasons given: performance degraded after the IFS upgrade (models with
  fine-tuning steps were hit hardest), Aurora and Pangu lack precipitation, and
  GraphCast/FourCastNet were superseded by probabilistic versions.

> **The lesson every engineer should take away:** the AI weather models *worked* — and they
> *still* rotted when the system that produced their training data changed, because they were
> trained on the outputs of one specific solver/analysis version. **A learned surrogate is
> coupled to the solver that produced its training data.** This is the single most
> transferable, least-discussed finding in physics-ML — and it applies to your CFD surrogate
> too.

---

## 13.4 Transformers, latent surrogates, and "PDE foundation models"

**Why attention.** The kernel integral `∫K(x,y)u(y)dy` *is* attention with a learned kernel.
That equivalence is the entire justification — and it is a good one: attention handles
**unstructured meshes and arbitrary geometry**, exactly where FNO's uniform-grid FFT fails.
The price is `O(N²)` in the number of mesh points, so every paper in this area is
fundamentally about approximating linear-cost attention.

- **Choose a Transformer: Fourier or Galerkin (Cao, NeurIPS 2021)** — https://arxiv.org/abs/2105.14995
  — showed the softmax is unnecessary; linear "Galerkin-type" attention is enough. **The
  conceptual origin of attention-for-PDEs.**
- **OFormer (2022)** — https://arxiv.org/abs/2205.13671 · [repo](https://github.com/BaratiLab/OFormer)
  — a mesh-free encoder–decoder transformer with linear attention. Solid, largely superseded.
- **Transolver (Wu et al., ICML 2024 Spotlight)** — https://arxiv.org/abs/2402.02366 ·
  [repo](https://github.com/thuml/Transolver) — "Physics-Attention": softly assign mesh points
  to `M ≪ N` learned *slices*, attend among the slices, scatter back. Linear cost, handles
  industrial geometry. [Transolver++](https://arxiv.org/abs/2502.02414) scales to
  million-point meshes. **The strongest general-geometry PDE surrogate as of 2026.**
- **Transolver is a Linear Transformer (2025/2026)** — https://arxiv.org/abs/2511.06294 —
  **read this immediately after Transolver.** It proves Physics-Attention is a special case of
  ordinary linear attention (not a physics-specific mechanism), and shows empirically that
  **removing the slice-attention step — the component the physics story rests on — usually
  *improves* accuracy.** Their simplified **LinearNO** matches or beats Transolver with
  **40% fewer parameters and 36.2% less compute.** **A perfect case study in how physics-ML
  papers narrate mechanisms they never isolated.** The architecture works; the stated reason
  it works was wrong.
- **Latent / diffusion PDE surrogates** — e.g.
  [Generative Latent Neural PDE Solver using Flow Matching](https://arxiv.org/abs/2503.22600).
  **The right tool for one specific job — probabilistic or multi-modal problems**
  (turbulence statistics, ensembles, data assimilation, inverse problems with several valid
  answers), where a deterministic model collapses to a blurry average. GenCast is the proof
  this matters. **Not** the right tool for a deterministic forward solve.

### Foundation models for PDEs — real, benchmarked, or marketing?

- **Multiple Physics Pretraining (Polymathic AI, 2023)** — https://arxiv.org/abs/2310.02994 —
  the first serious "train one transformer on many kinds of physics, fine-tune downstream."
  **Real and carefully reported**, and the intellectual seed of the sub-field.
- **Poseidon (Herde et al., ETH, NeurIPS 2024)** — https://arxiv.org/abs/2405.19101 ·
  [site](https://camlab-ethz.github.io/poseidon/) — a multiscale operator transformer,
  pretrained on fluid dynamics, evaluated on **15 downstream tasks.** **The most rigorous
  foundation-model-for-PDEs paper**; the credible headline is the sample-efficiency gain on
  downstream tasks. **Weights and data are CC BY-NC-SA 4.0 — non-commercial.**
- **DPOT (2024)** — https://arxiv.org/abs/2403.03542 — denoising auto-regressive pretraining,
  ~1B parameters. Real, benchmarked, and now the standard baseline everyone beats.
- **GPhyT / Towards a Physics Foundation Model (2025–26)** — https://arxiv.org/abs/2509.13805 ·
  [repo](https://github.com/FloWsnr/General-Physics-Transformer) — 1.8 TB / 2.4M snapshots,
  claims **7× lower error than DPOT.** **Strong work with unusually frank limitations, stated
  by the authors themselves:** 2D only, fixed 256×128 resolution, fluids/heat only, falls
  *"considerably short"* of numerical-solver precision, and *"no model is capable of
  high-fidelity predictions."* Believe the relative ranking; disbelieve any suggestion that
  this replaces a solver.
- **Do Physics Foundation Models Learn Generalizable Physics? A Bias-Aware Benchmark (2026)** —
  https://arxiv.org/html/2605.29283 · [data](https://huggingface.co/datasets/90879c/PhysBiasBench) —
  **the essential skeptical counterweight.** It benchmarks DPOT, GPhyT, MORPH, MPP and
  Poseidon under controlled distribution shift. Findings: a shift in dynamic scale amplifies
  error **~4.75×**; a joint out-of-distribution shift **7–8×**; **37.5% of
  architecture–PDE pairs show *negative* transfer from pretraining**; **25% of the larger
  models are *worse* than their smaller versions**; and training on more complex data made
  the robustness damage *worse*. Conclusion: these models are **"conditional rather than
  universal generalists," and scaling is not fixing it.**
- **NVIDIA Apollo (announced SC25, Nov 2025)** — https://blogs.nvidia.com/blog/apollo-open-models/ —
  an open model family for CFD, structural mechanics, electromagnetics, semiconductors, and
  weather. **The "up to 500×" and "35×" numbers are vendor-supplied, single-customer,
  unrefereed, and carry no error bars — treat them as marketing until a third party reproduces
  them.**

---

## 13.5 Kolmogorov–Arnold Networks — the 2026 verdict

**The idea.** An MLP puts fixed nonlinearities on *nodes* and learns linear weights on
*edges*. A KAN flips this: it puts **learnable one-dimensional spline functions on the edges**
and just sums at the nodes. The pitch for physics was interpretability: read off the learned
1D functions and fit formulas to them — so a KAN might *discover* the law instead of just
fitting it.

- **KAN: Kolmogorov–Arnold Networks (Liu et al., 2024; ICLR 2025)** — https://arxiv.org/abs/2404.19756
  — **genuinely creative — and the most over-hyped ML paper of 2024.** The physics and maths
  examples are small and hand-picked, and the "beats MLPs" claim did not survive fair
  benchmarking.
- **[KindXiaoming/pykan](https://github.com/kindxiaoming/pykan)** — 16.3k ★, MIT, **last
  release v0.2.8 (Nov 2024)**, 236 open issues. **The star count measures excitement, not
  health.** The author's own README is more candid than most of the follow-up literature:
  KANs are *"super slow,"* training them for PDEs *"may take hours to days on a single CPU,"*
  and *"KANs are likely not a simple plug-in that can be used out-of-the-box (yet)."*
- **A Critical Assessment of Claims, Performance, and Practical Viability (2024)** —
  https://arxiv.org/abs/2407.11075 — the first thorough reality check; it has aged well.
- **A Practitioner's Guide to Kolmogorov–Arnold Networks (Oct 2025)** — https://arxiv.org/html/2510.25781v1
  — **the best single verdict document as of 2026.** No universal winner; *"studies that
  emphasize fairness often reach divergent or contradictory conclusions"*; performance is
  dominated by the **choice of basis function**, and the best basis **changes from PDE to
  PDE**; the RBF width is numerically fragile; KANs degrade on non-smooth targets; and
  training is **5–20× slower per iteration** than an MLP.
- **SPIKANs: separable physics-informed KANs** — https://arxiv.org/abs/2411.06286 —
  representative of where the serious work went: **making KANs cheap enough to compete, rather
  than claiming they are better.**
- **From PINNs to PIKANs (Toscano et al., Oct 2024)** — https://arxiv.org/abs/2410.13228 — the
  CRUNCH group's current review: architectures, adaptive refinement, decomposition, adaptive
  weights and activations, optimisers, UQ and theory, plus PIKANs.

> **The 2026 verdict: KANs did not hold up as a general replacement for MLPs.** They are a
> legitimate, interesting *design space* — parameter-efficient, sometimes faster-converging
> per epoch, and meaningfully better at **recovering interpretable, low-dimensional
> closed-form relationships**, which is a real niche. They are **not** a drop-in accuracy
> upgrade for PINNs, they are slower on the wall clock, and their reported wins are heavily
> confounded by per-problem basis tuning. **If you have three days, spend them on operator
> learning, not KANs.** And if your actual goal is "recover an interpretable formula," what
> you probably want is PySR (Chapter 14 §14.5).

---

## 13.6 Benchmarks and datasets

**These datasets are enormous, and nobody puts the download size in the abstract.**

- **[PDEBench](https://github.com/pdebench/PDEBench)** ([paper](https://arxiv.org/abs/2210.07182) ·
  [data](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986))
  1D advection, Burgers, diffusion-reaction, diffusion-sorption; 2D diffusion-reaction, Darcy
  flow, incompressible and compressible Navier–Stokes; shallow water; 3D compressible
  Navier–Stokes. **336 files at ~7.7 GB *each* — several TB in total.** The archive refuses
  whole-collection downloads.
  **Licence: code MIT, data CC BY 4.0** — the most permissive of the big three.
  **Small start:** `download_direct.py` filtered to one PDE and one parameter — **budget
  ~8 GB for one file.** The 1D sets are the only laptop-friendly entry point.
  **The default benchmark, whose numbers reviewers recognise** — and also the one most
  affected by the weak-baselines critique in Chapter 14.
- **[PDEArena](https://github.com/pdearena/pdearena)** ([docs](https://pdearena.github.io/pdearena/)) —
  Navier–Stokes, shallow water, Maxwell. MIT. **A better-engineered training and evaluation
  harness than PDEBench** (a proper Lightning pipeline, strong U-Net and Clifford-layer
  baselines), smaller community. Use it if you care about *fair architecture comparison*;
  use PDEBench if you care about *comparability with published numbers*.
- **[The Well (Polymathic AI, NeurIPS 2024)](https://github.com/PolymathicAI/the_well)**
  ([paper](https://arxiv.org/abs/2412.00568) · [site](https://polymathic-ai.org/the_well/)) —
  **16 datasets, 15 TB total; individual datasets from 6.9 GB to 5.1 TB.** Biological/active
  matter, acoustic scattering, turbulent radiative layers, MHD, supernova explosions,
  extragalactic fluids, viscoelastic instability. **Licence: BSD-3-Clause — the most
  commercially usable.**
  **Stream instead of downloading** via `well_base_path="hf://datasets/polymathic-ai/"` —
  **the right answer on a laptop.** `active_matter` is the standard small starter.
  **By far the most physically *diverse* collection** — and the one that will most brutally
  expose a surrogate that has only ever seen Darcy flow.
- **[APEBench (TUM, NeurIPS 2024)](https://github.com/tum-pbs/apebench)**
  ([paper](https://arxiv.org/abs/2411.00180)) — 46+ PDEs in 1D/2D/3D, MIT, JAX.
  **The key differentiator: there is no dataset to download.** The data is generated **on the
  fly** by a fast spectral solver ([Exponax](https://github.com/Ceyron/exponax)) — and the
  solver is **differentiable**, so you get solver-in-the-loop training natively, plus proper
  **rollout metrics** instead of single-step error.
  **The best benchmark for a laptop, and the most methodologically modern of the four.
  Underrated — if you have limited disk and limited patience, start here.**
- **Leaderboards:** there is **no** widely accepted live leaderboard for operator learning —
  and that absence is itself a finding: it is a major reason the weak-baseline problem
  persists. The nearest equivalents are
  [Matbench Discovery](https://matbench-discovery.materialsproject.org/) for interatomic
  potentials and [WeatherBench 2](https://sites.research.google/weatherbench/) for weather —
  both in *neighbouring* fields, which tells you where evaluation culture is actually mature.

### Self-generated benchmarks — do this first

The most valuable benchmark for an engineer is the one you write yourself, because you control
the baseline and you know the ground truth.

1. **1D heat / Burgers via finite differences or `scipy.integrate.solve_ivp`** — ~30 lines.
   Generate 1,000 initial conditions from a Gaussian random field, solve them, train an FNO
   from `neuraloperator`. **The complete operator-learning loop, in one afternoon.**
2. **2D Darcy flow** — the canonical parametric problem (permeability field → pressure field).
   `neuraloperator`'s `load_darcy_flow_small()` lets you skip data generation entirely on day
   one.
3. **Lid-driven cavity or flow past a cylinder** in FEniCSx / py-pde — the first problem where
   surrogates start visibly failing. That is the educational point.
4. **Always log the classical solver's wall-clock time.** If you cannot say "the FD solver took
   X ms per instance, and the surrogate took Y ms after Z hours of training on N solver runs,"
   you cannot claim a speedup.

---

## 13.7 If you learn one thing beyond PINNs

**Learn the Fourier Neural Operator, hands-on, with `neuraloperator` v2.0.0 — and learn it by
benchmarking it yourself against a classical solver you wrote.**

Concretely: `pip install neuraloperator`, run the built-in `load_darcy_flow_small()` example on
your laptop CPU, then do the one experiment that teaches the whole field — **train at 64×64 and
evaluate at 256×256.** It works, with no retraining. **That single result is the clearest
physical intuition available for what "learning a map between function spaces" actually means —
and it is a property no PINN, no classical ROM, and no ordinary CNN has.**

**Why FNO, and not something newer:**

- **It teaches the core concept in the cleanest form.** The whole architecture is FFT →
  multiply the low modes → inverse FFT. You can understand it *completely* — which is not true
  of Transolver or any foundation model. And once you see *why* the FFT restricts you to
  uniform periodic grids, you immediately understand why every later architecture exists —
  the field's whole design tree from one insight.
- **The tooling is the best in physics-ML**, and actively maintained: MIT, v2.0.0 (Oct 2025),
  PyTorch Ecosystem, runs on CPU, ships a small dataset. Compare `lululxvi/deeponet`:
  16 commits, dead since 2020, non-commercial licence.
- **It forces you into the right workflow.** To train it, you must generate data with a
  classical solver — which teaches the economics of §13.1 better than any paper, and means
  you automatically have the baseline that 79% of published papers failed to build.
- **It is the right learning investment even though it is not the 2026 accuracy leader.**
  Transolver-class models beat FNO on real geometry, and you should learn one eventually. But
  **FNO is the concept made legible**; Transolver is an optimisation of that concept — whose
  own authors' explanation of why it works turned out to be wrong. Learn the idea first.
