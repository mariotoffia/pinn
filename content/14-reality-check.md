---
title: The 2026 Reality Check
subtitle: When to use what, what is actually deployed, and what to disbelieve
minutes: 19
---

# 14 — The 2026 Reality Check

Prefer being accurate and skeptical over being enthusiastic. Everything here is sourced.

---

## 14.1 The three papers that define the honest position

- **Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related PDEs
  (McGreivy & Hakim, Nature Machine Intelligence 6:1256–1269, 2024)** —
  https://www.nature.com/articles/s42256-024-00897-5 · [preprint](https://arxiv.org/abs/2407.07218)

  **The finding: of the articles claiming to outperform standard numerical methods on
  fluid-related PDEs, 79% (60 of 76) compared against a *weak baseline*.** On top of that:
  selective reporting and publication bias systematically bury negative results. Their
  conclusion: *"ML-for-PDE solving research is overoptimistic."*

  **The single most important paper in this whole document.** Not because ML for PDEs does not
  work — but because **most published claims that it works are not evidence that it works.**
  The practical test the paper gives you: *was the baseline solver run at the accuracy the ML
  model actually reaches, on comparable hardware, with a competent implementation?* Usually
  not. The standard failure: comparing a surrogate against a first-order solver on a fine
  grid, when a second-order solver on a coarse grid would match the surrogate's accuracy in
  less time than the surrogate's inference.

- **Can Physics-Informed Neural Networks beat the Finite Element Method? (Grossmann, Komorowska,
  Latz, Schönlieb, IMA J. Appl. Math. 89(1):143–174, 2024)** — https://arxiv.org/abs/2302.04107 ·
  [open](https://pmc.ncbi.nlm.nih.gov/articles/PMC11197852/)

  *"In terms of solution time and accuracy, physics-informed neural networks have not been able
  to outperform the finite element method in our study."* The only PINN advantage found:
  **faster evaluation of the already-solved PDE at new points.**

  **The definitive controlled head-to-head. The answer is no.** But note carefully what it does
  *not* say: they tested low-dimensional, well-posed, forward problems on friendly domains —
  **exactly the territory FEM was built for and has had 60 years to optimise.** The fair
  reading: PINNs lose decisively on FEM's home ground. The study says nothing about inverse
  problems or high dimensions. Full numbers in Chapter 09 §9.7.

- **Experience report of PINNs in fluid simulations: pitfalls and frustration (Chuang & Barba, SciPy 2022)** —
  https://arxiv.org/abs/2205.14249

  **Taylor–Green vortex (Re=100): about 32 hours of PINN training to match a finite-difference
  simulation that ran in under 20 seconds** — roughly **6,000×** in the wrong direction.
  **Cylinder flow (Re=200): the PINN failed outright**, behaving "like a steady-flow solver"
  and missing the vortex shedding entirely. Follow-up:
  [Predictive Limitations of PINNs in Vortex Shedding (2023)](https://arxiv.org/abs/2306.00230).

  **The most useful negative result in the PINN literature** — precisely because it is an
  honest, reproducible engineering report rather than a theory paper.

### And for operator learning specifically

- **A comprehensive comparison of neural operators for 3D industry-scale engineering designs
  (Zhong, Liu, Abueidda, Koric, Meidani, Oct 2025)** — https://arxiv.org/abs/2510.05995

  Compares branch-trunk models (the DeepONet family), graph-based, grid-based (the FNO family)
  and point-based models (Transolver/PointNet family) across six industrial 3D datasets:
  thermal, linear elasticity, elasto-plasticity, time-dependent plasticity, CFD.

  **Relative error:** heat sink **0.10%** · bracket **0.32–1.75%** · time-dependent bracket
  **5.6%** · **DrivAer automotive CFD (parametric): 23.1%** · **JEB freeform: 29.8%** ·
  **DrivAer++ freeform: 17.3%** (Transolver, the best). Inference **"at least 100×"** faster
  on the easiest case, **"at least 1000×"** on the others. Model sizes from ~100K to **162M**
  parameters. Their conclusion: *"no single neural operator architecture consistently
  outperforms others across all tasks."*

  **The most useful practical benchmark in this document — and quietly devastating. Read the
  gradient: simple parametric geometry with smooth physics → below 1% error, genuinely
  deployable. Freeform geometry with real CFD → 17–30% error, which is not an engineering
  answer.** Anyone selling you a CFD surrogate should be asked which end of that spectrum
  their problem sits on.

---

## 14.2 When is a PINN the right tool?

**Use a PINN when:**

- **Inverse problems / parameter identification.** Sparse noisy measurements, unknown
  coefficients, sources or boundary conditions. The PINN loss handles "fit the data AND
  satisfy the physics AND infer λ" in one optimisation, with essentially no extra machinery.
  **Doing the same with FEM means building an adjoint solver.** This is the strongest genuine
  PINN use case, and it is strong.
- **Data assimilation with a known governing equation** — filling in the field between sparse
  sensors (say, reconstructing a velocity field from limited PIV or 4D-flow MRI). The physics
  acts as an exceptionally good regulariser wherever the data is silent.
- **High-dimensional PDEs (d ≳ 4)** — Hamilton–Jacobi–Bellman, Fokker–Planck, Black–Scholes
  baskets, many-particle problems. Mesh-based methods die exponentially with dimension;
  Monte-Carlo-flavoured neural methods do not. **Here PINNs win on a genuine structural
  argument, not a constant factor.**
- **Ill-posed / mixed problems** with incomplete boundary conditions, where a classical solver
  cannot even be set up.
- **Free-boundary and PDE-constrained optimisation** problems, where a mesh would otherwise
  have to move.

**Do NOT use a PINN when:**

- **You have a well-posed forward problem on a meshable domain in 1–3D.** Use FEM/FV/spectral.
  This is Grossmann et al.'s finding, and it is not close.
- **You need a fast parametric surrogate.** A PINN retrains per instance. Use an operator or a
  classical reduced-order model.
- **The solution has shocks, sharp fronts, discontinuities, or multiscale/turbulent
  structure.** Spectral bias means the network learns the smooth part first and may never get
  the rest.
- **The dynamics are convection-dominated or chaotic over long horizons.** The optimiser
  prefers the smooth, wrong, steady solution to the correct oscillating one.
- **You need certified error bounds, conservation guarantees, or regulatory sign-off.** A
  PINN's error is an empirical observation, not a mathematical estimate.
- **You have plenty of solver data.** If you can generate data, supervised operator learning
  is easier to train and more accurate. **The physics loss is a *substitute* for data, not a
  bonus on top of it.**

---

## 14.3 The comparison table

| | **PINN** | **DeepONet** | **FNO** | **Transformer NO** | **Classical solver** |
|---|---|---|---|---|---|
| **What it learns** | One solution field for one PDE instance | The operator `a ↦ u`, as a learned basis × coefficients | The operator, as learned convolution kernels in Fourier space | The operator, as attention over mesh points | Nothing — discretises and solves directly |
| **Training data** | **None** (residual only); optional sparse data | Solver pairs (10³–10⁴), or none for PI-DeepONet | Solver pairs (10³–10⁴) | Solver pairs, typically more | None |
| **Inference cost** | **Full retrain per instance** (minutes→days) | Forward pass, ~ms | Forward pass, ~ms, `O(N log N)` | ~ms–s; linear variants reach 10⁶ points | Full solve per instance — but *predictable* |
| **Mesh dependence** | **Mesh-free** (collocation) | Output mesh-free; **input needs fixed sensors** | **Uniform, near-rectangular/periodic grid**; zero-shot super-resolution | **Arbitrary unstructured meshes** | The mesh is the method |
| **Realistic accuracy** | 1e-2–1e-4 on easy problems; fails on stiff/convective ones | ~1e-2–1e-3 in-distribution | ~1e-2–1e-3; degrades at shocks | **0.1–2% simple parametric; 17–30% freeform industrial CFD** | **Convergent to machine precision, with error bounds** |
| **Best use** | **Inverse, data assimilation, d≳4, ill-posed** | Parametric surrogates, fixed input sampling, interpretable basis | Uniform-grid parametric PDEs where multi-resolution matters | **Industrial geometry — CFD/FEA over CAD meshes** | **Everything well-posed in 1–3D that needs a trustworthy answer** |
| **Maturity (2026)** | Mature as a research method; **not production for forward solves** | Mature; solid tooling | Mature; the best tooling | **Moving fast — SOTA claims contested** | **Fully mature, certified, regulated** |
| **Honest one-liner** | Loses to FEM at FEM's job; wins where FEM cannot go | The interpretable one; check POD-DeepONet first | The one with super-resolution; hates non-rectangles | Handles real geometry; the physics story is oversold | **Still the correct default** |

---

## 14.4 What is actually working in industry

1. **Weather forecasting — the one unambiguous win. Grade: A.** ECMWF's AIFS went
   [operational in February 2025](https://www.ecmwf.int/en/forecasts/datasets/aifs-machine-learning-data);
   GenCast beats the ECMWF ensemble on 97.2% of targets; Aurora is in *Nature* with open MIT
   weights. **But** ECMWF retired all four external AI models in May 2026, after they degraded
   following the IFS Cycle 50r1 upgrade. **Real, operational — and at the same time the
   clearest demonstration that surrogates carry a maintenance liability tied to the solver
   that trained them.**

2. **Molecular / materials simulation (ML interatomic potentials) — the most commercially
   successful physics-ML, and it is not close. Grade: A.** MACE, NequIP and their successors
   deliver quantum-chemistry accuracy at classical-force-field cost — a real 10³–10⁶× speedup
   on a task people pay for, validated on a live adversarial leaderboard. Used daily in
   battery materials, catalysis, drug discovery and MOF screening. **Note why it worked: the
   target is a smooth, local, symmetry-constrained scalar function — not a chaotic PDE
   rollout. Do not generalise this success to CFD.**

3. **CFD surrogates in engineering design — real but narrow. Grade: C+.** Deployed for
   early-stage design-space exploration and shape optimisation, where 20% error is acceptable
   as a *ranking* signal. Vendors (Ansys, Siemens, Autodesk, NVIDIA Apollo/PhysicsNeMo) are
   shipping this. The §14.1 numbers are the reality: **0.1–2% on simple parametric geometry
   (deployable); 17–30% on freeform industrial CFD (a screening tool, not an answer).** Nobody
   is certifying an aircraft on an FNO.

4. **Digital twins — mostly still marketing. Grade: D+ as currently sold.** The
   real-time-surrogate story is right in principle, and the best *conceptual* fit for operator
   learning. In practice, the
   [SINTEF field report (July 2026)](https://blog.sintef.com/digital-en/what-does-it-take-to-bring-physics-informed-machine-learning-to-industry/)
   is the essential corrective: industrial sensor data is irregular and biased (not neat
   Gaussian noise); **models are trained only inside safe operating ranges, and are therefore
   useless exactly at the thresholds where the decisions are made**; often you cannot measure
   the quantity of interest at all (*"if the inside of your aluminium furnace is hot enough to
   melt aluminium, it is also hot enough to melt any thermometer"*); and post-deployment drift
   invalidates models after process changes. Their punchline: in one deployment **linear
   regression beat the complex algorithms**, and *"boring, maintainable models often survive
   industrial reality better than large general ones."* **The barriers are data and workflow
   integration, not architecture.**

5. **Medical / biomechanics — promising, pre-clinical. Grade: C.** Cardiovascular flow
   reconstruction from sparse 4D-flow MRI is the genuinely compelling case, because it is an
   **inverse/assimilation problem with sparse data** — the one regime where PINNs actually
   have an argument. **No regulatory pathway exists for an uncertified neural surrogate in a
   diagnostic loop** — and that, not accuracy, is the binding constraint.

6. **Seismic inversion, reservoir simulation, plasma/fusion control — Grade: B−.** Active,
   with credible published deployments (DeepMind's tokamak control the standout). **These
   share the profile that predicts success: expensive repeated solves, tolerant accuracy
   requirements, and a hard real-time constraint that classical methods physically cannot
   meet.**

> **The pattern.** Physics-ML has genuinely won where **(a)** enormous, homogeneous training
> data already existed for other reasons (ERA5, the Materials Project), **(b)** the target
> function is smooth and symmetry-constrained rather than chaotic, and **(c)** there is a hard
> real-time or throughput constraint no classical method can meet. Where those do not hold,
> classical solvers are still winning — and the honest answer is that they should be.

---

## 14.5 Variants and neighbouring directions worth knowing

**PINN variants** (one link, one line each):

- **Hard constraints** — [hPINN (Lu et al. 2021)](https://arxiv.org/abs/2102.04626) ·
  [distance functions (Sukumar & Srivastava)](https://arxiv.org/abs/2104.08426) — impose the
  boundary conditions exactly, deleting a loss term and its weight.
- **Deep Ritz** — [Weinan E & Bing Yu, 2017](https://arxiv.org/abs/1710.00211) — minimise the
  **energy functional** instead of the squared strong-form residual. Only first derivatives,
  so lower smoothness demands and cheaper autodiff. The trade: it only applies to problems
  with variational structure, and boundary conditions are harder to impose.
- **Variational PINNs** — [hp-VPINNs (Kharazmi, Zhang, Karniadakis)](https://arxiv.org/abs/2003.05385)
  · [code](https://github.com/ehsankharazmi/hp-VPINNs) — trial space = the network; test
  space = piecewise high-order polynomials. **The most principled bridge between PINNs and
  finite elements.**
- **Domain decomposition** — [cPINN](https://doi.org/10.1016/j.cma.2020.113028) (flux
  continuity across interfaces, so conservation is structural) ·
  [XPINN](https://www.global-sci.com/cicp/article/view/6911) (arbitrary space-**time**
  decomposition, naturally parallel). **Caveat:**
  [When Do XPINNs Improve Generalization?](https://doi.org/10.1137/21M1447039) finds that
  decomposition helps only when the solution is complex enough to justify less data per
  subdomain — a real trade-off, not a free win.
- **Bayesian PINNs** — [B-PINNs (Yang, Meng, Karniadakis)](https://arxiv.org/abs/2003.06097) —
  a Bayesian prior on the weights with the PINN likelihood; posterior via HMC or variational
  inference. **Given that PINNs fail silently, a posterior over solutions is arguably the
  missing safety mechanism, not a luxury.**
- **Efficiency** — [Separable PINNs (NeurIPS 2023)](https://arxiv.org/abs/2306.15969) —
  factorises the network per axis and uses forward-mode AD, turning `Nᵈ` collocation points
  into `d·N` network evaluations. **A rare case where the algorithmic idea, not the
  hyperparameters, does the work.**
- **Multiscale** — [FBPINNs (Moseley)](https://github.com/benmoseley/FBPINNs) — finite-basis
  domain decomposition; the natural next step once vanilla PINNs disappoint you.

**Neighbouring, not core:**

- **Differentiable CFD / learned closures** —
  [Kochkov et al., PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2101784118) — **arguably
  the most credible speedup in the field** (~8–10× via learned corrections *inside* a real
  solver) — credible precisely because the solver still enforces conservation, and ML only
  supplies the closure.
- **Solver-in-the-loop** — [Um et al., NeurIPS 2020](https://arxiv.org/abs/2007.00016) ·
  [ΦFlow](https://github.com/tum-pbs/PhiFlow) — train against the *differentiable solver's*
  rollout rather than one-step targets. The correct way to build hybrid schemes.
- **Neural ODEs** — [Chen et al., NeurIPS 2018](https://arxiv.org/abs/1806.07366) ·
  [torchdiffeq](https://github.com/rtqichen/torchdiffeq) — **the mirror image of a PINN**: a
  PINN replaces the *solver* with a network and keeps the equation; a Neural ODE replaces the
  *equation* with a network and keeps the solver. **Universal Differential Equations** merge
  the two — known physics plus a learned residual term — and are often the right engineering
  answer.
- **Hamiltonian / Lagrangian NNs** — [HNN](https://arxiv.org/abs/1906.01563) ·
  [LNN](https://arxiv.org/abs/2003.04630) — learn the Hamiltonian and derive the dynamics from
  it, so **energy conservation is exact by construction.** The cleanest illustration of
  "build the physics into the architecture, not the loss."
- **Symbolic regression** — [PySR](https://github.com/MilesCranmer/PySR) (**the most practical
  route to an actual interpretable formula**) · [PySINDy / SINDy](https://github.com/dynamicslab/pysindy)
  ([PNAS 2016](https://www.pnas.org/doi/10.1073/pnas.1517384113)) · [AI Feynman](https://github.com/SJ001/AI-Feynman).
  **SINDy is the complement to PINN inverse problems: a PINN fits the coefficients of a
  *known* equation form; SINDy discovers the form itself.** Combining them handles noisy,
  sparse data where SINDy's derivative estimates alone would be hopeless.
- **Equivariant / geometric deep learning** — [e3nn](https://e3nn.org/) — the machinery that
  made ML interatomic potentials work, and the field's best evidence that **architectural
  symmetry beats data augmentation.**
- **ML interatomic potentials** — [MACE](https://github.com/ACEsuit/mace) ·
  [NequIP](https://www.nature.com/articles/s41467-022-29939-5) ·
  [Matbench Discovery](https://matbench-discovery.materialsproject.org/) — **the best-run
  leaderboard in physics-ML.** Note that almost all current leaders are closed or only
  partially open — a real and worsening reproducibility problem.

---

## 14.6 The 2026 reality check, in eleven points

1. **Assume any claimed speedup is wrong until you have seen the baseline.** 79% of papers
   claiming to beat numerical methods on fluid PDEs compared against a weak baseline. That is
   a *measured* rate, not a suspicion — and the correct default prior for every new paper you
   read.
2. **PINNs do not beat FEM at forward solves, and this is settled.** Grossmann et al. tested
   Poisson, Allen–Cahn and Schrödinger; PINNs lost on both time and accuracy. Chuang & Barba
   needed 32 hours to reproduce a 20-second finite-difference run — and then failed outright
   on vortex shedding.
3. **But PINNs are genuinely good at something specific: inverse problems, data assimilation,
   and high-dimensional PDEs.** These are not consolation prizes — they are regimes where
   classical methods need a custom adjoint solver, or die exponentially with dimension.
   **Learn PINNs for these.**
4. **Operator learning does not remove the classical solver — it amortises it.** No solver, no
   operator-learning project.
5. **"Resolution-invariant" and "beats the curse of dimensionality" are overstated.**
   Lanthaler & Stuart proved a curse of parametric complexity that afflicts DeepONet and FNO
   alike. **There is no free lunch in infinite dimensions either.**
6. **Neural-operator accuracy on real industrial geometry is 17–30% relative error, not
   0.1%.** That is a screening and ranking tool, not an engineering answer — and it will not
   be signed off.
7. **"Foundation models for physics" do not yet generalise.** The 2026 bias-aware benchmark
   found 4.75× error amplification under scale shift, 7–8× under joint shift, negative
   transfer in 37.5% of architecture–PDE pairs, and larger models doing worse 25% of the
   time. **More data made robustness worse.**
8. **KANs did not hold up.** No universal advantage over MLPs, 5–20× slower per iteration,
   results dominated by per-problem basis choice. 16.3k GitHub stars measured excitement, not
   evidence.
9. **A learned surrogate is coupled to the solver that generated its training data — and it
   rots when that solver changes.** ECMWF retired four AI weather models in May 2026 because
   an *upstream numerical model upgrade* degraded them. **The field's most important and
   least-discussed operational finding.**
10. **The papers that explain mechanistically why an architecture works are frequently
    wrong.** Transolver's "Physics-Attention" is ordinary linear attention, and deleting the
    component the physics story rests on usually improves accuracy. **Judge architectures by
    ablations and independent reproductions — never by the story in the abstract.**
11. **The most credible successes are hybrids, not replacements.** ML-accelerated CFD,
    solver-in-the-loop training, PINO and ML interatomic potentials all keep a physics engine
    in the loop, and let ML supply only the part the physics engine is bad at. **Pure
    end-to-end neural PDE solving is the least successful branch of this field — despite
    receiving the most attention.**

---

## 14.7 The habit that matters more than any architecture

Every time you report a speedup, report:

- **(a)** the wall-clock cost of the classical solve **at the accuracy your model actually
  reaches**;
- **(b)** the total cost of generating the training data;
- **(c)** the number of queries at which the surrogate breaks even.

**Almost nobody does this.** If you do, you will be a more reliable engineer than most of the
literature you are reading.
