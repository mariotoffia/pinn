---
title: Frameworks and Tools
subtitle: What to install, what to avoid, and what is quietly dead
minutes: 20
---

# 11 — Frameworks and Tools

All status, versions and star counts were checked against the PyPI JSON API and the GitHub API
on **22 August 2026.** Where a GitHub web page disagreed with the API, the API was trusted.

**Baseline assumed throughout: a local CPU (any OS) plus Colab for GPU jobs — no local NVIDIA
GPU. If you do have one, the CUDA-only entries below open up for you.**

---

## 11.1 What to install — three lines

```bash
# 1. The PINN library + PyTorch CPU backend. Pure Python, installs clean on any OS.
pip install deepxde torch                       # then: export DDE_BACKEND=pytorch

# 2. Ground-truth generators + visualisation. No compiled code, no conda, no pain.
pip install scikit-fem py-pde matplotlib pyvista

# 3. Differentiable simulation, for the Physics-based Deep Learning book's notebooks.
pip install phiflow
```

Then `dde.config.set_default_float("float64")` and **stay on the CPU.**

---

## 11.2 Tier 1 — PINN-first libraries

| Library | Backend | Stars | Last release | Maintained? | CPU-only | Best for |
|---|---|---:|---|---|---|---|
| **DeepXDE** | TF1, TF2, PyTorch, JAX, Paddle | 4,380 | v1.15.0 · 2025-12-05 | **Yes** (pushed 2026-08-18) | **Yes** | **The default.** Widest set of worked examples |
| **NVIDIA PhysicsNeMo** | PyTorch + CUDA | 3,184 | v2.1.1 · 2026-06-08 | Yes (pushed 2026-08-22) | **No — macOS unsupported** | Industrial multi-GPU physics ML |
| **PhysicsNeMo-Sym** | PyTorch + CUDA | 335 | v2.4.0 · 2026-03-09 | **ARCHIVED 2026-05-29** | No | Nothing — use `physicsnemo[sym]` |
| **PINA** | PyTorch + Lightning + PyG | **2** ⚠ (repo reset) | v0.3.2 · 2026-08-01 | Package yes; **repo anomaly** | Yes | Watch it — do not build on it yet |
| **TorchPhysics** | PyTorch | 472 | v1.1.2 · 2026-06-12 | Yes — **org moved** | Yes | PINN + Deep Ritz + DeepONet + FNO in one |
| **NeuroMANCER** | PyTorch | 1,368 | v1.5.6 · 2025-09-26 | Yes (pushed 2026-08-05) | Yes | Constrained optimisation, control, MPC |
| **NeuroDiffEq** | PyTorch | 789 | v0.7.0 · 2025-07-08 | Slow but alive | Yes | ODE systems, solution bundles, teaching |
| **IDRLnet** | PyTorch | 249 | v2.0.0 · **2023-06-30** | **Dead** | — | Nothing |
| **SciANN** | Keras/TF2 | 367 | v0.7.0.1 · **2023-02-16** | **Dead** | — | Nothing |
| **PyDEns** | TF | 315 | v1.0.2 · **2022-01-20** | **Dead** | — | Nothing |
| **NeuralPDE.jl** | Julia / Lux.jl | 1,215 | v6.2.2 | **Yes** (pushed 2026-08-21) | Yes | Symbolic PDE input, Bayesian PINNs |

### DeepXDE — the default choice

`lululxvi/deepxde` · 4,380 ★ · [docs](https://deepxde.readthedocs.io) ·
[paper: SIAM Review 63(1)](https://arxiv.org/abs/1907.04502) · **License: LGPL-2.1** ← note
this if you plan commercial use.

**Backends, and the coverage catch.** DeepXDE advertises five backends, but *example coverage
is very uneven*. Of 33 verified forward examples:

| Backend | Examples supported |
|---|---:|
| `tensorflow.compat.v1` (TF1) | 30 / 33 |
| **PyTorch** | **26 / 33** |
| TensorFlow 2.x | 21 / 33 |
| **JAX** | **10 / 33** |
| PaddlePaddle | 32 / 33 |

**Use the PyTorch backend.** The JAX backend covers less than a third of the gallery, and has a
different Hessian API (`dy_xx, _ = dde.grad.hessian(...)` returns a tuple) — so example code is
**not copy-paste portable between backends.** Fractional PDEs and integro-differential
equations are **TF1/Paddle only.** Set the backend explicitly:
`DDE_BACKEND=pytorch python script.py` — autodetection picks whatever it finds first, and will
silently change under you.

**The examples are short** — a 31-line 2D Poisson solver on an L-shaped domain is
about as good as "read it in full" ever gets. Chapter 12 has the full gallery with line counts.

**Its strongest technical feature is adaptive sampling**: composable geometry primitives
(`Interval`, `Rectangle`, `Disk`, `Polygon`, `Sphere`, plus union/difference/intersection) with
`GeometryXTime`, and **residual-based adaptive refinement (RAR)** and periodic resampling built
in.

**Verdict: install this first.** It is the only library whose example gallery is both broad
*and* short enough to actually read. Check the LGPL with a lawyer before vendoring or modifying
it inside a commercial product; using it as an unmodified pip dependency is normally fine.

### NVIDIA PhysicsNeMo — industrial-grade, and CUDA-only

`NVIDIA/physicsnemo` · 3,184 ★ · Apache-2.0 · [docs](https://docs.nvidia.com/physicsnemo/latest/overview.html)

`NVIDIA/modulus` now redirects here — the rename is complete. `pip install nvidia-physicsnemo`,
`import physicsnemo`. The legacy PyPI packages `nvidia-modulus` 0.9.0 and `nvidia-modulus.sym`
1.8.0 are frozen at 2024-11-27; **do not install them.**

**On PhysicsNeMo-Sym:** it **still exists but was archived on 2026-05-29** —
*"physicsnemo-sym has been upstreamed into NVIDIA/physicsnemo."* The symbolic/PINN half now
ships as `pip install "nvidia-physicsnemo[sym]"`, exposed as `physicsnemo.sym`. **Note: the
v2.0 migration removed the pre-built PDE classes** in favour of inline SymPy definitions, so
older Modulus-Sym tutorials will not run as written.

**The show-stopper for anyone without an NVIDIA GPU:** the official system requirements list
**Ubuntu 24.04 and Windows** as the supported systems; **macOS is not supported**, and an
NVIDIA GPU (Turing T4 or newer) is required. This is not "CPU is slow" — CPU-only setups are
simply unsupported. **Skip unless you have the hardware.** Read the
[Lid-Driven Cavity PINN tutorial](https://docs.nvidia.com/physicsnemo/latest/user-guide/pinns-tutorials/lid_driven_cavity_flow.html)
for the physics if you like; revisit the framework only if you get an NVIDIA machine.

### TorchPhysics — alive, and it moved (with poor signage)

**The repo moved from Bosch Research to https://github.com/Qewton-Labs/torchphysics.** PyPI's
project URLs now point there; `boschresearch/torchphysics` redirects, and raw file requests
against the old path return 404 on every branch. **Update bookmarks and any pinned git
dependencies.**

v1.1.2 · 2026-06-12 · **Apache-2.0** (friendlier than DeepXDE's LGPL) ·
docs https://torchphysics.ai/ · originally Bosch × University of Bremen (ZeTeM).

**Unusually wide scope for its size:** PINNs, the **Deep Ritz method**, DeepONets and
physics-informed DeepONets, **Fourier Neural Operators** and physics-informed FNO, PCANN
model-order reduction — plus mesh-free domain construction with composable primitives and
boolean operations. It has the cleanest separation of *domain* from *sampling strategy* of any
library here.

**Verdict: the best "second library" after DeepXDE** — and the one to reach for when you want
to compare PINNs against Deep Ritz or an operator baseline without learning a second API.
Its first-party tutorials are runnable notebooks —
[the PINN introduction](https://colab.research.google.com/github/Qewton-Labs/torchphysics/blob/main/examples/tutorial/Introduction_Tutorial_PINNs.ipynb)
and [a physics-informed DeepONet](https://colab.research.google.com/github/Qewton-Labs/torchphysics/blob/main/examples/tutorial/Tutorial_PIDeepONet.ipynb)
open in Colab (add `!pip install torchphysics` up top).

### PINA — good ideas, and a confusing house move

The confusing part first: the project **moved house, and both addresses exist**. Development
lives at **https://github.com/pina-org/PINA** (786 ★, full history, **v0.3.2 · July 2026 ·
MIT**); the old `mathLab/PINA` remains online and stops at **v0.2.6** — so which repo (and
which docs build) you land on decides which API you read about, and **0.2 → 0.3 was a
redesign, not a patch.**

Technically attractive: PyTorch + Lightning + PyG, PINNs and neural operators under one
interface, pure Python — and its **24 tutorials each open in Colab** with an install cell,
starting with
[tutorial 1](https://colab.research.google.com/github/pina-org/PINA/blob/master/tutorials/tutorial1/tutorial.ipynb).
**Verdict: fine for guided experiments through the tutorials. Before building on it, pin
`pina-mathlab==0.3.*` and check the docs page you are reading says 0.3, not 0.2.6** — mixing
the two costs hours of API drift.

### NeuroMANCER — excellent, but not a PINN library

`pnnl/neuromancer` · 1,368 ★ · BSD-2-Clause · [docs](https://pnnl.github.io/neuromancer/) —
Pacific Northwest National Laboratory's framework for **differentiable programming**:
parametric constrained optimisation, physics-informed system identification, and
**differentiable model predictive control.**

**If your interest in PINNs is really "I want to embed physics in a controller or an
optimiser," this is a better fit than any PINN library.** If you want to solve a PDE, it is
the wrong tool.

### IDRLnet, SciANN, NeuroDiffEq, PyDEns

Three of these four are dead.

- **IDRLnet** — last release 2023-06-30, last push 2024-10-21. **Dead.**
- **SciANN** — **dead, and confusingly so.** `sciann/sciann` is now a **1-star stub** whose
  README says *"Moved to [`ehsanhaghighat/sciann`](https://github.com/ehsanhaghighat/sciann)."*
  The real repo was last pushed 2023-12-20, and PyPI is stuck at 0.7.0.1 (Feb 2023). It is a
  Keras/TF2 wrapper, so it also inherits the TF-on-Apple-Silicon install pain, for zero
  benefit. It still appears in older link lists — which is exactly why it is worth naming.
  **Doubly avoid.**
- **NeuroDiffEq** — **the only survivor.** 789 ★, pushed 2026-04-22, v0.7.0 (2025-07-08), MIT,
  with a 2025 follow-up paper ([arXiv 2502.12177](https://arxiv.org/abs/2502.12177)). Its niche
  is useful: **solving ODEs and ODE systems with "solution bundles"** — one network
  trained across a whole range of parameters or initial conditions at once — plus inverse
  problems. **Fine as a supplement; wrong as a foundation.**
- **PyDEns** — last release 2022-01-20. **Dead.**

### Julia: NeuralPDE.jl and SciML — is the detour worth it?

Everything here checks out as intensely alive. Every package below was pushed within two weeks
of the check — **the healthiest ecosystem in this whole survey, by activity.**

| Package | Registry version | Stars | Last push |
|---|---|---:|---|
| **NeuralPDE.jl** | 6.2.2 | 1,215 | 2026-08-21 |
| **DifferentialEquations.jl** | 8.0.3 | 3,147 | 2026-08-21 |
| **ModelingToolkit.jl** | 11.39.1 | 1,658 | 2026-08-22 |
| **DiffEqFlux.jl** | 4.9.0 | 922 | 2026-08-18 |
| **Lux.jl** | 1.31.4 | 724 | 2026-08-10 |
| SciMLBenchmarks.jl | — | 344 | 2026-08-21 |

**What Julia does better.** NeuralPDE.jl lets you write the PDE **symbolically**, via
ModelingToolkit, and the physics-informed loss is *generated for you* — you declare
`Dt(u(t,x)) ~ Dxx(u(t,x))` instead of hand-coding Hessian calls. That is a real abstraction
win. It also ships **Bayesian PINNs**, DAEs, integro-differential equations, and
the Deep Galerkin Method — and the surrounding differential-equations ecosystem is the
best in any language.

**A friction point:** the published stable docs were still built for v5.18.1 (March 2025) while
the registry serves 6.2.2. A whole-major-version gap between docs and package means tutorials
may not match the API you installed. The docs also now steer you to **Lux.jl over Flux.jl**,
citing type-promotion and precision bugs in Flux — worth noting, since precision is exactly
what PINNs are sensitive to.

**Verdict for a Go/Python engineer: no, not now — with one exception.** Learning Julia is not a
weekend: a new language, package manager, debugger story, compile-latency workflow, and a much
smaller pool of Stack Overflow answers — all while *also* learning PINNs. **Two hard things at
once is how learning paths die.** The exception: **if your day job becomes stiff ODE/DAE
systems, parameter estimation, or scientific simulation as a primary deliverable**, the SciML
stack is worth learning on its own merits, and NeuralPDE.jl comes free with it. Do it as a
*separate* project, after you can write a PINN from scratch in PyTorch. Bookmark
[SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl) either way — a useful
cross-language sanity check on solver performance claims.

---

## 11.3 Tier 2 — differentiable simulation

| Project | Backends | Stars | Version · date | Status | Mac/CPU |
|---|---|---:|---|---|---|
| **PhiFlow** | NumPy, PyTorch, **JAX**, TF | 1,926 | v3.4.0 · 2025-08-02 | Active | **Yes** |
| **JAX-Fluids** | JAX | 626 | v0.2.1 · 2025-03 | Active | Yes |
| **JAX-CFD** | JAX | 960 | v0.2.1 · 2024-05 | **UNMAINTAINED** | — |
| **XLB** | JAX, Warp, Neon | 502 | v0.3.1 · 2026-01-12 | Active | CPU install available |
| **Taichi** | CPU, CUDA, **Metal**, Vulkan | 28,336 | v1.7.4 · 2025-07-31 | Slowing | **Yes — Metal** |
| **NVIDIA Warp** | CPU, CUDA | 7,024 | v1.16.0 · 2026-08-03 | Very active | CPU yes; GPU = NVIDIA only |
| **Exponax** | JAX | 225 | — | Active | Yes |

**[PhiFlow](https://github.com/tum-pbs/PhiFlow) (TUM, Thuerey group) is the one to install.**
MIT license, four backends, and — decisively — **it is the framework the *Physics-based Deep
Learning* book is built on**, so the book's notebooks and the library reinforce each other.
Differentiable Navier–Stokes on a CPU at teaching resolutions is entirely practical.

**JAX-CFD is dead, and says so.** The README carries an explicit banner: *"🚨 JAX-CFD is no
longer maintained. For alternatives, consider JAX-Fluids, Phi Flow or Exponax."* **Note the
trap:** the repo was pushed as recently as 2026-07-08, so activity-sniffing tools will call it
healthy. Those are housekeeping commits; the last release was 2024-05-13. Its notebooks are
still instructive reading.

**Taichi** has a **Metal backend**, making it one of
very few tools here that can use an Apple GPU. But the release cadence has stalled —
13 months since the last one. Healthy, decelerating.

**NVIDIA Warp** is more relevant than its name suggests: it explicitly supports **CPU execution
on Apple Silicon macOS.** GPU acceleration needs an NVIDIA card, but the CPU path is real and
supported.

### Neural ODEs / differentiable solvers

| Project | Stars | Version · date | Status | Verdict |
|---|---:|---|---|---|
| **[Diffrax](https://github.com/patrick-kidger/diffrax)** | 2,086 | v0.7.2 · 2026-02-18 | Active | **Best in class.** ODE/SDE/CDE in JAX |
| **[torchdiffeq](https://github.com/rtqichen/torchdiffeq)** | 6,475 | v0.2.5 · 2024-11-21 | Stale (pushed 2025-04) | Still the PyTorch default; stable, not evolving. Fine on CPU |
| **[torchsde](https://github.com/google-research/torchsde)** | 1,726 | v0.2.6 · 2023-09-26 | Stale | Works, effectively frozen |

Diffrax pairs with **Equinox** (2,955 ★, v0.13.8) and **Optimistix**. If you touch JAX at all,
this is the corner of the ecosystem worth learning.

### The Physics-based Deep Learning book — the spine of this path

**https://physicsbaseddeeplearning.org — live. Version 0.3, the "GenAI Edition."**
Repo [tum-pbs/pbdl-book](https://github.com/tum-pbs/pbdl-book) · 1,358 ★ · Thuerey et al., TUM.

A Jupyter Book with **Colab links on the notebooks** — *"all code examples can be executed on
the spot, from your browser."* Chapters: Models & Equations → Neural Surrogates and Operators →
**Physical Losses (PINNs)** → **Differentiable Physics** → Probabilistic Learning →
**Reinforcement Learning** → Improved Gradients → Fast Forward Topics.

**This is the single best hands-on resource in the space.** Two things put it above any
library's docs: it teaches **why PINNs underperform**, not only how to run them, and it places
PINNs in context as *one* technique alongside differentiable physics and neural operators —
which is the right framing. Note the repo has not been pushed for about twelve months, so
"current" means current as of mid-2025.

---

## 11.4 Tier 3 — supporting infrastructure

**L-BFGS:** see Chapter 05 §5.2 for the full comparison and pitfalls. Summary: start with
`torch.optim.LBFGS(..., line_search_fn="strong_wolfe")` in float64 on the CPU; fall back to
`scipy.optimize.minimize(method="L-BFGS-B")`, the most battle-tested implementation in
existence. **jaxopt is deprecated** — use **Optax** for L-BFGS and **Optimistix** for
root-finding in JAX.

**Classical solvers as baselines:** see Chapter 07 §7.3. Summary: **scikit-fem** (pure Python,
trivial install, real FEM) and **py-pde** (MIT, finite differences, the shortest path to a 1D
reference solution).

**Visualisation:** matplotlib for fields, PyVista for 3D, Plotly for interactivity.

---

## 11.5 Link lists, indexes and benchmarks

| Resource | Stars | Last update | Verdict |
|---|---:|---|---|
| **[awesome-pinns](https://github.com/AI-in-Transportation-Lab/awesome-pinns)** | 134 | **2026-08-22** | **Best index.** Auto-updated from arXiv; **1,023 papers**, MIT. (Repo recently moved from `gauravfs-14` to this org) |
| [erfanhamdi/awesome-pinn](https://github.com/erfanhamdi/awesome-pinn) | 55 | 2026-06-14 | Small, human-curated, CC0. Decent library section |
| [idrl-lab/PINNpapers](https://github.com/idrl-lab/PINNpapers) | 1,532 | **2023-12-08** | **Stale — 2.5 years.** Popular but frozen; pre-2024 classics only |
| **[i207M/PINNacle](https://github.com/i207M/PINNacle)** | 448 | 2026-07-24 | **Best PINN benchmark.** NeurIPS 2024; 20 cases × 11 methods, MIT |
| **[pdebench/PDEBench](https://github.com/pdebench/PDEBench)** | 1,191 | 2026-03-30 | **Best SciML dataset benchmark.** MIT |
| [tum-pbs/apebench](https://github.com/tum-pbs/apebench) | 108 | 2026-06-01 | NeurIPS 2024; autoregressive emulator benchmark. **No dataset to download** |
| [SciML/SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl) | 344 | 2026-08-21 | Cross-language solver benchmarks |
| [DeepXDE research page](https://deepxde.readthedocs.io/en/latest/user/research.html) | — | live | Hundreds of papers sorted by application — useful for checking whether anyone has tried PINNs on *your* problem |

**PINNacle is the one to read.** It benchmarks vanilla PINNs against loss reweighting
(LRA, NTK), adaptive resampling (RAR), MultiAdam, gPINN, hp-VPINN, adaptive activations
(LAAF/GAAF) and FBPINNs, across 20 problems. **The antidote to cherry-picked paper
results** — it tells you which of the dozen published "PINN improvements" actually generalise.
(It is built on DeepXDE — one more reason DeepXDE is the right foundation.)

---

## 11.6 AVOID / DEAD

**Dead — do not start a project here:**

- **PyDEns** — last release 2022-01-20.
- **SciANN** — last release 2023-02-16, and the top search result is a 1-star stub.
  **Doubly avoid.**
- **IDRLnet** — last release 2023-06-30.
- **jaxopt** — README: *"JAXopt is no longer maintained nor developed."* **Still recommended
  all over the internet.** Use Optax / Optimistix.
- **JAX-CFD** — README banner says unmaintained; the recent pushes are housekeeping. Use
  JAX-Fluids, PhiFlow or Exponax.
- **`nvidia-modulus` / `nvidia-modulus.sym` on PyPI** — frozen at 2024-11-27. Superseded by
  `nvidia-physicsnemo`.
- **`NVIDIA/physicsnemo-sym`** — archived 2026-05-29, merged upstream. The v2.0 migration
  removed the pre-built PDE classes, so all older Modulus-Sym tutorials are broken as written.
- **`fenics` on PyPI (2019.1.0)** — that is legacy FEniCS from 2019, *not* FEniCSx. Install
  `fenics-dolfinx` from conda-forge.
- **`maziarraissi/PINNs`** — 6,098 ★ and historically important, but TF1 and unmaintained by
  its own admission. **Read, never run.** Its star count keeps it at the top of search
  results — that is the trap.
- **`idrl-lab/PINNpapers`** — frozen since December 2023. Use
  [awesome-pinns](https://github.com/AI-in-Transportation-Lab/awesome-pinns).
- **`lululxvi/deeponet`** — 821 ★, 16 commits, one release in Dec 2020, **CC BY-NC-SA 4.0
  (non-commercial!)**, pins DeepXDE v0.11.2. **Effectively abandoned reproduction code.** Use
  DeepXDE's own DeepONet implementation instead.

**Stale — usable, but expect no fixes:** torchdiffeq, torchsde, hjmshi/PyTorch-LBFGS, Taichi,
`okada39/pinn_cavity` (TF 2.1, ~30 h training, pushed 2020).

**Requires CUDA — unusable without an NVIDIA GPU:** NVIDIA PhysicsNeMo (macOS explicitly
unsupported), [jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi) (README says
GPU-only, CUDA 12.4 — excellent code; read it, and run it on Colab), NVIDIA Warp's GPU path.

**Handle with care:** PINA (repo reset + docs one major version behind), TorchPhysics (org
moved; old raw URLs 404), DeepXDE's LGPL-2.1 (the only non-permissive license among the Tier 1
libraries you would realistically use).
