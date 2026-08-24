---
title: Worked Examples
subtitle: What to run, in order — the starter kit and the verified example gallery
minutes: 18
---

# 12 — Worked Examples

Everything below was verified to exist in August 2026. Line counts are actual.

---

## 12.1 The path, in order

1. **[Ben Moseley's Colab workshop notebook](https://github.com/benmoseley/harmonic-oscillator-pinn-workshop)**
   — write a PINN from scratch, no framework.
2. **`starter/scripts/` in this kit** — 00 → 07, from environment check to inverse problem.
3. **DeepXDE `Poisson_Lshape.py`** (31 lines), with a **scikit-fem** FEM solution overlaid as
   ground truth.
4. **`Burgers.py`, then `Burgers_RAR.py`**, back to back — the diff between them *is* the
   lesson about collocation sampling.
5. **`Lorenz_inverse.py`** (92 lines) for inverse problems.
6. **[The Physics-based Deep Learning book](https://physicsbaseddeeplearning.org)**, end to
   end — to understand why PINNs are often the *wrong* answer, and what differentiable physics
   offers instead.
7. **Skim [PINNacle](https://github.com/i207M/PINNacle)** before you believe any paper's
   accuracy claim.

Use Colab's GPU only for Beltrami flow, jaxpi, and neural-operator work.

---

## 12.2 The starter kit in this repository

Everything in `starter/` runs on **CPU, in float64, in minutes** — no GPU, no framework beyond
PyTorch + NumPy + Matplotlib.

```bash
cd starter
uv venv && uv pip install -e .        # or: pip install -e .
python scripts/00_check_environment.py
```

| Script | What it does | Runtime |
|---|---|---|
| `00_check_environment.py` | Proves the four hardware facts on *your* machine: device, float64, second-order autograd, and whether MPS fails as predicted | ~10 s |
| `01_mlp_from_scratch.py` | Forward pass, hand-derived backprop, gradient check against finite differences — **NumPy only, no autograd** | ~15 s |
| `02_autodiff_playground.py` | `∂u/∂x`, `∂²u/∂x²`, the Burgers residual, Idiom A vs Idiom B timing, `gradgradcheck` | ~15 s |
| `03_spectral_bias.py` | Fit `sin(x) + 0.3·sin(15x)` with and without Fourier features. **Watch the high frequency arrive last — or never** | ~40 s |
| `04_pinn_oscillator.py` | Damped harmonic oscillator ODE. Hard-constrained initial condition vs soft penalty, side by side | ~4 min |
| `05_pinn_heat1d.py` | 1D heat equation against an exact analytic solution. Your first real error number | ~2 min |
| `06_pinn_burgers.py` | The canonical benchmark. Adam → L-BFGS, with flags for Fourier features, causal weighting, RWF, gradient-norm balancing and RAR | ~4 min (`--plain`), longer with `--all` |
| `07_pinn_inverse.py` | Recover an unknown diffusion coefficient from sparse noisy data. **The thing PINNs are actually good at** | ~2 min |

`pinnlab/` holds the reusable pieces: `device.py` (device/dtype checks), `nets.py` (tanh MLP
with Glorot init, Fourier features, random weight factorisation), `derivatives.py` (both
autograd idioms), `sampling.py` (uniform / Latin hypercube / Sobol / RAR), `losses.py`
(gradient-norm balancing, causal weighting), `train.py` (the Adam → L-BFGS loop),
`plotting.py` (the three-panel prediction/reference/error figure), and `reference.py` (exact
Burgers and heat solutions).

**`06_pinn_burgers.py` is the ablation machine.** Run it with `--plain`, then with `--all`, and
compare. Every flag corresponds to a numbered step in Chapter 10.

---

## 12.2b The interactive labs (marimo) — and the guided tours

Three chapters of this path have **interactive companion labs** in `notebooks/` — reactive
[marimo](https://marimo.io/) notebooks where sliders re-run real PyTorch experiments in
seconds. They pair with the **▶ Guided tour** buttons on chapters 02, 03, 05, 08 and 09 of
this site. The contract: every lab runs as-is (the solutions are in the cells), so you can
never get stuck — the exercises invite you to replace a solution with your own attempt, with
hints one click away. And each lab file carries its own dependency list in a comment header
(PEP 723), so the `--sandbox` flag below builds the right environment on first run — PyTorch
included. The only tool you install is uv.

```bash
uvx marimo edit --sandbox notebooks/02_autodiff_lab.py     # the hinge exercise, interactive
uvx marimo edit --sandbox notebooks/03_spectral_bias_lab.py
uvx marimo edit --sandbox notebooks/04_oscillator_pinn_lab.py
```

| Lab | Pairs with | The measured result it teaches |
|---|---|---|
| `02_autodiff_lab.py` | Ch. 03 tour | FD vs autodiff ~1e-6; Idiom A vs B ~1e-15; `gradgradcheck` passes |
| `03_spectral_bias_lab.py` | Ch. 05 tour | The fast mode stalls without Fourier features, converges with them |
| `04_oscillator_pinn_lab.py` | Ch. 08 tour | hard IC ≤ soft λ=1 ≪ soft λ=100; L-BFGS off costs ~100× |

**The routing rule** (also in `notebooks/README.md`): small CPU/float64 concept experiments →
marimo locally; excellent existing hosted notebooks (Moseley's workshop, the Tancik demo,
d2l.ai, 6.S191 labs) → Colab, linked from the tours with purpose, prerequisites and expected
results; anything needing a GPU (jaxpi, Beltrami, neural operators) → Colab/Kaggle per
Chapter 04.

---

## 12.3 The starting point — a 1D ODE: the damped harmonic oscillator

**Ben Moseley — "So, what is a physics-informed neural network?"** — `blog + notebook` `free` —
https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/ — published
2021-08-28, last updated 2024-11-03. The structure: data-driven ML → why it fails away from the
data → add the ODE residual → compare the extrapolation. **Still the best single-sitting
introduction.**

- Original code: [benmoseley/harmonic-oscillator-pinn](https://github.com/benmoseley/harmonic-oscillator-pinn)
  — 676 ★, MIT, one notebook. Last pushed 2022-03-22.
- **Better: [benmoseley/harmonic-oscillator-pinn-workshop](https://github.com/benmoseley/harmonic-oscillator-pinn-workshop)**
  — 165 ★, MIT, **last pushed 2025-11-19**, with a real **"Open in Colab" badge.** Two
  notebooks (instructor version + student version with blanks to fill in), PyTorch from
  scratch — **no PINN library at all** — covering both the forward problem *and* inversion.
  [YouTube recording](https://www.youtube.com/watch?v=G_hIppUWcsc).

**This is the correct first thing to do, and the workshop repo is the better of the two.**
Coding a PINN in raw PyTorch before touching any framework is what makes the frameworks
readable afterwards.

Also: **DeepXDE `examples/pinn_forward/ode_system.py`** — 50 lines, supports PyTorch *and* JAX.

---

## 12.4 The verified DeepXDE gallery

**33 forward examples + 11 inverse examples.** Galleries:
[forward](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward.html) ·
[inverse](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse.html)

| Example | Lines | Backends |
|---|---:|---|
| `pinn_forward/Poisson_Lshape.py` | **31** | TF1, TF2, PyTorch, **JAX**, Paddle |
| `pinn_forward/Poisson_Neumann_1d.py` | 38 | TF1, TF2, PyTorch, Paddle |
| `pinn_forward/ode_system.py` | 50 | TF1, TF2, PyTorch, **JAX**, Paddle |
| `pinn_forward/Burgers.py` | **55** | TF1, TF2, PyTorch, Paddle |
| `pinn_forward/Poisson_periodic_1d.py` | 56 | + JAX |
| `pinn_forward/Burgers_RAR.py` | 64 | — |
| `pinn_forward/Poisson_multiscale_1d.py` | 68 | — |
| `pinn_forward/Poisson_Dirichlet_1d.py` | 74 | + JAX |
| `pinn_forward/wave_1d.py` | 81 | TF1, PyTorch, Paddle |
| `pinn_forward/diffusion_1d_exactBC.py` | 82 | — |
| `pinn_forward/diffusion_1d.py` | 83 | + **JAX** |
| `pinn_inverse/Lorenz_inverse.py` | 92 | TF1, TF2, PyTorch, JAX |
| `pinn_forward/Kovasznay_flow.py` | 104 | TF1, TF2, PyTorch, Paddle |
| `pinn_forward/heat.py` | 109 | TF1, TF2, PyTorch, Paddle |
| `pinn_forward/Beltrami_flow.py` | 234 | TF1, TF2, PyTorch, Paddle |

---

## 12.5 By PDE

### Poisson (2D, elliptic) — start your 2D work here

- **`Poisson_Lshape.py` — 31 lines.** 2D Poisson on a non-convex L-shaped domain built from a
  `Polygon`, with Dirichlet boundaries. **PyTorch and JAX supported.** The re-entrant corner
  creates a real solution singularity, so this is a legitimate test, not a toy. **The shortest
  genuinely useful example in this whole survey.**
- The 1D trio — `Poisson_Dirichlet_1d.py`, `Poisson_Neumann_1d.py`, `Poisson_Robin_1d.py` — is
  the cleanest way to learn how each boundary-condition type is written.
- `Poisson_periodic_1d.py` shows periodicity built into the architecture;
  `Poisson_multiscale_1d.py` demonstrates **Fourier features** — the spectral-bias fix, live.
- **Baseline:** scikit-fem solves 2D Poisson on an L-shape in ~15 lines. **Do the FEM version
  first**, then the PINN, then overlay the error. **This single exercise teaches more than any
  blog post.**

### Burgers (1D, nonlinear) — the canonical benchmark

- **`Burgers.py` — 55 lines.** 2,540 collocation points, 15,000 Adam steps + 1,000 L-BFGS. The
  reference dataset `examples/dataset/Burgers.npz` ships with the repo, so the L² error metric
  works offline. **Minutes on a CPU. This is the one to run.**
- **`Burgers_RAR.py` — 64 lines.** The same problem with residual-based adaptive refinement.
  **Run both back to back; the diff is the lesson.**
- **[rezaakb/pinns-torch](https://github.com/rezaakb/pinns-torch)** — 938 ★, BSD-3, pushed
  2026-02-08. A faithful PyTorch re-implementation of the original PINNs suite (Burgers,
  Navier–Stokes, Schrödinger) using CUDA Graphs and TorchScript, claiming up to 9× over the
  TF1 original. The good "modern port of the classic paper." (Its JAX sibling `pinns-jax` is
  stale — skip.) Its guided
  [Schrödinger tutorial](https://colab.research.google.com/github/rezaakb/pinns-torch/blob/main/tutorials/0-Schrodinger.ipynb)
  opens straight in Colab.
- **[maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)** — the original. **Read, do
  not run** — TF1, unmaintained by its own admission, will not install on modern Python.

### Heat / diffusion

- **`heat.py`** (109 lines) and **`diffusion_1d.py`** (83 lines — one of the few with **JAX**
  support; the best entry point if you want to see the JAX code path).
- **`diffusion_1d_exactBC.py`** (82 lines) demonstrates **hard constraints** — building the
  boundary/initial conditions into the network output, so the loss has one less competing
  term. **One of the highest-value PINN tricks; study it early.**
- `heat_resample.py` (112) and `diffusion_1d_resample.py` (85) show periodic resampling.
- **Ground truth:** py-pde in ~10 lines, or `scipy` directly, or the analytic series solution
  used in `starter/scripts/05_pinn_heat1d.py`.

### Wave (hyperbolic)

- **`wave_1d.py` — 81 lines.** TF1/PyTorch/Paddle (**no JAX, no TF2**). Implements the example
  from [arXiv:2012.10047](https://arxiv.org/abs/2012.10047). Note that it uses a
  loss-weighting helper (`get_initial_loss`) — **wave equations are a known hard case for
  vanilla PINNs, and this example is honest about needing help to converge.**
- `Klein_Gordon.py` (96 lines) — TF1/TF2/Paddle only, **no PyTorch.**

### Navier–Stokes — where CPU limits start to bite

Be realistic here. **The 2D lid-driven cavity is where vanilla PINNs stop being cute.** Expect
to need loss weighting, hard boundary constraints, and Fourier features. **If your cavity PINN
does not converge, that is the normal result — not your bug.**

- **`Kovasznay_flow.py` — 104 lines.** Steady 2D Navier–Stokes with an **exact analytical
  solution**, so you get a true error number. **The right first Navier–Stokes problem** — the
  only one where you can prove correctness without a CFD reference.
- **`Beltrami_flow.py` — 234 lines.** 3D unsteady Navier–Stokes with an analytical solution.
  Feasible on CPU but slow; a Colab session is more comfortable.
- **`pinn_inverse/Navier_Stokes_inverse.py`** — the classic cylinder-wake inverse problem from
  the original paper. Doubles as your inverse-problem exercise.
- **[PredictiveIntelligenceLab/jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi)** —
  442 ★, MIT. **The README states it is GPU-only (CUDA 12.4).** It has the best example
  *catalogue* of any PINN research codebase (Allen–Cahn, Stokes, Kuramoto–Sivashinsky,
  lid-driven cavity, Navier–Stokes tori and cylinder, Grey–Scott, Ginzburg–Landau, Kolmogorov
  flow, Rayleigh–Taylor) and implements the modern training tricks that make hard PINNs
  converge. **Read the code; run it on a Colab GPU; do not try it locally.**
- **[okada39/pinn_cavity](https://github.com/okada39/pinn_cavity)** — 164 ★, lid-driven cavity
  at Re=100, TF 2.1, stream-function formulation. **Flags: last pushed 2020-07-16, and the
  repo's own training log shows ~30 hours of optimisation.** The stream-function trick
  (incompressibility enforced by construction) is worth learning. **Read, do not run.**

### Inverse problems — the strongest collection anywhere

DeepXDE has **11 inverse demos.**

- **`Lorenz_inverse.py` — 92 lines.** Recovers the three Lorenz parameters via
  `dde.Variable(1.0)` from 400 points. **PyTorch and JAX.** **The clearest introduction to
  inverse PINNs in existence** — the whole mechanism fits on one screen: declare the unknowns
  as trainable variables, add a data-fit term, train.
- **`diffusion_1d_inverse.py`** — recovers a diffusion coefficient. The canonical "discover a
  PDE coefficient from sparse data" exercise. (`starter/scripts/07_pinn_inverse.py` is a
  from-scratch version of the same idea.)
- **`elliptic_inverse_field.py`** — recovers an entire spatially varying *field*, not just one
  number. Genuinely harder, and more realistic.
- **`Navier_Stokes_inverse.py`** — as above.
- Also in the gallery: Brinkman–Forchheimer parameters, diffusion-reaction systems, a
  fractional Poisson inverse problem, and Lorenz with an external input.
- **Hosted one-click inverse lab:** ETH's
  [Tutorial 04 — PINNs for Inverse Problems](https://colab.research.google.com/github/camlab-ethz/AI_Science_Engineering/blob/main/Tutorial%2004%20-%20PINNs%20for%20Inverse%20Problems%20.ipynb)
  (PyTorch; the course entry is in §12.7 — and yes, the filename really has a space before
  `.ipynb`).

---

## 12.6 General tutorial collections worth your time

- **[Ceyron/machine-learning-and-simulation](https://github.com/Ceyron/machine-learning-and-simulation)**
  — 1,199 ★, MIT, pushed 2026-05-22. Felix Koehler's notebooks and handwritten notes
  accompanying his YouTube channel: **automatic differentiation and adjoints**, CFD, FEM,
  PDEs. **Genuinely excellent and underrated** — the adjoint/autodiff material is the best
  free explanation of *why* differentiable simulation works.
- **[lululxvi/tutorials](https://github.com/lululxvi/tutorials)** — the DeepXDE author's slides
  and PINN lecture material; pairs with his
  [video lecture](https://www.youtube.com/watch?v=Wfgr1pMA9fY).
- **[benmoseley/FBPINNs](https://github.com/benmoseley/FBPINNs)** — 570 ★, pushed 2026-05-29.
  Finite-basis PINNs: domain decomposition for multiscale problems. **The natural next step
  once vanilla PINNs disappoint you.** Active.
- **[neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)** —
  3,821 ★. FNO and neural operators. Not PINNs, but the other major branch — and increasingly
  the more practical one. See Chapter 13.

---

## 12.7 Courses and long-form tutorials on PINNs specifically

- **APMA 2070 / ENGN 2912V: Deep Learning for Scientists and Engineers (George Karniadakis, Brown)** —
  `course` `free` — https://sites.brown.edu/crunch-group/apma-2070-deep-learning-for-scientists-engineers/
  · public materials (Spring 2024): https://github.com/raj-brown/APMA_2070_ENGN_2912_SPRING_2024
  **The definitive PINN course, from the field's founder.** Four modules: Basics → Neural
  Differential Equations (equation discovery, PINNs) → Neural Operators (DeepONet) → SciML
  Uncertainty Quantification, plus multi-GPU SciML. Slides, notebooks, homework, projects.
  The notebooks deep-link straight into Colab:
  [PyTorch primer](https://colab.research.google.com/github/raj-brown/APMA_2070_ENGN_2912_SPRING_2024/blob/main/Lecture_4_Notebook/1-pytorch.ipynb) ·
  [optimisers](https://colab.research.google.com/github/raj-brown/APMA_2070_ENGN_2912_SPRING_2024/blob/main/Lecture_5_Notebook/optimizer_00.ipynb) ·
  [PINNs](https://colab.research.google.com/github/raj-brown/APMA_2070_ENGN_2912_SPRING_2024/blob/main/Lecture_8_Notebook/pinns.ipynb) ·
  [DeepONet](https://colab.research.google.com/github/raj-brown/APMA_2070_ENGN_2912_SPRING_2024/blob/main/Lecture_10_Notebook/operators.ipynb) ·
  [DeepXDE](https://colab.research.google.com/github/raj-brown/APMA_2070_ENGN_2912_SPRING_2024/blob/main/Lecture_11_Notebook/deepXde.ipynb).
  (No license file — run them in place.)
- **Deep Learning for Science and Engineering Teaching Kit (NVIDIA × Brown)** — `course/video`
  `free with account` — https://www.nvidia.com/en-us/on-demand/deep-learning-for-science-and-engineering —
  the polished on-demand version of the course above, with hands-on PhysicsNeMo tutorials.
- **AI in the Sciences and Engineering (Siddhartha Mishra & Ben Moseley, ETH Zürich)** —
  `course` `free` — https://github.com/camlab-ethz/AI_Science_Engineering — **the strongest
  European counterpart to APMA 2070**, from the group behind much of the PINN theory this path
  cites — co-taught by the author of Chapter 08's workshop lab, and taught again in Spring
  2026. Twelve PyTorch tutorial notebooks (function approximation → PINN training → inverse
  PINNs → FNO/CNO → autodiff → GNNs), all openable in Colab straight from the repo, with the
  lecture recordings public. No license file: use the notebooks in place.
- **Scientific Machine Learning (Krishna Kumar, UT Austin)** — `course` `free` —
  https://kks32-courses.github.io/sciml/ — 12 modules: NN foundations and autodiff → PINNs →
  Neural ODEs → operator learning (DeepONet, FNO) → GNNs → SINDy → UQ/Bayesian methods.
  PyTorch and JAX notebooks, MIT-licensed, **last updated August 2025.** **Currently the best
  free, modern, notebook-first SciML course in Python.**
- **Physics-Informed Neural Networks — Cornell Virtual Workshop** — `tutorial` `free` —
  https://cvw.cac.cornell.edu/SciML/pinns/index — authored by Krishna Kumar and Chishiki-AI,
  published **August 2025.** Free, self-paced, in the browser. Why standard NNs fail on physics
  problems, autodiff for PDE derivatives, data-driven vs physics-informed comparison, worked
  examples (damped oscillator, heat equation, inverse problems). **The gentlest good on-ramp.**
  Its six companion notebooks (MLP → PINNs → DeepONet → differentiable simulation) open
  one-click in Colab from
  [chishiki-ai/sciml-course](https://github.com/chishiki-ai/sciml-course) — updated August 2026.
- **Parallel Computing and Scientific Machine Learning, MIT 18.337J (Chris Rackauckas)** —
  `course/book` `free` — https://book.sciml.ai/ · [lectures](https://book.sciml.ai/lectures/) ·
  [repo](https://github.com/SciML/SciMLBook) — Julia-based, and **the best treatment anywhere
  of where PINNs sit among the alternatives.** Start with
  ["Introduction to Scientific Machine Learning through Physics-Informed Neural Networks"](https://book.sciml.ai/notes/03-Introduction_to_Scientific_Machine_Learning_through_Physics-Informed_Neural_Networks/)
  — it derives PINNs from first principles, and is refreshingly unsentimental about their cost
  relative to classical solvers and Universal Differential Equations.
- **Physics Informed Neural Networks — University of Oxford (David Kay)** — `course` `syllabus public`
  — https://www.cs.ox.ac.uk/teaching/courses/2025-2026/pinn/ — **Hilary Term 2026**, 16
  lectures: from classical numerical methods through the mathematics of NNs and optimisation
  to building PINNs, boundary/initial conditions, nonlinear multi-output systems, and
  applications in fluids and phase-field modelling. The materials sit behind university login;
  **the syllabus alone is a well-judged curriculum worth borrowing.**
- **Physics Informed Machine Learning (Steve Brunton, UW)** — `video` `free` —
  https://www.youtube.com/playlist?list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa — 24 videos.
  **The best free conceptual overview available.** Brunton's framing — *what* to model, *how*
  to embed physics, and *when not to* — is exactly the framing this path needs, and he is
  notably honest about the limitations.
- **Lu Lu group teaching (Yale)** — https://lugroup.yale.edu/teaching — S&DS 266/566 and
  S&DS 6890. The slides are not public; the DeepXDE documentation is effectively the practical
  companion.
- **CRUNCH Group seminars (Brown)** — https://sites.brown.edu/crunch-group/ — the
  "Machine Learning + X" seminar archive (2018–2024) is the closest thing the field has to a
  running record of itself.
