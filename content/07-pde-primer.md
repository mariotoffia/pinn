---
title: PDE Primer and Classical Baselines
subtitle: Enough numerics to know what you are competing with — and the solvers that generate your ground truth
minutes: 18
---

# 07 — PDE Primer and Classical Baselines

**A PINN with no reference solution is a plot, not a result.** This chapter gives you (a) the
minimum PDE vocabulary needed to read the literature, and (b) the classical solvers that produce
trustworthy ground truth on your machine, with no compilation pain.

**Target: about 12 hours**, most of it hands-on.

---

## 7.1 The vocabulary you need

**A PDE (partial differential equation) links a field to its own derivatives.** A "field" is a
quantity defined everywhere in space and time, like temperature or velocity. The classic
equations, and what each one teaches you:

| Equation | Form | Type | What it teaches |
|---|---|---|---|
| **Heat / diffusion** | `u_t = ν·u_xx` | parabolic | Smoothing. Sharp features fade fastest. The *friendliest* PINN problem |
| **Wave** | `u_tt = c²·u_xx` | hyperbolic | Transport without loss. Information travels at a finite speed |
| **Poisson / Laplace** | `∇²u = f` | elliptic | Steady state, no time. The purest test of a spatial approximation |
| **Burgers (viscous)** | `u_t + u·u_x = ν·u_xx` | nonlinear parabolic | Nonlinearity plus near-shocks. **The canonical PINN benchmark** |
| **Navier–Stokes** | `u_t + (u·∇)u = −∇p/ρ + ν∇²u`, `∇·u = 0` | mixed | Coupled, constrained (incompressibility) — the real thing |

(Here `u_t` is shorthand for ∂u/∂t, and `u_xx` for ∂²u/∂x².)

**Play before you read.** **[VisualPDE](https://visualpde.com/)** — `interactive` `free` — runs
these equations live in your browser: poke the [heat equation](https://visualpde.com/basic-pdes/heat-equation.html),
then the [wave equation](https://visualpde.com/basic-pdes/wave-equation.html), and watch how
differently they treat a disturbance. Twenty minutes here builds more intuition than an hour of
reading, and it costs nothing to install.

**Well-posedness (Hadamard):** a problem is well-posed if a solution *exists*, is *unique*, and
depends *continuously* on the input data. All three matter for PINNs:
- No existence → you are minimising toward nothing.
- No uniqueness → a low residual is compatible with the *wrong* solution. This is why the
  initial and boundary conditions carry so much weight, and why soft penalties are dangerous.
- No continuous dependence → a tiny residual does not bound the solution error. That is the
  constant `C` in `‖u − û‖ ≤ C·‖residual‖` — for stiff problems it is huge.

**Boundary conditions** — the rules imposed at the edge of the domain. You will implement all of
these:
- **Dirichlet:** `u = g` on the boundary — prescribe the value.
- **Neumann:** `∂u/∂n = g` — prescribe the flux through the boundary.
- **Robin:** `a·u + b·∂u/∂n = g` — a mix of the two.
- **Periodic:** `u(x) = u(x + P)` — the domain wraps around. In a PINN this is the one you
  should **build into the architecture** rather than penalise (Chapter 10, Step 4).

**Strong form vs weak form.** A PINN checks the equation *pointwise* — so the network must be
smoothly differentiable, up to the order of the PDE, **everywhere**. A finite element method
instead integrates the equation against test functions, which moves derivatives off the
solution; it only needs `u ∈ H¹` (one weak derivative). **This is a real, structural tightening
of the requirement**, and it is one reason PINNs struggle where FEM is comfortable: sharp
interfaces, discontinuous coefficients, corner singularities.

**Non-dimensionalisation** means rescaling the problem so `x, t` and `u` are all of order 1
(roughly between −1 and 1). For PINNs this is a correctness requirement:
Glorot initialisation *assumes* moderate-range inputs, and loss terms with different physical
units are otherwise numerically incomparable before training even starts. See Chapter 10,
Step 0.

**Dimensionless numbers you will meet:** Reynolds `Re = UL/ν` (inertia vs viscosity), Péclet
`Pe = UL/D` (advection vs diffusion), and the CFL number (a stability condition for explicit
schemes). When a paper says "PINNs fail at high Re," it means the convection-dominated regime
from Chapter 09.

---

## 7.2 The three classical method families, in one table

| | Idea | Strength | Weakness | Use it as |
|---|---|---|---|---|
| **Finite differences (FD)** | Replace derivatives with local stencils on a grid | Trivial to write, fast, transparent | Awkward geometry, boundary handling | **Your 1D ground truth. 30 lines of NumPy** |
| **Finite volume (FV)** | Integrate over cells, track fluxes | Conservation built in; handles shocks | More machinery | Conservation laws, CFD |
| **Finite elements (FEM)** | Weak form, piecewise basis on a mesh | Arbitrary geometry, rigorous error theory, adaptivity | Meshing, assembly | **Your 2D ground truth, and the baseline you will be compared against** |
| **Spectral** | Global basis (Fourier/Chebyshev) | Exponential convergence on smooth periodic problems | Needs smoothness and simple domains | Fast reference solutions for benchmarks |

**Why this matters:** FEM has sixty years of error analysis, error estimators, convergence
theory and certification behind it. A PINN has a test-set number. When you report a PINN result,
you are implicitly claiming to beat all of that. Chapter 14 has the measured outcomes.

Background reading if you want it: MIT's
[Learn Differential Equations (Strang & Moler)](https://ocw.mit.edu/courses/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/)
for the numerical-methods segments, and Chris Rackauckas's
[Parallel Computing and Scientific Machine Learning](https://book.sciml.ai/) for the best
treatment anywhere of *where PINNs sit among the alternatives*.

---

## 7.3 The solvers to install — ranked by install effort

| Solver | Method | Version (Aug 2026) | License | Install |
|---|---|---|---|---|
| **[scikit-fem](https://github.com/kinnala/scikit-fem)** | FEM | v12.0.2 · 2026-06-05 | BSD-3 | **Trivial — pure Python, no compiled code** |
| **[py-pde](https://github.com/zwicker-group/py-pde)** | Finite differences | v0.58.0 · 2026-07-10 | MIT | **Trivial** — pip/conda, numba JIT |
| **[FiPy](https://github.com/usnistgov/fipy)** | Finite volume | v4.0.3 · 2026-06-18 | NIST public domain | Easy — pip/conda |
| **[FEniCSx / DOLFINx](https://github.com/FEniCS/dolfinx)** | FEM | v0.11 · June 2026 | LGPL-3.0 | Moderate — **conda-forge only** |
| **[Firedrake](https://www.firedrakeproject.org/install.html)** | FEM | v2026.4.1 | LGPL-3.0 | **Hard** — build PETSc from source |
| **[deal.II](https://github.com/dealii/dealii)** | FEM (C++) | active | LGPL | Hard — C++ toolchain |

**Install scikit-fem on day one.** Its README states it *"has minimal dependencies"* and
*"contains no compiled code"* — pure Python 3.10+, NumPy and SciPy only, assembling standard
SciPy sparse matrices that you solve yourself. Nothing to compile, no MPI, no PETSc, no conda
required on any platform. It supports 1D, triangle, quad, tet and hex elements plus
Raviart–Thomas, Nédélec, MINI, Crouzeix–Raviart and Argyris — a real FEM library, not a toy.
[Colab notebooks](https://github.com/kinnala/scikit-fem-notebooks).

**py-pde is the runner-up, and the better choice for *time-dependent* problems** — finite
differences on structured grids, with transparent numba/JAX/torch compilation, plus a Binder
link so you can try it with zero install. **For a 1D Burgers or heat-equation reference
solution, py-pde is the shortest path in existence.**

**FEniCSx** is the "serious" answer for real FEM, and the docs bless macOS via conda:
`conda install -c conda-forge fenics-dolfinx mpich pyvista`. Use a **dedicated conda
environment** — mixing conda-forge MPI into a pip-based ML environment is a classic way to break
your PyTorch install. Also: the `fenics` package on PyPI is **2019.1.0, from 2019** — that is
the legacy "FEniCS classic", a different and older project. Do not install it.

**Firedrake** officially supports ARM Macs, but installing means cloning and compiling PETSc
with specific flags, and the docs devote a long section to "Common installation issues." An
excellent library with the wrong cost/benefit for a learning path. Use their Colab notebooks if
curious.

---

## 7.4 Visualisation

- **Matplotlib** — the workhorse patterns for PDE fields: `pcolormesh`/`imshow` for scalar
  fields on structured grids, `tricontourf` for unstructured FEM output, `quiver`/`streamplot`
  for vector fields, shared `vmin`/`vmax` across panels, and **a diverging colormap centred at
  zero for error plots.**
  **Always plot prediction, reference and *error* side by side.** A PINN that looks right and
  is 5% wrong is the standard failure mode — only the error panel reveals it.
- **PyVista** — https://docs.pyvista.org/ — VTK made Pythonic; the right tool for 3D fields,
  meshes, streamlines and isosurfaces. Plays well with FEniCSx. Use the static or `trame`
  backends in Jupyter.
- **Plotly** — best for interactive 3D surfaces and notebooks you plan to share.

---

## 7.5 The exercise for this chapter

Do all three. They take one afternoon, and they change how you read every PINN paper afterwards.

**1. Write a 1D heat-equation solver in NumPy.** Explicit finite differences, 30 lines. Then
watch it blow up when you violate the CFL stability condition `ν·Δt/Δx² ≤ 1/2`. That
instability is something a PINN does not have — and also the flip side of why a PINN has no
convergence order to report.

**2. Solve 2D Poisson on an L-shaped domain with scikit-fem, ~15 lines.** The re-entrant corner
creates a genuine singularity in the solution. Save the result: you will use it as ground truth
in Chapter 12, when you run DeepXDE's 31-line `Poisson_Lshape.py` and overlay the error.

**3. Generate a Burgers reference with py-pde** at `ν = 0.01/π`, `x ∈ [-1,1]`, `t ∈ [0,1]`,
`u(0,x) = −sin(πx)`. Watch the sharp internal layer form near `x = 0` around `t ≈ 0.4`. **That
near-shock is why Raissi et al. chose this problem** — and having your own reference means you
can compute an independent relative L² error instead of trusting a bundled `.npz` file.

**And build one habit now:** every time you report a speedup, also report (a) the wall-clock
cost of the classical solve *at the accuracy your model actually reaches*, (b) the total cost of
generating any training data, and (c) the number of queries needed before the surrogate breaks
even. Chapter 14 explains why: 79% of published papers claiming to beat numerical methods on
fluid PDEs compared against a weak baseline.

---

## 7.6 Solve it in your browser — the classical labs, one click each

Everything here runs in Colab with **no local install** (or one stated install cell). This is
the classical craft, hands-on — the thing Chapter 09 measures PINNs against.

- **CFD Python: 12 steps to Navier–Stokes (Lorena Barba)** — `notebook course` `free` `CC-BY` —
  https://github.com/barbagroup/CFDPython — the classic. Sixteen NumPy-only notebooks from 1D
  linear convection all the way to a 2D cavity flow. Start at
  [Step 1](https://colab.research.google.com/github/barbagroup/CFDPython/blob/master/lessons/01_Step_1.ipynb "The start of Barba’s 12-steps-to-Navier–Stokes; steps 1–4 plus the CFL lesson teach stability by feel") and do at least steps
  1–4 plus the CFL lesson — the fastest way to *feel* stability limits and truncation error.
- **Numerical Analysis with Applications in Python (John S. Butler, TU Dublin)** —
  `notebook book` `free` `MIT` — https://john-s-butler-dit.github.io/NumericalAnalysisBook/ —
  derives and codes the heat-equation solvers this chapter talks about:
  [FTCS](https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2008%20-%20Heat%20Equations/801_Heat%20Equation-%20FTCS.ipynb "The explicit (FTCS) heat-equation solver derived and coded, stability discussion attached")
  and
  [BTCS](https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2008%20-%20Heat%20Equations/802_Heat%20Equation-%20BTCS.ipynb "The implicit (BTCS) counterpart — what unconditional stability buys"),
  stability discussion attached. Pairs exactly with exercise 1 above.
- **Practical Numerical Methods with Python (Barba et al.)** — `notebook course` `free` `CC-BY` —
  https://github.com/numerical-mooc/numerical-mooc — the broader sibling: convection, diffusion,
  Burgers, then **Module 5's iterative Laplace/Poisson solvers** — the elliptic baselines PINN
  papers compare against. Try
  [the 2D Laplace lesson](https://colab.research.google.com/github/numerical-mooc/numerical-mooc/blob/master/lessons/05_relax/05_01_2D.Laplace.Equation.ipynb "Iterative elliptic solvers — the classical baseline PINN papers compare against").
  Archived read-only since August 2026 — frozen, but it runs fine.
- **scikit-fem, zero install** — `notebook` `free` `BSD` — the §7.3 recommendation without even
  a local environment:
  [ex01 — Poisson with unit load](https://colab.research.google.com/github/kinnala/scikit-fem-notebooks/blob/main/ex01.ipynb "Real FEM — mesh, elements, assembly, solve — in about a minute; pip install is its first cell")
  (`pip install scikit-fem` is its first cell). Real FEM — mesh, elements, assembly, solve —
  in about a minute.
- **py-pde, zero install** — `notebook` `free` `MIT` —
  [Tutorial 2 — solving pre-defined PDEs](https://colab.research.google.com/github/zwicker-group/py-pde/blob/master/examples/jupyter/Tutorial%202%20-%20Solving%20pre-defined%20partial%20differential%20equations.ipynb "Solve pre-defined PDEs with finite differences; reuse it for the Burgers reference in §7.5")
  — add one `!pip install py-pde` cell at the top, run, then reuse it for exercise 3.
- **Real FEniCSx inside Colab** — `notebook` `free` — production-grade FEM with no local
  install: open the Dokken tutorial's
  [fundamentals notebook](https://colab.research.google.com/github/jorgensd/dolfinx-tutorial/blob/main/chapter1/fundamentals_code.ipynb "Production-grade FEM in the browser — paste the FEM-on-Colab install cell at the top first (takes a few minutes)")
  and paste the FEniCSx install cell from [FEM on Colab](https://fem-on-colab.github.io/)'s
  Packages page at the top (the install takes a few minutes). One caution: check that the
  fem-on-colab build matches the tutorial's DOLFINx version (0.11.x at the time of writing) —
  if a cell errors on an API name, that mismatch is why. Tutorial text:
  https://jsdokken.com/dolfinx-tutorial/.
