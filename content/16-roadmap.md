---
title: Roadmap
subtitle: A 14-week plan with concrete weekly deliverables
minutes: 10
---

# 16 — Roadmap

Assumes **about 8 hours per week.** Adjust freely — the ordering matters more than the pace.

Every week has a **deliverable**: something that exists when the week is over. If you cannot
produce it, do not move on. This is the single most reliable defence against the "I watched a
lot of videos and cannot write a PINN" failure mode.

---

## Phase A — Foundations (weeks 1–4, ~32 h)

### Week 1 — Math reload
**Read:** Ch. 01 in full.
**Do:** 3B1B Essence of Linear Algebra (selected episodes), Strang's six factorisation
lectures, the derivatives half of Khan Academy multivariable. (If derivatives themselves feel
rusty, start with 3B1B's Essence of Calculus.)
**Deliverable:** a one-page note, in your own words, defining gradient, Jacobian, Hessian, and
the multivariable chain rule — with a worked 2-input/1-output example done by hand.

### Week 2 — Backprop by hand
**Read:** Ch. 02 §2.1–2.2. Nielsen Ch. 1–2, **with a pen.**
**Do:** derive the four backprop equations yourself.
**Deliverable:** a NumPy MLP that trains on a toy 1D regression problem, with a **hand-written
backward pass and a finite-difference gradient check that passes.**
→ `starter/scripts/01_mlp_from_scratch.py` is the reference; write yours first, then compare.

### Week 3 — micrograd and the autodiff mindset
**Do:** Karpathy Lecture 1 (micrograd), typing every line. Read `micrograd/engine.py` end to
end. Then Ch. 03 §3.1–3.3.
**Deliverable:** your own ~150-line scalar autograd engine that can compute a second
derivative (`grad(grad(f))`) and gets the right answer on `f(x) = sin(x)`.

### Week 4 — Environment, and the hinge exercise
**Read:** Ch. 04 in full. Set up the project exactly as in §4.1.
**Do:** run `00_check_environment.py` and `02_autodiff_playground.py`.
**Deliverable:** **the Burgers residual computed by autodiff at 4,096 random points, in
float64, with `gradgradcheck` passing** — and the MPS failure reproduced on your own machine,
so you never wonder about it again.

---

## Phase B — Practice (weeks 5–7, ~24 h)

### Week 5 — Training craft
**Read:** Ch. 05 §5.1–5.3 and §5.7. Karpathy's "Recipe."
**Do:** UvA Tutorials 3 and 4. Distill's momentum article.
**Deliverable:** a small experiment comparing Glorot vs Kaiming initialisation on a **tanh**
MLP, with a plot of the per-layer activation statistics. Explain in two sentences why the gain
factor matters.

### Week 6 — Spectral bias, made visible
**Read:** Ch. 05 §5.6. The Fourier Features paper and its project page.
**Do:** run `03_spectral_bias.py`; run the Tancik Colab.
**Deliverable:** a figure showing a plain MLP failing to fit `sin(x) + 0.3·sin(15x)` while a
Fourier-feature MLP succeeds, with the training curves for both. **Keep this figure — it
explains half of Chapter 09.**

### Week 7 — PDE primer and classical baselines
**Read:** Ch. 07 in full.
**Do:** all three exercises in §7.5.
**Deliverable:** three reference solutions you trust — a NumPy 1D heat solver (including
watching it blow up when you break the CFL condition), a scikit-fem 2D Poisson solution on an
L-shaped domain, and a py-pde Burgers reference at `ν = 0.01/π`. **Save them. Everything
downstream is measured against these.**

---

## Phase C — PINNs (weeks 8–11, ~32 h)

### Week 8 — The method
**Read:** Ch. 08 in full, twice. **Paper #1** (Raissi et al., JCP 2019), with the code open.
**Do:** Ben Moseley's workshop notebook. Then `04_pinn_oscillator.py`.
**Deliverable:** a damped-harmonic-oscillator PINN you wrote yourself, with **soft-IC and
hard-IC versions side by side**, and a plot showing the difference in convergence.

### Week 9 — Your first PDE
**Do:** `05_pinn_heat1d.py`, then DeepXDE's `Poisson_Lshape.py` (31 lines).
**Deliverable:** a three-panel figure — PINN prediction, scikit-fem reference, and **error** —
for 2D Poisson on the L-shape, with a relative L² number. **Point out the error concentration
near the re-entrant corner, and say why it is there.**

### Week 10 — Failure modes
**Read:** Ch. 09 in full. **Papers #2, #4, #6.**
**Do:** reproduce a failure on purpose. Take the 1D convection equation `u_t + β·u_x = 0` and
sweep β ∈ {1, 5, 10, 30, 50}. Plot the final relative error against β.
**Deliverable:** the failure curve, plus a one-paragraph explanation of *which* mechanism from
Ch. 09 you think dominates — and how you would test that.

### Week 11 — The recipe
**Read:** Ch. 10 in full. **Paper #3** (the Expert's Guide), with Table 1 open.
**Do:** `06_pinn_burgers.py` with `--plain`, then add one flag at a time.
**Deliverable:** **your own ablation table** — relative L² error for plain / +Fourier / +RWF /
+causal / +gradient-norm balancing / +L-BFGS on Burgers, over **three seeds each.** Compare
against the paper's 6.7e-4. This is the single most valuable artifact of the whole path.

---

## Phase D — Beyond (weeks 12–14, ~24 h)

### Week 12 — Inverse problems, where PINNs win
**Do:** `07_pinn_inverse.py`, then DeepXDE's `Lorenz_inverse.py` and
`diffusion_1d_inverse.py`.
**Deliverable:** recover a diffusion coefficient from **sparse data with 1% noise**, with a
plot of the recovered value against training step and the final relative error. Then repeat at
5% noise, and say where it breaks.

### Week 13 — Operator learning
**Read:** Ch. 13 §13.1–13.3 and §13.7.
**Do:** `pip install neuraloperator`, run `load_darcy_flow_small()`, then **train at 64×64 and
evaluate at 256×256.**
**Deliverable:** the zero-shot super-resolution result, plus a written comparison — in your own
words — of when you would reach for an FNO instead of a PINN.

### Week 14 — Judgment
**Read:** Ch. 14 in full. **Papers #9 and #12.** Skim PINNacle.
**Do:** pick a problem from your actual work and write the honest analysis.
**Deliverable:** a one-page memo answering, for that problem: is it forward or inverse? Does a
classical solver exist? How many times will you solve it? What is the baseline wall-clock time
at the accuracy you need? **And therefore: PINN, operator, classical solver, or hybrid — and
why.**
**If the answer is "classical solver," you have learned the most valuable thing on this path.**

---

## Optional Phase E — Reinforcement Learning

Read **Ch. 06 §6.10 first**, then pick a budget:

- **~10 h (recommended for most):** the Spinning Up intro + the PBDL RL chapter + the tokamak
  *Nature* paper. Deliverable: a two-paragraph note on when RL is, and is not, the right tool
  for a physics problem.
- **~35 h:** add tabular Q-learning, DQN, PPO, and one custom PDE Gymnasium environment.
  Deliverable: a PPO agent controlling boundary heating on your Week 7 heat solver.
- **~90 h:** the full order in Ch. 06 §6.9. Deliverable: the Burgers control problem solved
  both ways — model-free PPO and differentiable-physics gradient descent — with the sample
  counts compared.

---

## Milestones to check yourself against

| After | You should be able to |
|---|---|
| **Week 4** | Write the Burgers residual with autodiff from a blank file, and explain why float64 and CPU |
| **Week 7** | Produce a trusted reference solution for any 1D/2D problem you meet |
| **Week 9** | Train a PINN and honestly measure its error against a classical solution |
| **Week 11** | Make a PINN converge on a problem where the naive version fails — and say which fix did it |
| **Week 12** | Solve an inverse problem — the thing PINNs are actually best at |
| **Week 14** | Read a physics-ML paper and correctly predict, before the results section, whether the baseline is weak |

That last one is the real graduation.
