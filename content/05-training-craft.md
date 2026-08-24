---
title: The Craft of Training
subtitle: Optimisers, initialisation, L-BFGS, loss balancing, spectral bias — the part that decides whether a PINN converges
minutes: 25
---

# 05 — The Craft of Training

A PINN is a small `tanh` MLP. Representing the solution is not the hard part — Krishnapriyan et
al. proved that by fitting the true solution with ordinary regression, easily. **PINNs fail at
optimisation.** This chapter teaches that optimisation craft: first in general machine-learning
form, then aimed at the PINN case.

**Target: about 20 hours.** This is the highest-value non-physics chapter on the path.

One term used throughout: a problem is **ill-conditioned** when its loss surface is like a long,
narrow valley — extremely steep in some directions and almost flat in others. Gradient descent
zig-zags across the steep walls and crawls along the flat floor. Most PINN training pain is
exactly this.

---

## 5.1 Optimisers

- **Why Momentum Really Works** — `interactive` `free` — https://distill.pub/2017/momentum/
  Gabriel Goh's Distill article. The best intuition-builder for *why* momentum fixes
  ill-conditioning, with a live slider showing the eigenvalue story. **This is the exact mental
  model you need when a PINN residual loss stalls: it is a conditioning problem, not a
  capacity problem.**
- **An overview of gradient descent optimization algorithms (Ruder)** — `blog+paper` `free` —
  https://www.ruder.io/optimizing-gradient-descent/ (https://arxiv.org/abs/1609.04747) — the
  fastest way to get SGD → Adagrad → RMSprop → Adam into one coherent picture, with the update
  equations side by side.
- **d2l.ai — Optimization chapter** — `book` `free` — https://d2l.ai/chapter_optimization/index.html
  A runnable-code treatment of convexity, momentum and Adam, plus a dedicated
  [learning-rate scheduling section](https://d2l.ai/chapter_optimization/lr-scheduler.html).
  Better than a blog post because you can execute every claim.
- **Goodfellow Ch. 8: Optimization for Training Deep Models** — `book` `free` —
  https://www.deeplearningbook.org/contents/optimization.html — the theory layer:
  ill-conditioned Hessians, plateaus, cliffs. Read it once, so that "ill-conditioned" means
  something specific to you.
- **AdamW — Decoupled Weight Decay Regularization** — `paper` `free` — https://arxiv.org/abs/1711.05101
  Why "L2 penalty in the loss" and "weight decay" are *not* the same thing under Adam. It
  matters here because **weight decay interacts badly with residual losses, which have no
  natural scale** — the Expert's Guide (Chapter 10) explicitly warns that it hurts accuracy on
  PINN forward problems.
- **SGDR: warm restarts / cosine annealing** — `paper` `free` — https://arxiv.org/abs/1608.03983
- **Optax optimizer API** — `docs` `free` — https://optax.readthedocs.io/en/latest/api/optimizers.html
  Worth reading even if you use PyTorch: its "chain of gradient transformations" design makes
  it obvious that Adam = (scale by RMS) ∘ (running average) ∘ (scale by learning rate).
- **Understanding Deep Learning — optimiser notebooks** — `notebook` `free` — Prince's
  fill-in-the-blank Colab notebooks let you *build* this section:
  [6.2 gradient descent](https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap06/6_2_Gradient_Descent.ipynb) →
  [6.3 SGD](https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap06/6_3_Stochastic_Gradient_Descent.ipynb) →
  [6.4 momentum](https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap06/6_4_Momentum.ipynb) →
  [6.5 Adam](https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap06/6_5_Adam.ipynb),
  plus [7.3 initialisation](https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap07/7_3_Initialization.ipynb)
  for §5.3 — watch the layer statistics explode or vanish as you change the gain.
  (CC BY-NC-ND: run, don't redistribute modified copies.)

---

## 5.2 L-BFGS — the PINN endgame optimiser

> Almost every PINN paper runs Adam for N steps, then switches to L-BFGS to push the residual
> down another 2–4 orders of magnitude. **Adam finds the right valley; L-BFGS uses curvature to
> race down it.** L-BFGS only works on *full-batch, deterministic, smooth* objectives — which is
> exactly what a PINN collocation loss is, and exactly what most deep learning is not.

- **Numerical Optimization: Understanding L-BFGS (Aria Haghighi)** — `blog` `free` —
  https://aria42.com/blog/2014/12/understanding-lbfgs — Newton → quasi-Newton → BFGS →
  limited-memory BFGS, with the two-loop recursion written out. **The best free explanation on
  the internet. Read it before you touch the API.**
- **`torch.optim.LBFGS`** — `docs` — https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html

**Pitfalls, in the order they usually bite:**

1. **It requires a `closure()`** — a function that re-evaluates the model and returns the loss —
   unlike every other PyTorch optimiser. Forgetting `optimizer.zero_grad()` inside the closure
   silently accumulates gradients across the line search.
2. **`line_search_fn` defaults to `None`** — a fixed step size, which often diverges on PINN
   problems. **You almost always want `line_search_fn="strong_wolfe"`.** This single argument
   is the most common fix for "my L-BFGS made things worse."
3. **It is a full-batch method.** Resampling collocation points between L-BFGS steps poisons
   its curvature estimate. **Freeze your collocation set during the L-BFGS phase.**
4. **It needs float64.** In float32 the curvature pairs are numerical garbage and the method
   stalls — this is the FP64 finding from Chapter 04, restated.
5. **`max_iter` counts *inner* iterations per `.step()` call**, not epochs. This trips everyone
   exactly once.

**Alternatives, verified August 2026:**

| Tool | Status | Verdict |
|---|---|---|
| `scipy.optimize.minimize(method="L-BFGS-B")` | Rock solid | **Most reliable.** Flatten weights ↔ float64 NumPy vector. Costs a host copy per iteration; fine for small PINNs |
| [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) | v0.1.0, Jan 2026, active | SciPy-shaped API with **real autograd derivatives**: BFGS, L-BFGS, CG, Newton-CG, Newton-Exact, Trust-NCG, Dogleg. The best PyTorch answer |
| [Optax](https://optax.readthedocs.io/en/stable/_collections/examples/lbfgs.html) | v0.2.8, active | **Where JAX's L-BFGS now lives.** `optax.lbfgs()` + `optax.scale_by_zoom_linesearch()` |
| [Optimistix](https://docs.kidger.site/optimistix/) | v0.1.0, active | JAX nonlinear least-squares / root-finding. Gauss–Newton and Levenberg–Marquardt fit residual minimisation better than plain Adam |
| [jaxopt](https://github.com/google/jaxopt) | **DEPRECATED** | README: *"JAXopt is no longer maintained nor developed."* Its L-BFGS moved into Optax. **Still widely recommended online — do not start here** |
| [hjmshi/PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS) | Stale since Feb 2023 | The best-designed variant (Armijo/weak-Wolfe line searches, Powell damping). Read for the ideas; do not depend on it |

**Start with** `torch.optim.LBFGS(..., line_search_fn="strong_wolfe")` in float64 on the CPU. If
it misbehaves, move to the SciPy wrapper.

Read DeepXDE's `model.compile("L-BFGS")` code path to see how a mature library sequences
Adam → L-BFGS, including closure handling and stopping rules: https://github.com/lululxvi/deepxde

And run the Adam-vs-L-BFGS story in one hosted notebook: Purdue's ME 539 (a live course,
running Fall 2026) compares the two optimisers on a physics-informed loss —
[open in Colab](https://colab.research.google.com/github/PredictiveScienceLab/data-analytics-se/blob/master/lecturebook/lecture26/hands-on-26.1.ipynb)
(GPL-3.0 · course book: https://predictivesciencelab.github.io/data-analytics-se/).

---

## 5.3 Initialisation, activations, normalisation

"Initialisation" means how the weights are randomly set before training starts. It matters far
more than beginners expect — set the scale wrong and signals shrink or explode layer by layer.

- **Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio)** —
  `paper` `free` — https://proceedings.mlr.press/v9/glorot10a.html — the Xavier/Glorot paper.
  It was **derived for tanh and sigmoid** — which is precisely why it is the right default for
  PINNs.
- **Delving Deep into Rectifiers (He et al.)** — `paper` `free` — https://arxiv.org/abs/1502.01852 —
  He initialisation, derived for ReLU. Read it to understand why blindly using PyTorch's
  ReLU-tuned default (`kaiming_uniform_`) on a **tanh** PINN quietly shrinks the activations
  layer by layer. Note: `torch.nn.init.calculate_gain('tanh')` = 5/3.
- **d2l — Numerical Stability and Initialization** — `book` `free` —
  https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html
- **UvA Tutorial 4: Optimization and Initialization** — `interactive` `free` —
  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html
  Xavier and He derived and then tested in runnable notebooks, plus an optimiser comparison.
  **PINN training failures are very often initialisation and conditioning failures.**
  One-click: [Tutorial 4 in Colab](https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial4/Optimization_and_Initialization.ipynb) ·
  [Tutorial 3 (activations)](https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial3/Activation_Functions.ipynb).
- **Karpathy Zero to Hero, Lecture 3** — https://youtu.be/P6sfmUTpUmc — 1h55m of live coding in
  which he *shows* you dead tanh units, saturated activations, and the exact effect of the
  initialisation gain on forward and backward statistics. **If you watch one video in this
  chapter, watch this one.**
- **Batch Normalization** (https://arxiv.org/abs/1502.03167) and **How Does Batch Normalization
  Help Optimization?** (https://arxiv.org/abs/1805.11604) — Santurkar et al. demolish the
  original "internal covariate shift" story and show BatchNorm's real effect is smoothing the
  loss landscape. Read both; the second is the real explanation.
- **Layer Normalization** — https://arxiv.org/abs/1607.06450 — the batch-independent
  alternative, and what you would reach for if you *had* to normalise inside a PINN.

### Why PINNs avoid BatchNorm

There is no single canonical citation; here is the reasoning:

1. A PINN's "batch" is a set of collocation points sampled from the domain. BatchNorm would
   make the network's output at a point `x` depend on which *other* points happened to be in
   the same batch — breaking the pointwise function that the residual assumes.
2. BatchNorm behaves differently in train and eval modes (batch statistics vs running
   statistics). So **the function you differentiate during training is not the function you
   evaluate afterwards** — poison for a method whose entire loss is built from derivatives of
   that function.
3. Higher-order autodiff through batch statistics is expensive and numerically messy.
4. PINNs are small, full-batch tanh MLPs, where BatchNorm's optimisation benefits are marginal
   anyway.

The standard PINN stabilisers are instead: **scale the inputs into [-1, 1]**
(non-dimensionalisation), **Glorot initialisation**, and **weight normalisation / random weight
factorisation** — see Chapter 10.

---

## 5.4 Loss landscapes and gradient pathologies

- **Visualizing the Loss Landscape of Neural Nets** — `paper+repo` `free` —
  https://arxiv.org/abs/1712.09913 · https://github.com/tomgoldstein/loss-landscape —
  filter-normalised 2D slices of the loss surface. The technique transfers directly: plotting
  your PINN's loss along random directions is a useful diagnostic. Companion visual
  essay: https://losslandscape.com/
- **Understanding and Mitigating Gradient Flow Pathologies in PINNs (Wang, Teng, Perdikaris, 2020)** —
  `paper` `free` — https://arxiv.org/abs/2001.04536 · SIAM J. Sci. Comput. 43(5) ·
  code https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs
  **The central paper of this chapter for a PINN person.** It diagnoses a **stiffness in the
  gradient flow** of the combined loss: the gradient sizes `‖∇θ L_r‖` and `‖∇θ L_bc‖` can
  differ by orders of magnitude, so gradient descent effectively ignores one of the
  objectives. The fix proposed — **learning-rate annealing** — rebalances the loss weights
  every `f` steps using gradient statistics. Everything downstream cites this paper.
- **When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective (Wang, Yu, Perdikaris)** —
  `paper` `free` — https://arxiv.org/abs/2007.14527 · JCP 449:110768 — the theoretical
  companion. In the infinite-width limit, PINN training is governed by a **PINN NTK** (neural
  tangent kernel — a matrix describing how fast each direction gets learned) that splits into
  boundary and residual blocks whose eigenvalues differ enormously — hence one part of the
  problem converges while the other barely moves. Leads to NTK-based adaptive weighting.
- **Challenges in Training PINNs: A Loss Landscape Perspective (Rathore et al., ICML 2024)** —
  `paper` `free` — https://arxiv.org/abs/2402.01868 — traces the difficulty to
  **ill-conditioning caused by the differential operator inside the residual**, shows
  empirically that **Adam → L-BFGS beats either alone**, and introduces NysNewton-CG. This is
  the paper that made "the problem is second-order curvature, not architecture" the consensus
  view.
- **Visualizing the loss landscapes of physics-informed neural networks (Feb 2026)** — `paper` `free` —
  https://arxiv.org/abs/2602.05849 — **a live dissenting view, worth knowing about.** Applying
  loss-landscape tools to PINNs, the authors find landscapes that look *smooth,
  well-conditioned and convex near the solution*, "challenging prevailing intuitions." Hold
  this in tension with Krishnapriyan and Rathore. The field has not fully settled what
  "ill-conditioned" means here.

---

## 5.5 Balancing several losses at once — the PINN problem in general-ML clothing

Your loss is `λ_r·L_r + λ_bc·L_bc + λ_ic·L_ic + λ_d·L_d` — physics, boundary, initial condition
and data terms, each with a weight λ. Choosing those λ's decides whether you get the answer or a
convincing-looking artifact. The general multi-task-learning literature got here first:

- **Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall, Gal, Cipolla)** — `paper` `free` —
  https://arxiv.org/abs/1705.07115 — learns one weight per task as a noise level:
  `L = Σ (1/2σᵢ²)Lᵢ + log σᵢ`. One learnable number per loss term. Cheap, and often the first
  thing to try.
- **GradNorm** — `paper` `free` — https://arxiv.org/abs/1711.02257 — balances the weights so
  each task's *gradient size* at a shared layer is comparable. Conceptually the ancestor of
  Wang & Perdikaris's learning-rate annealing.
- **Gradient Surgery for Multi-Task Learning (PCGrad)** — `paper+repo` `free` —
  https://arxiv.org/abs/2001.06782 · https://github.com/WeiChengTseng/Pytorch-PCGrad — detects
  *conflicting gradients* (negative cosine similarity) and projects one onto the plane
  perpendicular to the other. Directly useful when boundary and residual gradients pull in
  opposite directions.
- **Multi-Task Learning as Multi-Objective Optimization (MGDA)** — `paper` `free` —
  https://arxiv.org/abs/1810.04650 — the principled version: find a Pareto-stationary point
  with multiple-gradient descent.
- **In Defense of the Unitary Scalarization for Deep Multi-Task Learning** — `paper` `free` —
  https://arxiv.org/abs/2201.04122 — **the deliberate skeptic's counterweight.** Kurin et al.
  (NeurIPS 2022) show that a plain fixed-weight sum, properly regularised and tuned, matches or
  beats most specialised multi-task optimisers. **Takeaway for PINNs: try a well-tuned static
  weighting and a proper learning-rate sweep before reaching for GradNorm or PCGrad.** Many
  reported adaptive-weighting wins are tuning wins in disguise.

Chapter 10 gives the PINN-specific schemes the field actually settled on (gradient-norm
balancing, NTK balancing, SA-PINN, RBA).

---

## 5.6 Spectral bias and the NTK — why your PINN converged to a smooth blob

> Networks learn **low frequencies first** — the broad, smooth shape of a function before its
> fine wiggles. A PDE solution with sharp gradients or many scales is therefore the *last* thing
> your network learns. And because your loss is built from derivatives of the network, the
> missing high-frequency part dominates the residual.

- **On the Spectral Bias of Neural Networks (Rahaman et al., ICML 2019)** — `paper` `free` —
  https://arxiv.org/abs/1806.08734 — the Fourier-analysis-of-training result, with clean
  synthetic experiments. Start here.
- **Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks (Xu et al.)** —
  `paper` `free` — https://arxiv.org/abs/1901.06523 (overview: https://arxiv.org/abs/2201.07395)
  The independent, more quantitative statement of the same finding. **The critical observation
  for PINNs: this is the exact opposite of classical iterative solvers.** Jacobi and
  Gauss–Seidel kill *high*-frequency error fastest — that is what makes multigrid work.
  Networks kill *low* frequencies fastest. So a century of intuition about iterative PDE
  solvers flips upside down, and multiscale problems — which classical methods handle by
  hierarchy — become the worst case for PINNs.
- **3Blue1Brown — But what is a Fourier series?** — `video` `free` —
  https://www.youtube.com/watch?v=r6sGWTCMz2k — if "frequencies of a function" feels abstract,
  this 25-minute video makes it concrete. Watch it before the two papers above.
- **Overview: Frequency Principle / Spectral Bias in Deep Learning** — `survey` `free` —
  https://link.springer.com/article/10.1007/s42967-024-00398-7 — collects the theory,
  experiments and remedies (multiscale DNNs, PhaseDNN) in one readable document.
- **Neural Tangent Kernel (Jacot, Gabriel, Hongler)** — `paper` `free` — https://arxiv.org/abs/1806.07572
  — the original NTK paper. Dense; read an explainer first:
  - **Some Math Behind Neural Tangent Kernel (Lil'Log)** — https://lilianweng.github.io/posts/2022-09-08-ntk/
    — every step shown; the best free NTK derivation for someone comfortable with linear
    algebra.
  - **Understanding the Neural Tangent Kernel (EigenTales)** — https://www.eigentales.com/NTK/ —
    shorter and more geometric. Read this one first, then Lil'Log.
- **PyTorch NTK tutorial** — https://docs.pytorch.org/tutorials/intermediate/neural_tangent_kernels.html
  — computes an empirical NTK with `jacrev` + `vmap`. Useful practice for the transforms *and*
  the theory.
- **Fourier Features (Tancik et al.)** — https://arxiv.org/abs/2006.10739 — **the fix.**
- **On the eigenvector bias of Fourier feature networks (Wang, Wang, Perdikaris)** — `paper` `free` —
  https://arxiv.org/abs/2012.10047 · CMAME 384:113938 — connects spectral bias to the NTK
  eigenvalues and shows that random Fourier features shift the network's "preferred"
  frequencies to whatever band you choose via σ. **The theory behind the single most useful
  architectural fix in Chapter 10.**

---

## 5.7 Debugging discipline

- **A Recipe for Training Neural Networks (Karpathy)** — `blog` `free` —
  https://karpathy.github.io/2019/04/25/recipe/ — know your data → build an end-to-end skeleton
  with dumb baselines → **overfit one batch** → regularise → tune → squeeze. Its core thesis —
  **"neural net training fails silently"** — is the single most important sentence for a PINN
  practitioner, whose loss can reach 1e-6 while the solution is completely wrong.
- **Troubleshooting Deep Neural Networks (Josh Tobin)** — `guide` `free` —
  http://josh-tobin.com/troubleshooting-deep-neural-networks.html — a literal decision tree,
  with a bias/variance breakdown that tells you *which* knob to turn. Karpathy gives the
  philosophy; Tobin gives the flowchart. Video version: https://fullstackdeeplearning.com/spring2021/lecture-7/
- **CS231n — Learning and Evaluation** — `notes` `free` — https://cs231n.github.io/neural-networks-3/
  — gradient checking, sanity checks, babysitting training, update-to-weight ratios. **The
  gradient-check section is directly reusable for verifying your PDE residual's autodiff.**

---

## 5.8 Hyperparameter tuning

- **Deep Learning Tuning Playbook (Google Research)** — `guide` `free` —
  https://github.com/google-research/tuning_playbook — ~30k stars. How to choose model,
  optimiser and batch size; the explore-then-exploit round structure; scientific vs nuisance vs
  fixed hyperparameters; quasi-random search over grid search; deciding how long to train.
  **The most useful single document here after Karpathy's recipe.** PINN papers are notorious
  for under-tuned baselines; this is how you avoid being that person.
- **Optuna** — `library` `free` — https://optuna.org/ · https://arxiv.org/abs/1907.10902 —
  define-by-run search spaces, the TPE sampler, and (underused) **pruners** that kill
  unpromising trials early. For PINNs this is the pragmatic choice: a small search space
  (width, depth, learning rate, loss weights, collocation count) on one machine.
- **Ray Tune** — `library` `free` — https://docs.ray.io/en/latest/tune/index.html — for when you
  outgrow one machine, or want ASHA / Population-Based Training. Can use Optuna as its backend.

---

## 5.9 Generalisation when your loss is a PDE residual

There is no single canonical reference for this, so here is the argument to internalise:

1. **You have no test set in the usual sense.** Your "training data" is a set of collocation
   points *you* chose, from a domain you fully control; more points are free. There is no
   underlying data distribution to generalise *from*, and no label noise to overfit *to*.
   Classical statistical learning theory is mostly the wrong lens.
2. **What replaces it is a PDE-theoretic error bound.** The real question is: *does a small
   residual imply a small solution error?* That is a **stability** property of the differential
   operator, not a statistical one. For well-posed problems there is a bound of the form
   `‖u − û‖ ≤ C·‖residual‖`, where the constant `C` comes from the operator's stability. For
   stiff or ill-conditioned operators, `C` is huge — and a tiny residual buys you nothing.
3. **The real failure mode is therefore "low loss, wrong answer"** — not "low train loss, high
   test loss." The remedies are different too: more and better-placed collocation points where
   the residual is large (adaptive sampling), and checking against a conventional solver or a
   conserved quantity — not dropout and early stopping.
4. **Standard regularisers usually hurt.** Dropout makes the network non-deterministic, which
   corrupts the very derivatives you are differentiating. Heavy weight decay biases the network
   toward the trivial zero solution.
5. Supporting reading: [Characterizing Possible Failure Modes in PINNs](https://arxiv.org/abs/2109.01050)
   for the "it is optimisation, not generalisation" evidence, and the De Ryck / Mishra /
   Molinaro line of work for the actual residual→error bounds
   (e.g. https://arxiv.org/abs/2006.16144).

For context on the classical generalisation phenomena you will still meet in conversation:
[Deep Double Descent](https://arxiv.org/abs/1912.02292) ·
[readable write-up](https://windowsontheory.org/2019/12/05/deep-double-descent/) ·
[Belkin et al. on the interpolation threshold](https://arxiv.org/abs/1812.11118).
