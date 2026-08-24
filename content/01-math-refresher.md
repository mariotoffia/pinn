---
title: Math Refresher
subtitle: The minimum that matters, and what to skip
minutes: 17
---

# 01 — Math Refresher

You studied this once. You do not need to re-derive Gaussian elimination. You need three things:

1. **A geometric feel for linear algebra** — seeing a matrix as a transformation of space, and
   knowing what eigenvectors and the SVD (singular value decomposition) mean in pictures.
2. **Real comfort with derivatives of functions of several variables** — gradients, Jacobians,
   Hessians, and the multivariable chain rule. A PINN is, at its core, "differentiate the
   network's output with respect to its inputs, and put the result into a PDE."
3. **Basic intuition for differential equations** — what such an equation *is* as an object, and
   what "the heat equation" describes physically.

Everything else is optional. **Target: about 17 hours.** The CUT LIST at the bottom is as
important as the list above it.

Quick vocabulary, used all through this path: the **gradient** is the vector of a function's
partial derivatives (which direction is uphill). The **Jacobian** is the matrix of all first
derivatives of a vector-valued function. The **Hessian** is the matrix of all second derivatives.
If those words feel rusty, this chapter is exactly for you.

---

## 1.0 If your single-variable calculus is rusty too

- **3Blue1Brown — Essence of Calculus** — `video` `free` — https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
  Twelve short videos that rebuild what a derivative *is*, visually, including the chain rule —
  the single rule that all of deep learning runs on. If you can already differentiate
  `sin(x²)` without thinking, skip this section. If not, spend the evening here first. **~3 h.**

---

## 1.1 Linear algebra

- **3Blue1Brown — Essence of Linear Algebra** — `video` `free` — https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
  Sixteen short videos that rebuild matrices and eigenvectors as *geometric transformations*.
  The single highest-value math item on this whole path. **~4 h.**
  **Watch:** the videos on matrices as transformations and composition of transformations
  (episodes 3–4), dot products (9), change of basis (13), and eigenvectors and eigenvalues (14).
  **Skip:** the determinant (6) and Cramer's rule (12) episodes — they matter very little for
  PINNs.
  Text version, good for revisiting: https://www.3blue1brown.com/topics/linear-algebra

- **A 2020 Vision of Linear Algebra (Gilbert Strang, MIT)** — `video` `free` — https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/
  Six short lectures that re-teach the whole subject through matrix factorisations: `A=CR`,
  `A=LU`, `A=QR`, `A=QΛQᵀ`, `A=UΣVᵀ`. If you only have two hours for linear algebra, spend them
  here, after 3Blue1Brown. **~2 h.**

- **MIT 18.06SC Linear Algebra (Strang, OCW Scholar)** — `course` `free` — https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/
  The full self-study course — lectures *plus* problem-solving videos, problem sets with
  solutions, and summary notes. **Use it as a reference to look things up, not as a course to
  sit through.** If you do want a targeted path: skim Unit I (lectures 1–10), then lectures
  21–22 and 25 (eigenvalues, diagonalisation), 28 (positive definite matrices), 29 (SVD).
  **Skip:** determinants (18–20), the Markov/Fourier applications, complex matrices and the FFT,
  and linear programming.

---

## 1.2 Multivariable calculus — the part that carries the most weight

This is the section that pays for itself. A PINN residual is the chain rule, applied twice.

- **The Matrix Calculus You Need For Deep Learning (Parr & Howard)** — `reference` `free` —
  https://explained.ai/matrix-calculus/ (arXiv: https://arxiv.org/abs/1802.01528)
  The one document that closes the gap between "I remember partial derivatives" and "I can read
  ∂L/∂W in a paper." It builds up Jacobians, element-wise operations, and — the part that
  matters most — **the vector chain rule**, then applies it to a neuron and a full loss
  function. **~4 h** to read properly.
  **Do not skip §4.5 (the chain rules) or §6.** §8 is a standalone reference table — bookmark
  it and keep it open forever.

- **Khan Academy — Multivariable Calculus** — `course` `free` — https://www.khanacademy.org/math/multivariable-calculus
  The videos on gradients, Jacobians and Hessians here were made by Grant Sanderson (the
  3Blue1Brown author), and they are the clearest free explanations of exactly the objects PINNs
  use. **~2 h, targeted.**
  **Watch, in order:** partial derivatives → gradient → directional derivative → **Jacobian** →
  quadratic approximation → **Hessian**.
  **Skip the entire integration half** — line integrals, surface integrals, Green's, Stokes'
  and the divergence theorem. You need those to *derive* physical laws; you do not need them to
  build or read a PINN, which uses the PDE in its pointwise ("strong") form. Come back for the
  divergence theorem only if you later explore variational methods like Deep Ritz.

- **Mathematics for Machine Learning (Deisenroth, Faisal, Ong)** — `book` `free PDF` — https://mml-book.github.io/
  Your *reference book*, not your bedtime reading. **Read Ch. 5 (Vector Calculus — gradients,
  Jacobians, backprop, automatic differentiation, Hessians) and Ch. 7 (Continuous
  Optimization).** Skim Ch. 2–4 as a lookup. Skip Ch. 8–12 (classical machine learning).
  **~2 h, targeted.**

- **Mathematics for Machine Learning Specialization (Imperial College)** — `course` `audit free / paid cert` —
  https://www.coursera.org/specializations/mathematics-machine-learning
  **Only the Multivariate Calculus course is worth your time** (~18 h) — it builds
  Jacobian/Hessian fluency with code. The Linear Algebra course is slower than 3Blue1Brown for
  less payoff, and the PCA course is off this path entirely.

---

## 1.3 Probability — a small dose

PINNs are, almost everywhere, a deterministic least-squares method. There is no likelihood, no
posterior, no sampling. Take the two-hour version now. Take a full course only if you later head
toward Bayesian PINNs.

- **Seeing Theory (Brown University)** — `interactive` `free` — https://seeing-theory.brown.edu/
  Six chapters of interactive visualisations. The site is archived but fully working, and it is
  the fastest way to reload probability intuition. **~2 h.**
- **Harvard Stat 110 (Blitzstein)** — `course` `free` — https://stat110.hsites.harvard.edu/
  (free book: http://probabilitybook.net/ — the link opens the free PDF). The best free
  probability course there is. **Bookmark it for the day you want uncertainty estimates on
  physics models.** About 35 hours if you ever commit to it.
- **Probabilistic Machine Learning: An Introduction (Murphy)** — `book` `free draft` —
  https://probml.github.io/pml-book/book1.html — Ch. 2–4 as a reference. Has a Colab notebook
  for every chapter.

---

## 1.4 A first taste of differential equations

- **3Blue1Brown — Differential Equations** — `video` `free` — https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6
  Exactly the right dose. **Watch: the opening overview, the "what is a partial differential
  equation?" and heat-equation videos, and the Fourier-series video.** The heat-equation videos
  build the mental picture a PINN approximates: a smooth field that obeys a local rule at every
  point. The Fourier-series video ([direct link](https://www.youtube.com/watch?v=r6sGWTCMz2k))
  is the intuition behind **spectral bias** — the single most common reason PINNs fail, which
  you will meet properly in Chapter 05. **~2.5 h.**

- **Learn Differential Equations: Up Close with Gilbert Strang and Cleve Moler (MIT RES.18-009)** —
  `course` `free` — https://ocw.mit.edu/courses/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/
  Short, focused videos on first- and second-order ODEs, graphical and numerical methods, plus
  Moler on MATLAB's ODE solvers. **The numerical-methods segments are the ones that matter** —
  they show you the classical solvers that every PINN paper is silently being compared against.
  **~4 h, targeted.** Skip the Laplace transforms and the vector-spaces unit.

Chapter **07 — PDE Primer** goes further and gives you runnable *code* baselines. This section
is just for intuition.

---

## 1.5 Suggested order for this chapter

| # | Item | Hours |
|---|---|---:|
| 0 | 3B1B Essence of Calculus — only if derivatives feel rusty | (3) |
| 1 | 3B1B Essence of Linear Algebra (selected episodes) | 3 |
| 2 | Strang, A 2020 Vision of Linear Algebra (all six) | 2 |
| 3 | Khan Academy multivariable — the derivatives half only | 2 |
| 4 | Parr & Howard, Matrix Calculus (§1–6 properly; keep §8 open forever) | 4 |
| 5 | Seeing Theory, all six chapters | 2 |
| 6 | 3B1B Differential Equations (selected videos, incl. Fourier series) | 2.5 |
| 7 | MML book Ch. 5 + Ch. 7 as a wrap-up skim | 1 |
| | **Total** | **~16.5 (+3)** |

---

## 1.6 Do the math in a notebook

Two places to *run* this chapter instead of only reading it — both open in Colab in one click:

- **Mathematics for ML — companion tutorials** — `notebook` `free` — the MML book ships three
  exercise notebooks, each with a worked `.solution` twin:
  [linear regression](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_linear_regression.ipynb)
  (vector calculus in action),
  [PCA](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_pca.ipynb)
  (the linear algebra), and
  [Gaussian mixtures](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_gmm.ipynb)
  (the probability dose). Cambridge University Press copyright: run and learn in place — do not
  copy them into your own material.
- **Mathematical Python (Patrick Walls, UBC)** — `notebook course` `free` —
  https://patrickwalls.github.io/mathematicalpython/ — novice-level notebooks that
  compute [Riemann sums](https://colab.research.google.com/github/patrickwalls/mathematicalpython/blob/master/notebooks/integration/riemann-sums.ipynb),
  derivatives and linear algebra numerically in NumPy — compute a derivative *yourself* before
  Chapter 03 lets autodiff do it for you. (CC BY-NC-SA: link and run.)

---

## CUT LIST — famous, and deliberately skipped

- **The full MIT 18.06 lecture series (35 h).** Strang is a joy, but sitting through Gaussian
  elimination and determinant expansions to reach the SVD at lecture 29 is a poor use of your
  evenings. Use 3Blue1Brown plus the six factorisation lectures; dip into 18.06SC only for
  eigenvalues and the SVD.
- **Multivariable integration (line/surface integrals, Green's, Stokes', divergence).**
  Beautiful, and needed to *derive* PDEs from physics. Not needed to build or read a
  PINN.
- **Stat 110 in full (35 h).** World-class teaching for a problem you do not have yet.
- **The Coursera Linear Algebra and PCA courses.** Slower than 3Blue1Brown for less payoff, and
  PCA is a classical-ML detour. Take only Multivariate Calculus, and only if Parr & Howard did
  not click.
- **Tübingen "Mathematics for Machine Learning" (von Luxburg, 89 lectures)** —
  https://www.youtube.com/playlist?list=PL05umP7R6ij1a6KdEy8PVE9zoCv6SlHRS — excellent and
  rigorous, but it is a *reference library*, not a study plan. Dip in when one specific topic
  will not click.
- **Murphy's *Advanced Topics* volume.** Diffusion models and MCMC are not on the road to a
  PINN. Shelve it.
