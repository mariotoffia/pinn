---
title: Reading List
subtitle: The 12 papers to read in full, in order — and how to follow the field
minutes: 12
---

# 15 — Reading List

Opinionated, and ordered so that each paper earns the next. **Total: roughly 25–30 hours of
real reading.**

**If you only have a weekend: read 1, 2, 3, 5 and 8.** That is the irreducible core.

Before you start, watch **[CS230 Autumn 2018, Lecture 8: "Career Advice / Reading Research
Papers"](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)** — Andrew
Ng's multi-pass method for reading papers costs 90 minutes and makes everything below faster.

---

## The 12

**1. Raissi, Perdikaris & Karniadakis — PINNs: A deep learning framework for solving forward
and inverse problems involving nonlinear PDEs** (JCP 2019) — **3 h (+2 h coding)**
https://doi.org/10.1016/j.jcp.2018.10.045 · arXiv [Part I](https://arxiv.org/abs/1711.10561) /
[Part II](https://arxiv.org/abs/1711.10566)
*The origin. Short, clear — and everything after it is a reaction to it. Read the Burgers
section with [the code](https://github.com/maziarraissi/PINNs) open beside you, and
**implement it yourself before moving on.***

**2. Krishnapriyan, Gholami, Zhe, Kirby & Mahoney — Characterizing possible failure modes in
PINNs** (NeurIPS 2021) — **2 h** · https://arxiv.org/abs/2109.01050
*Immediately after #1, before you believe anything. Proves the failures are optimisation, not
network capacity — on problems a first-year student could solve by hand. Gives you curriculum
training and time-marching, the two most reliable fixes. **Reading it second is the whole
pedagogical point.***

**3. Wang, Sankaran, Wang & Perdikaris — An Expert's Guide to Training PINNs** (2023) — **3 h**
https://arxiv.org/abs/2308.08468 · [JAX-PI](https://github.com/PredictiveIntelligenceLab/jaxpi)
*The operating manual. Non-dimensionalisation, Fourier features, RWF, causal weighting,
gradient-norm balancing, Adam→L-BFGS, random resampling — each component tested by switching
it off, with defaults. **Read Table 1 twice**, and note the 16–25% errors on the hard
problems.*

**4. Wang, Sankaran & Perdikaris — Respecting causality is all you need for training PINNs**
(2022) — **1.5 h** · https://arxiv.org/abs/2203.07404
*The sharpest single insight in the field: standard PINNs learn the future before the past.
Three lines of code, transformative on time-dependent problems. Short and beautifully argued.*

**5. Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang — Physics-informed machine learning**
(Nature Reviews Physics 2021) — **2 h** · https://www.nature.com/articles/s42254-021-00314-5
*The map. Observational vs inductive vs learning bias — the three-way taxonomy that lets you
place every variant and every alternative you will meet later. **Read it for structure, not
for claims.***

**6. Grossmann, Komorowska, Latz & Schönlieb — Can PINNs beat the Finite Element Method?**
(IMA J. Appl. Math. 2024) — **2 h** · https://arxiv.org/abs/2302.04107
*The reality check, with numbers: FEM is 1–3 orders of magnitude faster on 2D Poisson, 5–6 on
Allen–Cahn. **Read the Discussion section in full** — the authors are scrupulous about their
own limitations, and about the two places PINNs genuinely win (evaluation cost, and flat
scaling with dimension).*

**7. Wang, Yu & Perdikaris — When and why PINNs fail to train: An NTK perspective** (2020) — **2.5 h**
https://arxiv.org/abs/2007.14527
*The theory behind #3 and #4. It explains **why** the loss terms need balancing and **why**
high frequencies never get learned — as a statement about an eigenvalue spectrum. Skip the
theory and you will keep treating the fixes as folklore. Skim the proofs; absorb the kernel
decomposition.*

**8. Lagaris, Likas & Fotiadis — Artificial Neural Networks for Solving ODEs and PDEs** (IEEE TNN 1998)
— **1.5 h** · https://arxiv.org/abs/physics/9705023
*Twenty years early — and in one respect **better**: boundary conditions imposed exactly, by
construction. Reading it here, after you have fought with soft constraints, makes the
hard-constraint idea land properly — and inoculates you against the field's origin myth.*

**9. De Ryck & Mishra — Numerical analysis of PINNs and related models** (Acta Numerica 2024) — **4 h**
https://arxiv.org/abs/2402.10926
*What is actually **proven**. Approximation + generalisation + training error, unified. Their
conclusion — **training error is the bottleneck, and the least understood term** — is the
thesis of the modern field. Long; read the framework and the summary theorems, skip the
case-by-case proofs on a first pass.*

**10. Rathore, Lei, Frangella, Lu & Udell — Challenges in Training PINNs: A Loss Landscape
Perspective** (ICML 2024) — **2 h** · https://arxiv.org/abs/2402.01868
*Where the field turned: the problem is curvature, not architecture. Establishes Adam→L-BFGS
as the practical baseline, and opens the second-order line of work that dominates 2025–26.*

**11. Wang, Li, Chen & Perdikaris — PirateNets: Physics-informed Deep Learning with Residual
Adaptive Networks** (JMLR 2024) — **2.5 h** · https://arxiv.org/abs/2402.00326
*Solves a specific, maddening puzzle — **why deeper PINNs get worse** — with adaptive skip
connections plus a physics-informed final-layer initialisation. The current default
architecture, and a good model of how to debug an architecture properly.*

**12. Toscano et al. — From PINNs to PIKANs: Recent Advances in Physics-Informed Machine
Learning** (2024) — **3 h** · https://arxiv.org/abs/2410.13228
*The exit ramp. Where the field is now, what is live, what is speculative — **read it last**,
once you have the judgment to discount the enthusiasm. Pair it with a skim of
[Webb, Jerad & Cartis (2026)](https://arxiv.org/abs/2607.02194) for the current accuracy
frontier, and [the Feb 2026 loss-landscape paper](https://arxiv.org/abs/2602.05849) for a live
dissent on whether PINN loss landscapes are pathological at all.*

---

## If you want more than twelve

- **Scientific Machine Learning through PINNs: Where we are and What's next (Cuomo et al., 2022)** —
  https://arxiv.org/abs/2201.05624 — the most-cited general survey. Comprehensive, and by now
  appropriately dated.
- **Adaptive Physics-informed Neural Networks: A Survey (2025)** — https://arxiv.org/abs/2503.18181 —
  focused on adaptivity and re-optimisation — the axis that matters most for deployment.
- **PINNacle (NeurIPS 2024 D&B)** — https://arxiv.org/abs/2306.08827 — 20+ PDEs × 10+
  variants. **No variant dominates.** Skim it before believing any accuracy claim.
- **A Practitioner's Guide to KANs (2025)** — https://arxiv.org/html/2510.25781v1 — the best
  single verdict document on KANs for PDEs.
- **Weak baselines and reporting biases (McGreivy & Hakim, NMI 2024)** —
  https://www.nature.com/articles/s42256-024-00897-5 — arguably deserves the number #0 on the
  list above.

---

## Where to follow the field

**Journals — where the substance is published**

- **[Journal of Computational Physics](https://www.sciencedirect.com/journal/journal-of-computational-physics)**
  — the home venue: the original PINN paper, the NTK paper, SA-PINNs, B-PINNs.
- **[CMAME](https://www.sciencedirect.com/journal/computer-methods-in-applied-mechanics-and-engineering)**
  — hp-VPINNs, cPINN, RBA, hard constraints. A stronger engineering/FEM readership — which
  means harsher, and therefore more useful, refereeing of comparisons against classical
  methods.
- **[SIAM journals](https://epubs.siam.org/)** and **[Acta Numerica](https://www.cambridge.org/core/journals/acta-numerica)**
  — where the rigorous numerical analysis lives.
- **[Nature Machine Intelligence](https://www.nature.com/natmachintell/) / Nature Reviews
  Physics / Nature Computational Science** — high-visibility reviews and flagship results.
  **Read with Chapter 14 applied.**
- **[JMLR](https://jmlr.org/)** — PirateNets and the more ML-theoretic work.

**Conferences**

- **NeurIPS / ICML / ICLR** main tracks — increasingly where the optimisation-theoretic PINN
  work lands.
- **[ML4PS — Machine Learning and the Physical Sciences (NeurIPS workshop)](https://ml4physicalsciences.github.io/2025/)**
  — running annually since 2017; the best single snapshot of where physics-ML is each
  December.
- **AAAI-MLPS**, **SIAM CSE** (odd years), **SIAM UQ** (even years), **WCCM/ECCOMAS**,
  **USNCCM** — the applied-mechanics side, where PINNs are compared against FEM by people who
  use FEM daily.

**arXiv categories:** `math.NA` / `cs.NA` (theory and careful comparisons), `cs.LG`
(architectures and optimisers), `physics.comp-ph` (applications), `stat.ML` (Bayesian PINNs,
UQ). A standing query on `abs:"physics-informed"` across `math.NA` and `cs.LG` catches
essentially everything.

**Groups worth watching**

- **[CRUNCH, Brown (George Karniadakis)](https://sites.brown.edu/crunch-group/)** — originated
  PINNs, DeepONet, XPINN/cPINN, B-PINN, PIKAN. The highest output in the field, by a wide
  margin.
- **[Predictive Intelligence Lab, UPenn (Paris Perdikaris)](https://ai4science.seas.upenn.edu/publications)**
  — the training-pathology and best-practices line: gradient pathologies, NTK, causality, RWF,
  the Expert's Guide, PirateNets, JAX-PI. **If you read one group's back catalogue in order,
  read this one.** [Sifan Wang's page](https://sifanexisted.github.io/publications/) is the
  cleanest index.
- **[Lu Group, Yale (Lu Lu)](https://lugroup.yale.edu/)** — DeepXDE, DeepONet, hard-constraint
  PINNs.
- **[Anandkumar group, Caltech/NVIDIA](https://tensorlab.cms.caltech.edu/users/anima/)** —
  Fourier Neural Operators; the main intellectual competitor to the PINN framing.
- **[ETH Zürich — Mishra & De Ryck](https://camlab.ethz.ch/)** — rigorous error analysis.
  **The people most likely to tell you a claim is unproven.**
- **Berkeley — Mahoney & Krishnapriyan** — the failure-modes line and the
  optimisation-theoretic critique.
- **[Cambridge DAMTP — Schönlieb (CIA group)](https://www.damtp.cam.ac.uk/research/cia/)** —
  source of the FEM comparison; reliably skeptical.
- **NVIDIA** — PhysicsNeMo; the main force pushing physics-ML into industrial deployment.
- **[Polymathic AI](https://polymathic-ai.org/)** — MPP, The Well; the
  foundation-model-for-science line.
- **[TUM Physics-based Simulation (Nils Thuerey)](https://ge.in.tum.de/)** — PhiFlow, the PBDL
  book, solver-in-the-loop, APEBench.

**Trackers**

- **[awesome-pinns](https://github.com/AI-in-Transportation-Lab/awesome-pinns)** — **1,023
  papers, auto-updated daily from arXiv.** The best passive tracker for the field.
- **[awesome-scientific-machine-learning](https://github.com/MartinuzziFrancesco/awesome-scientific-machine-learning)**
  — broader SciML scope, human-curated.
- **[DeepXDE research page](https://deepxde.readthedocs.io/en/latest/user/research.html)** —
  hundreds of papers sorted by application. **Useful for checking whether anyone has already
  tried PINNs on *your* problem.**

---

## Two things to carry away

**First:** the PINN objective is a beautiful idea — turn a PDE into an infinite supply of free
labels, via autodiff — and it is a genuinely new capability for inverse problems, data
assimilation, awkward geometry and high dimension. **That part is not hype.**

**Second:** as a forward solver, on problems classical methods can handle, PINNs lose — usually
by orders of magnitude — and they lose *silently*. The field's own best practitioners publish
16–25% relative errors on hard benchmarks, and say so plainly in Table 1. **Anyone who tells
you PINNs replace numerical solvers either has not read the papers or is selling something.**

The interesting open question, as of August 2026, is whether the second-order optimisation wave
(papers #10 and #12, plus DSGNAR) actually moves that ceiling — a question worth **watching**
rather than betting on.
