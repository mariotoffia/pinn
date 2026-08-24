---
title: Start Here
subtitle: What this path is, who it is for, and an honest picture of the field
minutes: 15
---

# Start Here

This is a complete, self-contained learning path. It takes you from **"I can program, and I once
took a linear algebra course"** to **"I can build, train, debug, and honestly judge a
Physics-Informed Neural Network (PINN)."**

A quick note on the name: a **PINN** is a neural network that is trained to obey a physical law
(a differential equation) instead of being trained on a big dataset. That is the whole subject of
this path. Do not worry if that sentence does not fully make sense yet — it will.

The path was written *from* an **Apple Silicon MacBook Pro**, running everything **on the
CPU**, with **Google Colab or Kaggle** for the rare job that needs a GPU. But it is not
Mac-only. The core configuration — **CPU, float64** — exists on every computer, and the free
GPU lane (Colab/Kaggle) runs in a browser. **On Windows or Linux, with or without an NVIDIA
GPU, everything essential works the same** — Chapter 04 §4.9 gives the three small
substitutions.

Everything is free unless it is marked `[paid]`. Every link was checked live in **August 2026**.

---

## The whole field in one paragraph

A PINN uses a small, smooth neural network `u_θ(x,t)` as a stand-in for the solution of a
**partial differential equation (PDE)** — an equation that links a quantity (like temperature) to
its own rates of change in space and time. The network's derivatives are computed exactly, using
**automatic differentiation** (the same machinery behind normal deep learning). Training then
pushes the equation's error — the **residual** — toward zero at randomly chosen points. In other
words: the equation itself becomes an endless source of free training labels. That idea is
genuinely elegant. For **inverse problems** (finding unknown physical constants from
measurements), **data assimilation** (filling gaps between sparse sensors), **awkward
geometries**, and **high-dimensional PDEs**, it is a real ability that classical solvers do not
offer cheaply. But as a *forward solver* — just solving a known equation that a classical method
like the finite element method can already handle — PINNs lose, usually by a wide margin. Worse,
they fail **silently**: the loss goes down, the plot looks smooth, and the answer is still wrong.
So learn PINNs for what they are good at, and learn their failure modes at the same time as the
method.

If you remember nothing else from this path, remember that paragraph. It will save you months.

---

## Four hard facts about your hardware

These are facts, not opinions. They were checked against the PyTorch 2.13 source code and live
issue trackers.

1. **PyTorch's MPS backend (the Apple GPU) cannot compute in `float64`.** `float64` means
   64-bit "double precision" numbers. Apple's GPU language (Metal) simply has no `double` type —
   this is a hardware limit, not a missing feature. The shipped file
   `torch/_inductor/codegen/mps.py` literally contains
   `raise RuntimeError("float64 is not supported by MPS")`.
2. **MPS also cannot take a second derivative through `nn.Linear`.** You get
   `RuntimeError: derivative for aten::linear_backward is not implemented`
   ([pytorch#98498](https://github.com/pytorch/pytorch/issues/98498), open since April 2023).
   A PINN is *built* on that exact operation.
3. **PINNs need float64.** The paper [FP64 is All You Need (NeurIPS 2025)](https://arxiv.org/abs/2505.10949)
   shows that several famous PINN "failure modes" were never real optimisation dead ends. They
   were 32-bit rounding limits: the optimiser (L-BFGS) hit its stopping rule and quit early,
   while the answer was still wrong.
4. **So on your Mac, PINNs run on the CPU, in float64.** This is not a sad compromise. It is the
   only setup on this machine that computes the right answer. And you lose very little speed:
   PINN networks are tiny (3–6 layers of 128–512 units) with small batches. That is exactly the
   kind of work where a GPU spends more time on launch overhead than on math.

```python
# The two lines at the top of every PyTorch entry point on this path
import torch
torch.set_default_dtype(torch.float64)   # BEFORE constructing any module
DEVICE = torch.device("cpu")             # deliberate: MPS lacks f64 + double-backward
```

Run `starter/scripts/00_check_environment.py` on day one. It proves all four facts on your own
machine in about ten seconds.

**Not on a Mac?** Facts 1 and 2 are about Apple's GPU — but the conclusion is universal: fact 3
(PINNs need float64) applies to you too, and almost no consumer GPU does float64 well. A Windows
PC without an NVIDIA card lands in exactly the same place as the Mac: **CPU, float64, and free
cloud GPUs for the rare heavy job.** You lose nothing on this path. Details in Chapter 04 §4.9.

---

## How this path is organised

| Part | Chapters | What you get |
|---|---|---|
| **I — Foundations** | Math Refresher, Neural Networks, Autodiff | Backprop derived by hand; automatic differentiation as a core idea, not a black box |
| **II — Practice** | Environment, Training Craft, Reinforcement Learning | A working local setup, plus the optimisation skills that decide whether a PINN converges |
| **III — Physics** | PDE Primer, PINN Core, Failure Modes, The Recipe | The method, stated precisely — and where it breaks |
| **IV — Tooling** | Frameworks, Worked Examples | What to install, what to run, what to avoid |
| **V — Beyond** | Neural Operators, Reality Check, Reading List, Roadmap | What comes after PINNs, and an honest map of the field |

Chapter **03 — Automatic Differentiation** is the hinge of the whole path. Everything before it
is standard deep learning. Everything after it depends on being comfortable with
`grad(grad(u))` — taking a derivative of a derivative. Do not skim it.

**Guided tours.** Five chapters (02, 03, 05, 08, 09) carry a **▶ Guided tour** button at the
top — short, Brilliant-style stepped lessons with live animations that run right on this page:
you can train a real PINN in your browser, watch spectral bias happen, and replay measured
failure curves. Each tour ends by routing you to a hands-on lab: a local
[marimo](https://marimo.io/) notebook in `notebooks/` for small CPU experiments, or a verified
Colab notebook when hosted compute fits better. Every step has a Next button that always
works — being stuck is not a state the tours allow.

---

## Running the code: one download, nothing to install

This page is only half the path. The other half is code you run: **eight starter scripts**
(`00`–`07`) and **three interactive labs**. You do not need to clone anything, install Python,
or build an environment — a single binary carries all of it.

1. Download the archive for your machine from the
   **[Releases page](https://github.com/mariotoffia/pinn/releases/latest)** — macOS, Linux and
   Windows, Intel and ARM.
2. Unzip it. On macOS clear the quarantine flag once, because the binary is unsigned:
   `xattr -d com.apple.quarantine ./pinn` (on Windows: SmartScreen → *More info* → *Run anyway*).
3. Run what you need:

```bash
./pinn serve    # this page, served offline from the binary itself
./pinn init     # write the starter kit and the labs into ./pinn-work
./pinn run 00   # check your hardware - the four facts above, measured on your machine
./pinn lab 02   # the autodiff lab, with sliders, in your browser
```

The binary embeds this page, the starter kit and the labs. The one thing it cannot embed is
PyTorch, so the first `run` or `lab` fetches a **CPU-only** PyTorch once into a private
environment under `pinn-work/`. It never installs into your system Python and never touches an
environment you already have. After that first fetch, everything runs offline.

Wherever a chapter names a file like `starter/scripts/02_autodiff_playground.py`, that path
lives inside `pinn-work/` — or inside your clone, if you prefer working from source. The
repository is **[github.com/mariotoffia/pinn](https://github.com/mariotoffia/pinn)**, and
[RUNNING.md](https://github.com/mariotoffia/pinn/blob/main/RUNNING.md) walks through every
command in plain English.

**Want to install nothing at all?** The ~50 one-click Colab links spread through the chapters
run in a browser, on Google's hardware, with no local setup whatsoever.

---

## The one exercise that makes the whole path click

Do this after Chapter 03, before you read a single PINN paper:

> Take a 3-layer `tanh` MLP `u(x,t)`. Use autodiff to compute
> `∂u/∂t + u·∂u/∂x − ν·∂²u/∂x²` at a batch of random points.

That expression is the residual of **Burgers' equation**, a classic test problem. If you can
compute it, you have already written the core of a PINN — without knowing it. The rest of the
method is "square it, average it, and call `.backward()`."

`starter/scripts/02_autodiff_playground.py` walks you through exactly this.

---

## Time budget, honestly

| Track | Hours | What it gets you |
|---|---:|---|
| **Fast lane** — Ch. 00, 03, 07, 08, 09, 10 + starter kit | **~35 h** | You can write, train and debug a PINN, and you know when not to use one |
| **Full path** — everything except the RL chapter | **~150 h** | Real competence; you can read the research literature critically |
| **Full path + RL** | **~240 h** | Add control of physical systems governed by PDEs |

The RL chapter is upfront about one thing: **you need zero reinforcement learning to build a
PINN.** The chapter is there because RL matters for *controlling* physical systems — a different
and genuinely valuable job. Chapter 06 tells you exactly how much of it to take.

---

## Conventions used in this document

- Every resource carries a type tag, a cost tag, and a one-line reason why it is here.
- `[free]` / `[paid]` is stated for anything that costs money.
- **Bold verdicts** are opinionated on purpose. Disagree where you have evidence.
- A **CUT LIST** appears in several chapters: famous resources that were deliberately skipped,
  with the reasons.
- Every claim about a library's health (maintained / dead / archived) was checked in August 2026.

---

## Next

→ **[01 — Math Refresher](#/01-math-refresher)** if you want to reload the math first
→ **[03 — Automatic Differentiation](#/03-autodiff)** if you already know deep learning and want the fast lane
→ **[16 — Roadmap](#/16-roadmap)** for the week-by-week plan
