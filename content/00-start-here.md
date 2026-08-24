---
title: Start Here
subtitle: What this path is, who it is for, and a sober picture of the field
minutes: 15
---

# Start Here

This is a complete, self-contained learning path. It takes you from **"I can program, and I once
took a linear algebra course"** to **"I can build, train, debug, and judge a
Physics-Informed Neural Network (PINN)."**

A quick note on the name: a **PINN** is a neural network that is trained to obey a physical law
(a differential equation) instead of being trained on a big dataset. That is the whole subject of
this path. Do not worry if that sentence does not fully make sense yet — it will.

You do not need special hardware. Everything on this path runs on an ordinary computer —
Mac, Windows or Linux, with or without a GPU — because the configuration it uses, **CPU +
float64**, exists on every machine. The rare job that does need a GPU runs for free in a
browser, on **Google Colab or Kaggle**. The section **What hardware you need**, below, states
what a PINN workload requires and which lane your machine falls into; Chapter 04 has the
details for every platform.

Everything is free unless it is marked `[paid]`. Every link was checked live in **August 2026**.

---

## The whole field in one paragraph

A PINN uses a small, smooth neural network `u_θ(x,t)` as a stand-in for the solution of a
**partial differential equation (PDE)** — an equation that links a quantity (like temperature) to
its own rates of change in space and time. The network's derivatives are computed exactly, using
**automatic differentiation** (the same machinery behind normal deep learning). Training then
pushes the equation's error — the **residual** — toward zero at randomly chosen points. In other
words: the equation itself becomes an endless source of free training labels. That idea is
elegant. For **inverse problems** (finding unknown physical constants from
measurements), **data assimilation** (filling gaps between sparse sensors), **awkward
geometries**, and **high-dimensional PDEs**, it is a real ability that classical solvers do not
offer cheaply. But as a *forward solver* — just solving a known equation that a classical method
like the finite element method can already handle — PINNs lose, usually by a wide margin. Worse,
they fail **silently**: the loss goes down, the plot looks smooth, and the answer is still wrong.
So learn PINNs for what they are good at, and learn their failure modes at the same time as the
method.

If you remember nothing else from this path, remember that paragraph.

---

## What hardware you need

A PINN workload makes two hard demands of your hardware, and one thing turns out not to
matter:

1. **float64 (double precision).** The paper [FP64 is All You Need (NeurIPS 2025)](https://arxiv.org/abs/2505.10949)
   shows that several famous PINN "failure modes" were never real optimisation dead ends. They
   were 32-bit rounding limits: the optimiser (L-BFGS) hit its stopping rule and quit early,
   while the answer was still wrong.
2. **Second-order automatic differentiation** — taking a derivative of a derivative,
   `grad(grad(u))`. A PINN is *built* on that operation.
3. **Raw compute barely matters.** PINN networks are tiny (3–6 layers of 128–512 units) with
   small batches — exactly the kind of work where a GPU spends more time on launch overhead
   than on math. A plain CPU is not a fallback here; it is a first-class device.

Every CPU satisfies all of this. GPUs are another story: consumer GPUs are built for float32,
and their float64 and higher-order-autodiff support ranges from slow to absent. Find your
machine:

| Your machine | Your PINN lane |
|---|---|
| **Any CPU** (Mac, Windows, Linux) | ✅ **CPU + float64 — the default lane for this whole path** |
| **NVIDIA GPU (CUDA)** | ✅ Works — float64 and second derivatives are supported — but consumer cards run float64 at ~1/32 of their float32 speed, so the CPU often keeps up on PINN-sized work. The GPU shines on float32 work: neural operators (Ch. 13), RL (Ch. 06) |
| **Apple GPU (MPS)** | ❌ No float64 (Apple's GPU language, Metal, has no `double` type) and no second derivative through `nn.Linear` ([pytorch#98498](https://github.com/pytorch/pytorch/issues/98498), open since April 2023). Use the CPU |
| **AMD / Intel GPU** | ❌ Same verdict, different vendor (Chapter 04 §4.9). Use the CPU |
| **None of the above / a weak laptop** | ✅ **Colab or Kaggle** run everything in a browser, free — the lane whenever local support is missing or a job outgrows your machine |

```python
# The two lines at the top of every PyTorch entry point on this path
import torch
torch.set_default_dtype(torch.float64)   # BEFORE constructing any module
DEVICE = torch.device("cpu")             # or "cuda" if you have it — never "mps"
```

Run `starter/scripts/00_check_environment.py` on day one. In about ten seconds it verifies, on
your own machine, what your hardware can and cannot do for PINN training — including
reproducing the GPU failures above, if you have one of the GPUs in question. Details in Chapter 04 §4.9.

---

## How this path is organised

| Part | Chapters | What you get |
|---|---|---|
| **I — Foundations** | Math Refresher, Neural Networks, Autodiff | Backprop derived by hand; automatic differentiation as a core idea, not a black box |
| **II — Practice** | Environment, Training Craft, Reinforcement Learning | A working local setup, plus the optimisation skills that decide whether a PINN converges |
| **III — Physics** | PDE Primer, PINN Core, Failure Modes, The Recipe | The method, stated precisely — and where it breaks |
| **IV — Tooling** | Frameworks, Worked Examples | What to install, what to run, what to avoid |
| **V — Beyond** | Neural Operators, Reality Check, Reading List, Roadmap | What comes after PINNs, and where the field really stands |

Chapter **03 — Automatic Differentiation** is the hinge of the whole path. Everything before it
is standard deep learning. Everything after it depends on being comfortable with
`grad(grad(u))` — taking a derivative of a derivative. Do not skim it.

**Guided tours.** Five chapters (02, 03, 05, 08, 09) carry a **▶ Guided tour** button at the
top — short, Brilliant-style stepped lessons with live animations that run right on this page:
you can train a real PINN in your browser, watch spectral bias happen, and replay measured
failure curves. Each tour ends by routing you to a hands-on lab: a local
[marimo](https://marimo.io/) notebook in `notebooks/` for small CPU experiments, or a verified
Colab notebook when hosted compute fits better. Every step has a Next button that always
works, so you cannot get stuck.

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
./pinn run 00   # check your hardware against the requirements above
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

## Time budget

| Track | Hours | What it gets you |
|---|---:|---|
| **Fast lane** — Ch. 00, 03, 07, 08, 09, 10 + starter kit | **~35 h** | You can write, train and debug a PINN, and you know when not to use one |
| **Full path** — everything except the RL chapter | **~150 h** | Real competence; you can read the research literature critically |
| **Full path + RL** | **~240 h** | Add control of physical systems governed by PDEs |

The RL chapter is upfront about one thing: **you need zero reinforcement learning to build a
PINN.** The chapter is there because RL matters for *controlling* physical systems — a different
and valuable job. Chapter 06 tells you exactly how much of it to take.

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
