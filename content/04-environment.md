---
title: Environment Setup
subtitle: Apple Silicon, CPU, float64, Colab — the setup that computes the right answer
minutes: 19
---

# 04 — Environment Setup

Checked against PyPI and the live docs on **22 August 2026**.

| Package | Version | Released | Requires |
|---|---|---|---|
| `torch` | **2.13.0** | 2026-07-08 | Python ≥3.10 |
| `jax` / `jaxlib` | **0.11.1** | 2026-08-17 | Python **≥3.12** |
| `mlx` | 0.32.1 | 2026-08-18 | — |
| `flax` | 0.12.9 | 2026-08-18 | Python ≥3.12 |
| `equinox` | 0.13.8 | 2026-05-05 | Python ≥3.10 |
| `optax` | 0.2.8 | 2026-03-20 | Python ≥3.10 |
| `diffrax` | 0.7.2 | 2026-02-18 | Python ≥3.11 |
| `deepxde` | 1.15.0 | 2025-12-05 | — |
| `keras` | 3.15.1 | 2026-07-29 | — |
| `tensorflow` | 2.21.0 | 2026-03-06 | — |
| `nvidia-physicsnemo` | 2.1.1 | 2026-06-08 | Python ≥3.11,<3.14 |
| **`jax-metal`** | **0.1.1** | **2024-10-08 — dead** | — |

Two facts shape everything below:

> **PyTorch 2.13 macOS wheels are `macosx_14_0_arm64` only** — macOS 14 or newer, Apple Silicon
> only. Intel Mac support is gone.
> **JAX 0.11 requires Python ≥3.12**, while PyTorch accepts ≥3.10. The versions that work for
> both on a Mac are 3.12 / 3.13 / 3.14. **Pick 3.13.**

---

## 4.1 The recommended setup, in full

(`uv` is a fast, modern Python package and environment manager — think pip + virtualenv in one
tool. It is what this path uses everywhere.)

```bash
# 1. Toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh
# optional, only if you later need FEM/mesh native deps:
brew install miniforge

# 2. Project
uv init pinn-lab && cd pinn-lab
uv python pin 3.13

# 3. Core stack — PyTorch primary, JAX for the grad-of-grad clarity
uv add "torch>=2.13.0" "jax>=0.11.0" "equinox>=0.13.0" "optax>=0.2.8" \
       "numpy>=2.0" "scipy>=1.14" "matplotlib>=3.9"

# 4. Ground-truth generators and visualisation — no compiled code, no conda pain
uv add scikit-fem py-pde pyvista

# 5. Notebooks, tracking, quality of life
uv add --dev jupyterlab ipykernel ruff pytest
uv add tensorboard tqdm

# 6. VERIFY BEFORE WRITING ANY PHYSICS
uv run python 00_check_environment.py

# 7. Later, when you want a reference implementation to compare against
uv add deepxde        # then: DDE_BACKEND=pytorch uv run python example.py
```

At the top of **every** PyTorch entry point:

```python
import torch
torch.set_default_dtype(torch.float64)   # non-negotiable for PINNs
DEVICE = torch.device("cpu")             # deliberate: MPS lacks f64 + double-backward
```

and as lines 1–2 of every JAX file:

```python
import jax; jax.config.update("jax_enable_x64", True)
```

**Deliberately skipped:** `jax-metal` (dead), `jax-mps` (experimental, float32-only), TensorFlow
(unless DeepXDE forces it on you), Keras 3 (wrong abstraction level for custom residual losses),
PhysicsNeMo locally (CUDA-only, macOS unsupported), and `torch.compile` — leave it off until
your model is correct and a profiler says it would actually help.

---

## 4.2 Apple Silicon: what actually works

**MPS backend note** — https://docs.pytorch.org/docs/stable/notes/mps.html (needs macOS 14.0+).
**Apple's install page** — https://developer.apple.com/metal/pytorch/.
**Op coverage matrix** — https://qqaatw.dev/pytorch-mps-ops-coverage/ — check here before
assuming an operation works. Umbrella tracking issue:
[pytorch#77764](https://github.com/pytorch/pytorch/issues/77764).

**Works on MPS:** standard convolution/linear/attention forward passes and *first-order*
backward passes, most elementwise operations, reductions. FlexAttention landed on MPS in 2.13
with hand-written Metal kernels.

**Does not work — and will not:**

- **float64.** Apple GPUs have no 64-bit float units, and Metal Shading Language has no
  `double` type. Verified in the shipped 2.13.0 wheel: `torch/_inductor/codegen/mps.py`
  contains `raise RuntimeError("float64 is not supported by MPS")`. At runtime you get
  `TypeError: Cannot convert a MPS Tensor to float64 dtype...`. There is no flag, no fallback,
  no nightly build that fixes it. See the
  [PyTorch Forums thread running Jan–Jun 2026](https://discuss.pytorch.org/t/float64-tensors-on-mps/224325).
- **Second derivatives through `nn.Linear`.**
  `RuntimeError: derivative for aten::linear_backward is not implemented` —
  [#98498](https://github.com/pytorch/pytorch/issues/98498) (open since April 2023),
  [#83386](https://github.com/pytorch/pytorch/issues/83386), [#92206](https://github.com/pytorch/pytorch/issues/92206).
  Works on CPU and CUDA; fails on MPS.

Two independent show-stoppers. **On a Mac, PINNs run on the CPU.**

**`PYTORCH_ENABLE_MPS_FALLBACK=1`** is a debugging aid, not a production setting. It must be set
*before* `import torch`. Every unimplemented operation then makes a round trip to the CPU and
back, which can make the "accelerated" path slower than plain CPU. And it does not rescue every
operation ([#134416](https://github.com/pytorch/pytorch/issues/134416)).

**What speed to expect.** For PINN-shaped work, expect **no speedup from MPS — often a
slowdown.** A GPU wins when it has enough work to hide the cost of launching kernels: large
convolutions, transformer blocks, big batches. A PINN is a small MLP on a few thousand points,
evaluated through a *triple* pass over the graph. That is close to the worst case for launch
overhead. Reference measurements:
[Profiling Apple Silicon Performance for ML Training](https://arxiv.org/pdf/2501.14925).

### JAX on Apple Silicon

**`jax-metal` is dead.** Last PyPI release 0.1.1 (October 2024), declaring `jaxlib>=0.4.34`
against today's 0.11.1. The JAX maintainers closed all Metal-tagged issues in December 2025,
saying *"No active development on jax-metal"*
([Apple Developer Forums thread](https://developer.apple.com/forums/thread/815139)). It is
broken against modern JAX in any case
([jax#34109](https://github.com/jax-ml/jax/issues/34109)) — and Apple's own docs listed
`np.float64` as unsupported, so it was useless for PINNs even when it worked.

**`jax-mps`** (https://github.com/tillahoffmann/jax-mps) is a community plugin that routes
JAX → StableHLO → MLX to reach the Apple GPU. Actively developed, and about 3.7× faster than
CPU on a ResNet18 on an M4 — but it pins `jax<0.11`, is single-device, prints
`Platform 'mps' is experimental`, and **does not support float64** (MLX's GPU path is
float32-only).

**Verdict: plain CPU JAX is the correct choice on a Mac** — it is the only Mac JAX setup with
float64. Just `pip install jax`: the arm64 wheels are native, and the CPU backend is well
optimised.

### MLX — Apple's own framework

- Docs: https://ml-explore.github.io/mlx/build/html/index.html
- Function transforms: https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html
- Data types: https://ml-explore.github.io/mlx/build/html/python/data_types.html
- WWDC25 intro: https://developer.apple.com/videos/play/wwdc2025/315/

**Higher-order autodiff: yes, for real.** The docs state *"Using `grad()` on the output of
`grad()` is always ok. You keep getting higher order derivatives."* **float64: supported, but
CPU-only** — *"Arrays with type `float64` only work with CPU operations."* That is strictly
better than PyTorch-MPS, which has no float64 on any device.

**So can you build PINNs in MLX?** Technically yes. Practically, not on a learning path: no
DeepXDE backend, no Diffrax, no Optax, few reference implementations — and the moment you need
float64 you are on the CPU anyway, where PyTorch or JAX give you the same speed plus the entire
literature. **A fascinating side quest. Not the vehicle.**

---

## 4.3 float32 vs float64 — why this is a correctness issue

Deep learning defaults to float32 because gradient descent on noisy data tolerates noise. **A
PINN is not doing that.** It is solving an equation, and its loss is a residual that should go
to zero.

1. **Derivatives amplify rounding error.** float32 carries about 7 decimal digits
   (`eps ≈ 1.19e-7`); float64 about 16 (`eps ≈ 2.22e-16`). Every differentiation stage costs
   digits. On a 3-layer tanh MLP, `u_xx` computed in float32 vs float64 disagrees by about
   5e-7 relative — and that noise *is* the floor of your residual. Squared into a mean-square
   loss, float32 physically cannot push a residual loss much below ~1e-13, and in practice it
   stalls far earlier.
2. **It breaks the optimiser, not just the answer.** **[FP64 is All You Need: Rethinking Failure
   Modes in Physics-Informed Neural Networks (NeurIPS 2025)](https://arxiv.org/abs/2505.10949)**
   finds that PINNs' famous failure modes are **not** hopeless local minima but **insufficient
   numerical precision**: in float32, **L-BFGS hits its stopping rule and quits early** — the
   loss looks converged while the solution is bad. *"Upgrading to FP64 rescues optimization,
   enabling vanilla PINNs to solve PDEs without any failure modes."*
   ([NeurIPS poster](https://neurips.cc/virtual/2025/poster/120125))
3. **L-BFGS estimates curvature from differences of gradients.** Subtracting two nearly equal
   float32 numbers wipes out the digits — "catastrophic cancellation", built into the method.

For balance: [mixed precision for scientific ML](https://arxiv.org/abs/2401.16645) argues that
float16/float32 can save time and memory at equal accuracy for *some* workloads. Try that
*after* you have a working float64 baseline. Never as a starting point.

**Device summary:**

| Device | float64 | 2nd-order autograd through `nn.Linear` | Verdict for PINNs |
|---|---|---|---|
| **CPU** (Apple Silicon or x86) | ✅ full | ✅ works | **Use this on a Mac** |
| **CUDA** | ✅ (but ~1/32 speed on consumer/T4 cards) | ✅ works | Best if available |
| **MPS** | ❌ hard `TypeError` | ❌ not implemented | **Unusable** |
| **MLX GPU** | ❌ raises | ✅ | float32 work only |
| **MLX CPU** | ✅ | ✅ | Works, tiny ecosystem |
| **jax-metal** | ❌ | — | **Dead — ignore** |

Note the CUDA detail: float64 *works* there, but consumer and inference cards run it at about
1/32 of their float32 speed. On a Mac CPU the penalty is closer to 2× — one more reason the CPU
is a perfectly reasonable PINN device.

---

## 4.4 Making the same project run unchanged on Colab

Two rules: put your code in an installable package, and never hard-code the device.

```
pinn-lab/
├── pyproject.toml
├── src/pinn_lab/__init__.py
└── notebooks/01_burgers.ipynb
```

First cell of the notebook:

```python
import os, sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO = '/content/drive/MyDrive/pinn-lab'
    # or clone fresh: !git clone <url> /content/pinn-lab ; REPO='/content/pinn-lab'
    !pip install -q -e {REPO}          # editable install of YOUR package
else:
    REPO = os.path.abspath('..')

import pinn_lab                        # identical import path in both environments
```

Generate a Colab-friendly requirements file from your lockfile:

```bash
uv export --no-hashes --no-emit-project --format requirements-txt > requirements-colab.txt
```

Do **not** pin `torch` in that file. Colab ships a CUDA-matched build, and forcing a version
triggers a slow, often-broken reinstall.

### Colab free tier, 2026

- https://colab.research.google.com/ · FAQ: https://research.google.com/colaboratory/faq.html
- **12 hours maximum** per notebook, "depending on availability and your usage patterns";
  idle runtimes are shut down. Google explicitly **does not publish the limits** — *"Colab does
  not publish these limits, in part because they can vary over time."*
- The free-tier GPU is typically a **T4 (16 GB)**, but neither a GPU nor any particular model is
  guaranteed. Heavy or automated use gets throttled to CPU-only.
- **Checkpoint to Drive every N epochs.** Never assume a session survives.
- A T4 does support float64, but at roughly **1/32** of its float32 speed — so a float64 PINN on
  a free T4 is often *not* much faster than your Mac's CPU. **Measure before you migrate.**

### Kaggle Notebooks — the better free tier

https://www.kaggle.com/code — **30 GPU-hours per week** (T4×2 or P100) and 20 TPU-hours per
week, 12-hour maximum sessions, phone verification required. A real, published quota instead of
an unstated one. The catch: the clock counts wall time, so **stop idle sessions yourself.**

### Other remote options

- **Lightning AI Studios** — https://lightning.ai/ — about 22 free GPU-hours per month, with
  persistent VS Code/Jupyter environments that keep their state between sessions. The
  persistence is the real draw.
- **Modal** — https://modal.com/pricing — $30/month in free credits, serverless. Excellent for
  *sweeps* (launch 50 PINN configurations in parallel); poor for interactive exploration.

**Honest framing: a PINN learning project barely needs remote GPUs at all.** PINNs are small.
The bottleneck is your understanding, not compute.

---

## 4.5 uv, in detail

- Docs: https://docs.astral.sh/uv/ · install: https://docs.astral.sh/uv/getting-started/installation/
- Projects: https://docs.astral.sh/uv/guides/projects/
- **PyTorch integration guide** — https://docs.astral.sh/uv/guides/integration/pytorch/ — **the
  page that solves CPU-vs-CUDA wheel selection.** Read it before hand-editing index URLs.

**The thing nobody tells you: there is no separate MPS wheel.** The default macOS arm64 wheel
already contains MPS, and the `pytorch-cpu` index serves that same wheel on macOS. You choose
CPU vs MPS *at runtime* with `.to(device)`, not at install time.

Simplest cross-platform install:

```bash
uv pip install torch --torch-backend=auto     # detects CUDA, falls back to CPU
```

If you want one project that picks CPU wheels on your Mac and CUDA wheels elsewhere, use the
extras + explicit-index pattern from the uv PyTorch guide. Note that `uv lock --extra cpu` is
**not** valid — `--extra` belongs to `uv sync`.

**Miniforge** (https://github.com/conda-forge/miniforge) is the conda distribution to use on
Apple Silicon — conda-forge by default, native arm64, no Anaconda licensing questions. Use it
*only* if you need non-Python native dependencies (FEniCSx, certain HDF5 builds), and then in a
**dedicated environment**. Mixing conda-forge MPI into a pip-based ML environment is a classic
way to break your PyTorch install. **Do not install Anaconda proper.**

---

## 4.6 Experiment tooling

- **TensorBoard** — ships with PyTorch as `torch.utils.tensorboard`. Zero setup, local, good
  enough. **The pragmatic default for a learning project.** https://docs.pytorch.org/docs/stable/tensorboard.html
- **Weights & Biases** — https://docs.wandb.ai/quickstart — best-in-class run comparison.
  Logging each loss term separately — `wandb.log({"loss_pde": ..., "loss_bc": ...})` — is
  exactly how you diagnose PINN loss imbalance. The free tier is generous for personal use;
  needs an account.
- **MLflow** — https://mlflow.org/docs/latest/ml/tracking/ — fully local (`mlflow ui` on
  SQLite), no account, no telemetry.
- **Aim** — https://aimstack.readthedocs.io/en/latest/ — local-first with a fast UI; last
  release May 2025, maintenance has slowed.

**Config:** a frozen dataclass beats Hydra for your first several PINNs.

```python
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Config:
    layers: tuple = (2, 128, 128, 128, 1)
    lr: float = 1e-3
    n_collocation: int = 4096
    seed: int = 0
    fourier_sigma: float = 2.0

cfg = Config()          # frozen=True removes a whole class of mid-run mutation bugs
```

Adopt **Hydra** (https://hydra.cc/docs/intro/) once you have more than about three experiment
axes. Its multirun sweep (`python train.py -m lr=1e-3,1e-4 layers=4,8`) is genuinely useful for
PINNs, because you end up sweeping PDE parameters, collocation counts and loss weights at the
same time.

**Seeding:** https://docs.pytorch.org/docs/stable/notes/randomness.html — read the caveats.
Bit-for-bit reproducibility is only guaranteed *within* one release, platform and device.

```python
def set_seed(seed=0):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
```

Also log `uv.lock` plus `torch.__version__` in your run config — "same seed, different library
version" is the most common reproducibility failure in practice.

**Profiling:** https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html. For a
PINN, watch the **ratio of time spent in the second-derivative pass to the forward pass.** If
`u_xx` costs more than ~4–5× the forward pass, switch from batched
`autograd.grad(create_graph=True)` to `vmap`-ed `torch.func` transforms, or use forward mode
for the input derivatives. In JAX, always call `.block_until_ready()` around timed sections and
warm up first — JAX runs asynchronously, so naive timing measures dispatch, not compute.

---

## 4.7 TensorFlow / Keras 3 — the honest answer

**Not worth it in 2026 for this path, with two narrow exceptions.**

TensorFlow 2.21 is maintained, but it is not where research or new PINN work happens. Learning
`GradientTape` nesting for higher-order derivatives is real effort spent on a skill you will not
reuse.

1. **DeepXDE is backend-agnostic**, and much of its older example code and its original papers
   are TensorFlow-flavoured. You will *read* TF even if you never write it. Set the backend
   explicitly: `DDE_BACKEND=pytorch python pde.py` — autodetection picks whatever it finds
   first, and will silently change under you.
   (https://deepxde.readthedocs.io/en/latest/user/installation.html)
2. **NVIDIA PhysicsNeMo** (formerly Modulus, originally SimNet) began as TF and moved to
   PyTorch; its docs still carry TF-era vocabulary.

**Keras 3** (https://keras.io/keras_3/) genuinely runs on multiple backends (JAX, TensorFlow,
PyTorch, OpenVINO) and is a real engineering achievement — but it is the wrong abstraction level
here. Keras's value is hiding the training loop; **a PINN's entire difficulty lives in the
training loop.** Keras also documents a cross-backend divergence larger than 1e-7 in float32 —
a precision floor you cannot afford.

---

## 4.8 Top gotchas

1. **You will try MPS anyway, and it will fail twice, for two unrelated reasons.** Fixing
   float64 does not fix double-backward. Set `DEVICE = "cpu"` on day one and stop thinking
   about it.
2. **`jax_enable_x64` must run before any array is created, and it fails silently.**
3. **`jax-metal` is dead.** Do not spend an afternoon on it.
4. **`set_default_dtype` after building your model does nothing to that model.**
5. **`in_axes` (JAX) vs `in_dims` (torch.func).**
6. **Zero gradient on the output bias from a pure residual loss** → `allow_unused=True` or
   `torch.func.grad`.
7. **`PYTORCH_ENABLE_MPS_FALLBACK=1` quietly makes things slower**, and must be set before
   `import torch`.
8. **Colab guarantees nothing; Kaggle's clock runs on wall time.** Checkpoint, and stop idle
   sessions.

Bonus: `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'`
means you are running Python from inside the `torch/` package directory, so `torch/types.py` is
shadowing the standard library. `cd` somewhere else.

---

## 4.9 Not on a Mac? Windows and Linux

Everything above solved a Mac-specific puzzle — but its conclusion, **run PINNs on the CPU in
float64**, is available on every machine. Here is the honest map:

| Your machine | Your lane |
|---|---|
| **Windows, no NVIDIA GPU** | **Exactly the Mac's position.** CPU float64 locally, Colab/Kaggle for the rare GPU job. You lose nothing on this path |
| **Windows or Linux with an AMD / Intel GPU** | Treat it as "no GPU" for this path — see the DirectML note below |
| **Windows + NVIDIA GPU** | CPU float64 for PINN training (consumer cards run FP64 at ~1/32 speed — see §4.3); the GPU shines on float32 work — neural operators (Ch. 13), RL (Ch. 06), sweeps. PyTorch CUDA works natively; **JAX CUDA does not — use WSL2** |
| **Linux + NVIDIA GPU** | Everything in this chapter works as written, plus native CUDA for both frameworks. The best-supported setup there is |

**Windows, native (the short version).** Install uv from PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then §4.1 works unchanged — same `uv init`, same `uv add` line, same two lines at the top of
every file. Two facts make native Windows painless for this path: the default `torch` package
on Windows **is the CPU build** (the CUDA build lives on a separate index — you cannot install
it by accident), and **JAX's CPU wheels support Windows** (it is only JAX's *CUDA* build that
needs Linux or WSL2). The starter kit, the marimo labs and the generated `index.html` are all
platform-neutral.

**The Windows version of the MPS lesson.** If you have an AMD or Intel GPU, you will find
`torch-directml`, which routes PyTorch to any DirectX 12 GPU. Do not spend your afternoon
there: **Microsoft has placed DirectML in maintenance mode** (the banner is on the
[official repo](https://github.com/microsoft/DirectML)), operator coverage was never complete,
and the float64 + second-derivative combination a PINN needs was never its target. Same
two-strike verdict as Apple's MPS, different vendor: **on a non-NVIDIA GPU, the CPU is the
correct PINN device.** This keeps being true for a structural reason — fact 3 of Chapter 00 is
about *precision*, and consumer GPUs of every brand are built for float32.

**WSL2 — the "make my Windows a Linux" option.** Install Ubuntu under
[WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and this chapter applies
*verbatim*, including JAX CUDA (if you have NVIDIA) and the smoother path for native
dependencies: **FEniCSx on Windows is best installed under WSL2 or Docker** — the native
Windows conda packages are still in beta and lack PETSc. The pure-Python solvers this path
actually starts with (scikit-fem, py-pde) do not care what OS they run on.

**Linux.** §4.1 as written. Nothing to substitute.
