# Interactive labs (marimo) — and when to use Colab instead

This folder holds the **local labs**: interactive [marimo](https://marimo.io/) notebooks
that pair with the guided tours in the hub (`index.html` → open a chapter → "▶ Guided
tour"). marimo notebooks are plain `.py` files, they re-run reactively when you move a
slider, and cells can never run out of order — which makes them ideal for
"turn the knob, watch the effect" teaching.

**The contract: these labs always run.** Every task cell already contains a working
solution, so pressing "run" can never leave you stuck. The exercises ask you to *replace*
a solution with your own attempt; hints sit in fold-out accordions, and the original
solution is always one undo away. Click through = guaranteed success; type along = the
actual learning.

## Run a lab

(The plain-English, copy-paste version of everything on this page — and of the starter
scripts — is [RUNNING.md](../RUNNING.md) at the repository root.)

```bash
# from NOTHING but uv (Chapter 04 installs it in one line): each lab's own header
# (PEP 723) tells uv what to install — marimo, NumPy, PyTorch — once, cached
uvx marimo edit --sandbox notebooks/02_autodiff_lab.py

# or, inside an environment you manage yourself (make dev at the repo root)
.venv/bin/marimo edit notebooks/02_autodiff_lab.py
```

With `--sandbox` there is nothing to set up. One nuance, on Linux only: PyPI's default
torch there bundles CUDA — a several-GB download that still runs fine on CPU (macOS and
Windows wheels are already slim and CPU-only). The `pinn` binary and `make dev` both pin
PyTorch's CPU-only index instead; so does prefixing the command with
`UV_INDEX=https://download.pytorch.org/whl/cpu UV_INDEX_STRATEGY=unsafe-best-match`.
In an environment you manage yourself, you install PyTorch once (Chapter 04). Either way
the labs run on CPU, in seconds to a couple of minutes at the default sliders.

**VS Code:** accept the recommended extensions on first open, then either run the `lab: ...`
tasks (*Terminal ▸ Run Task…*) or open a lab file and start it from the
[marimo extension](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)'s
editor button. Details: "Working in VS Code" in the root README.

## The labs

| Lab | Pairs with | What it does | Success looks like |
|---|---|---|---|
| `02_autodiff_lab.py` | Ch. 03 tour | ∂u/∂x and ∂²u/∂x² by autodiff; the Burgers residual at thousands of points; Idiom A vs Idiom B; FD check with a movable h; `gradgradcheck` | FD agrees ~1e-6; idioms agree ~1e-15; gradgradcheck `True` |
| `03_spectral_bias_lab.py` | Ch. 05 tour | A real MLP fits `sin(x) + a·sin(kx)`; sliders for k, a, σ, budget; Fourier features on/off | Fast mode stalls without features; converges with them; σ too small/large both fail — hence σ ∈ [1,10] |
| `04_oscillator_pinn_lab.py` | Ch. 08 tour | A full PINN for the damped oscillator: Step-0 input scaling, soft-vs-hard IC, Adam → L-BFGS with a frozen set | Measured at defaults: soft λ=1 → 2.9e-3 · λ=100 → 3.0e-1 (worse!) · hard → 2.5e-3; L-BFGS off costs ~100× |

## marimo or Colab? The routing rule this path uses

| Situation | Use | Why |
|---|---|---|
| Concept labs: small, CPU, float64, seconds-to-minutes, slider-driven | **marimo, locally** | Reactive re-runs, no stale cells, no session limits, float64 CPU is the correct PINN configuration anyway (Ch. 04) |
| An external notebook that is already excellent and Colab-ready | **Colab** | Moseley's oscillator workshop, the Tancik Fourier-features demo, d2l.ai sections, MIT 6.S191 labs — all verified and linked from the tours with what/foundations/expected-result notes |
| Anything needing a GPU: jaxpi, Beltrami flow, neural-operator training | **Colab / Kaggle** | Chapter 04 §4.4 rules; these are linked at the right tour steps |
| The starter kit's measured baselines | **local scripts** | `starter/scripts/00–07` are the reference numbers the tours quote |

marimo files are not Jupyter files — Colab cannot open them. That is deliberate: the
local lane and the hosted lane carry different work. (If you ever need one in Jupyter:
`uvx marimo export ipynb notebooks/<lab>.py -o lab.ipynb`.)

## Where the tour data comes from

The chapter 09 tour plays back **measured** curves, not illustrations. They were produced
by `tools/experiments/beta_sweep.py` (vanilla + curriculum) and
`tools/experiments/beta_march.py` (time-marching rescue) — CPU, float64, seed 0 — and
baked into `src/tourdata.ts`. Re-run those scripts to regenerate or extend the data.

## Planned next labs (same pattern)

`01_backprop_by_hand` (NumPy, pairs with the Ch. 02 tour) · `05_heat1d` (first PDE with
an exact solution) · `06_burgers_ablation` (the recipe flags, interactive) ·
`07_inverse` (recover ν from noisy data — the thing PINNs are good at) ·
`13_fno_darcy` (train at 64², evaluate at 256², with `neuraloperator`).
