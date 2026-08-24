# PINN starter kit

Small, readable building blocks, plus eight progressive scripts — from a NumPy MLP with a
hand-written backward pass, all the way to a working inverse PINN.

**Everything here runs on the CPU, in float64, in minutes.** No GPU, and no framework beyond
PyTorch + NumPy + Matplotlib.

## Install

```bash
# with uv (recommended)
uv venv && uv pip install -e .

# or with pip
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

Optional extras:

```bash
uv pip install -e ".[solvers]"   # scikit-fem + py-pde, for chapter 07 ground truth
uv pip install -e ".[deepxde]"   # DeepXDE, to compare against a mature library
```

## Run, in order

```bash
python scripts/00_check_environment.py      #  10 s   checks what YOUR machine can do for PINNs
python scripts/01_mlp_from_scratch.py       #  15 s   backprop by hand, NumPy only
python scripts/02_autodiff_playground.py    #  15 s   the Burgers residual, two idioms, gradgradcheck
python scripts/03_spectral_bias.py          #  40 s   why PINNs need Fourier features
python scripts/04_pinn_oscillator.py        #   4 min first PINN: damped oscillator, soft vs hard IC
python scripts/05_pinn_heat1d.py            #   2 min first PDE, against an exact solution
python scripts/06_pinn_burgers.py --plain   #   4 min the canonical benchmark, 2019 baseline
python scripts/07_pinn_inverse.py           #   2 min recover a coefficient from sparse noisy data
```

Every script writes its figures to `out/` and prints the numbers that matter.
Reference figures — what a correct run looks like — are in [`examples/`](examples/).

## Results you should get

Measured on a CPU, in float64, with seed 0. Your numbers will differ a little; the *ordering*
should not.

| Script | Result |
|---|---|
| `01` | gradient check vs finite differences: **3.4e-08** relative. Final MSE **8.0e-06** |
| `02` | finite differences vs autodiff for `u_xx`: **1.9e-06**. Idiom A vs Idiom B: **2.3e-16** |
| `03` | a plain MLP recovers the 0.30 high-frequency mode as **0.05** (it never reaches half the true amplitude). With Fourier features: **0.3000** |
| `04` | soft IC λ=1 → **4.8e-04** · soft IC λ=100 → **1.1e-02** · **hard IC → 1.6e-04** |
| `05` | soft BC/IC → **2.2e-04** · `--hard` → **4.1e-05**, with the IC error exactly 0 and the BC error at machine precision |
| `06` | `--plain` → **1.4e-02** · `--all` → **4.7e-03**. Raissi et al. report 6.7e-04, using far more L-BFGS iterations |
| `07` | ν recovered from 60 noisy points: 0% noise → **0.055%** error · 1% → **0.28%** · 5% → **1.1%** · 20% → **5.8%** |

Two things worth noticing in that table. In `04`, the **hand-tuned λ=100 does worse than
λ=1** — the loss weights are not a knob you can set by feel. And in `06`, reproducing the
paper's headline 6.7e-4 takes several times more L-BFGS iterations than the default budget
here: run `--all --lbfgs-steps 20000` overnight if you want to watch it converge.

## The ablation that matters

`06_pinn_burgers.py` maps directly onto the ten-step recipe in Chapter 10:

```bash
python scripts/06_pinn_burgers.py --plain                # nothing on: the 2019 baseline
python scripts/06_pinn_burgers.py --fourier              # + step 2
python scripts/06_pinn_burgers.py --fourier --rwf        # + step 3
python scripts/06_pinn_burgers.py --all                  # everything, incl. Adam -> L-BFGS
python scripts/06_pinn_burgers.py --all --seed 1         # ...and check the seed variance
```

Run each over three seeds and build your own table. That table is the single most valuable
artifact of the whole learning path.

## Package layout

| Module | What it holds |
|---|---|
| `pinnlab/device.py` | Device/dtype selection and the environment report |
| `pinnlab/nets.py` | tanh MLP with Glorot init, Fourier features, random weight factorisation |
| `pinnlab/derivatives.py` | `d()`, `grad_wrt()`, `laplacian()` — both autograd idioms |
| `pinnlab/sampling.py` | uniform / Latin hypercube / Sobol / residual-based adaptive refinement |
| `pinnlab/losses.py` | gradient-norm loss balancing, causal temporal weighting |
| `pinnlab/train.py` | the Adam → L-BFGS loop, with logging |
| `pinnlab/plotting.py` | the three-panel prediction / reference / error figure |
| `pinnlab/reference.py` | exact Burgers solution via Cole–Hopf, and the 1D heat series solution |
