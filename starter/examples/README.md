# What the scripts should produce

Four reference figures, so you can tell at a glance whether your run went the way it should.

| File | From | What to look at |
|---|---|---|
| `03_spectral_bias_modes.png` | `03_spectral_bias.py` | The recovered amplitude of the high-frequency mode. A plain MLP flat-lines near 0.05 against a true value of 0.30; with Fourier features it reaches 0.30 within a few hundred steps. **This one figure explains half of chapter 09.** |
| `06_burgers_plain.png` | `06_pinn_burgers.py --plain` | The 2019 baseline. The shock is there but smeared, and the error is spread everywhere. |
| `06_burgers_all.png` | `06_pinn_burgers.py --all` | The whole recipe. **The entire error collapses into a thin band at the shock** — exactly where a PINN should struggle, and nowhere else. |
| `07_inverse_nu.png` | `07_pinn_inverse.py` | The discovered coefficient converging to 0.05 from a starting guess of 0.5, using 60 noisy points. |

Your own `out/` directory is gitignored, so these stay as the reference.
