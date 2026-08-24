# How to run everything

This is the short, plain-English guide. It lists every command in this project, what to
type, and what you should see. It assumes nothing is set up yet.

You will use the **terminal** (on a Mac: the Terminal app; on Windows: PowerShell). You
type a command, press Enter, and read what comes back. That is all a terminal is.

There are three things you can run:

1. **The book** — 19 chapters with guided tours, read in your browser.
2. **The starter scripts** — eight small Python programs. Each proves one idea.
3. **The labs** — three interactive notebooks with sliders, in your browser.

And while you read, the chapters link about fifty **one-click Colab notebooks** — those are
covered in their own short section near the end of this page.

---

## The easiest route: the `pinn` program

`pinn` is one single file that contains the whole book, all the scripts and all the labs.

**Get it** from the [Releases page](https://github.com/mariotoffia/pinn/releases).
Download the file for your computer, then make it runnable:

| Your computer | Download | Then type this once |
|---|---|---|
| Mac (Apple Silicon) | `pinn-darwin-arm64` | `chmod +x pinn-darwin-arm64 && mv pinn-darwin-arm64 pinn` |
| Mac (Intel) | `pinn-darwin-amd64` | `chmod +x pinn-darwin-amd64 && mv pinn-darwin-amd64 pinn` |
| Windows | `pinn-windows-amd64.exe` | nothing — just rename it to `pinn.exe` if you like |
| Linux | `pinn-linux-amd64` | `chmod +x pinn-linux-amd64 && mv pinn-linux-amd64 pinn` |

No release yet, or you cloned the repository? Build it yourself with `make pinn` (needs
Go 1.25+ and Node) and use `dist/bin/pinn` wherever you see `./pinn` below.

Run everything below from the folder where the `pinn` file is. On Windows, type
`.\pinn.exe` wherever you see `./pinn`.

### 1. Read the book

```bash
./pinn serve
```

Your browser opens by itself at `http://127.0.0.1:8000` with the full learning path —
no internet needed, nothing to install. The terminal keeps running while you read; press
**Ctrl-C** there when you are done.

Prefer not to run anything? The same book is online at
**https://mariotoffia.github.io/pinn/**.

### 2. Run the starter scripts

```bash
./pinn run 00
```

That runs script number 00, the environment check. The **first run downloads Python
packages once** (PyTorch — a few hundred MB, so a few minutes). Every run after that
starts instantly. At the end you should see:

```
VERDICT: train PINNs on cpu in torch.float64
```

All eight scripts, in course order — each row is: type it, wait, check the number.

| Type this | What it proves | Takes | You should see roughly |
|---|---|---:|---|
| `./pinn run 00` | what *your* machine can do for PINN training | 10 s | the verdict line above |
| `./pinn run 01` | backprop written by hand works | 15 s | gradient check ≈ 3e-08 |
| `./pinn run 02` | autodiff can build a PDE residual | 15 s | autodiff vs finite differences ≈ 2e-06 |
| `./pinn run 03` | plain networks miss fast wiggles | 40 s | 0.05 plain vs 0.3000 with Fourier features |
| `./pinn run 04` | your first PINN (an oscillator) | 4 min | hard IC 1.6e-04 beats soft IC 4.8e-04 |
| `./pinn run 05` | your first PDE (heat equation) | 2 min | error ≈ 2.2e-04 |
| `./pinn run 06 --plain` | the classic Burgers benchmark | 4 min | error ≈ 1.4e-02 (then try `--all`: ≈ 4.7e-03) |
| `./pinn run 07` | recovering physics from noisy data | 2 min | ν recovered to 0.055 % |

Your numbers will differ a little; the *ordering* should not (`starter/README.md` explains
each one). Every script also saves its plots to `pinn-work/starter/out/` — open them and
compare with the reference images in `pinn-work/starter/examples/`.

The first time you run anything, `pinn` creates a folder called **`pinn-work/`** next to
itself, holding the scripts and notebooks as normal files. That folder is yours: edit
anything; `pinn` never overwrites a file that exists.

### 3. Open the labs

```bash
./pinn lab 02
```

A browser tab opens with an interactive notebook: text, code, sliders. Drag a slider and
everything below it re-runs. You cannot get stuck — every exercise already contains a
working solution. The first lab also downloads packages once. Press **Ctrl-C** in the
terminal to close it.

| Type this | The lab | Pairs with |
|---|---|---|
| `./pinn lab 02` | compute a PDE residual with autodiff | the chapter 03 tour |
| `./pinn lab 03` | watch spectral bias happen, then fix it | the chapter 05 tour |
| `./pinn lab 04` | train a real PINN: soft vs hard constraints | the chapter 08 tour |

---

## The same, without the `pinn` program

Cloned the repository instead? You need **uv** once (a Python tool manager, one line):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from the repository folder:

| To do this | Type this |
|---|---|
| read the book | `make generate && make open` (needs Node) — or read it online |
| run script 00 | `cd starter && uv run scripts/00_check_environment.py` |
| run script 06 | `cd starter && uv run scripts/06_pinn_burgers.py --plain` |
| open lab 02 | `uvx marimo edit --sandbox notebooks/02_autodiff_lab.py` |

Same behaviour: the first run downloads packages once, after that everything is instant.

---

## The labs inside the book (no install — not even pinn)

The chapters link about fifty **"open in Colab"** notebooks: university labs and official
library tutorials that run on Google's computers, inside your browser. Wherever you meet one
while reading:

1. **Click the link.** It opens in Google Colab (you need a free Google account).
2. Choose **Runtime ▸ Run all**. That is it — the code runs on Google's machine, not yours.
3. If a notebook benefits from a GPU, take one for free: **Runtime ▸ Change runtime type**.

Two notes: a Colab session forgets everything when it disconnects — use
*File ▸ Save a copy in Drive* if you edited something you want to keep — and the biggest
concentrations of these labs are in chapters 07 (solve it classically), 08 (your first PINNs)
and 12–13 (worked examples and neural operators). Chapter 18 collects every one of them in a
single index, with a one-line summary each.

---

## When something goes wrong

- **Mac refuses to run it** ("cannot be opened — unidentified developer"). Right-click the
  `pinn` file → Open → Open. Or in the terminal: `xattr -d com.apple.quarantine ./pinn`.
  That is macOS being cautious about downloaded programs.
- **`./pinn serve` says the address is in use.** Something else is on port 8000. Type
  `./pinn serve -addr 127.0.0.1:0` and it picks a free port for you.
- **The first script or lab takes ages.** That is the one-time PyTorch download. One
  Linux-only note: the `uvx marimo edit --sandbox` route fetches PyPI's standard Linux
  PyTorch, which is several GB (it still runs fine on CPU); `./pinn lab` avoids that by
  selecting the small CPU-only build automatically.
- **A lab tab stops responding.** Close the tab, press Ctrl-C in the terminal, run the
  same command again.
- **You broke a file in `pinn-work/` and want the original.** Delete (or rename) that one
  file and run `./pinn init` — it restores missing files and never touches existing ones.
