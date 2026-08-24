# From Neural Networks to Physics-Informed Neural Networks

[![CI](https://github.com/mariotoffia/pinn/actions/workflows/ci.yml/badge.svg)](https://github.com/mariotoffia/pinn/actions/workflows/ci.yml)
[![Pages](https://github.com/mariotoffia/pinn/actions/workflows/pages.yml/badge.svg)](https://github.com/mariotoffia/pinn/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Dependencies](https://img.shields.io/badge/build%20dependencies-none-brightgreen)

**Read it now: <https://mariotoffia.github.io/pinn/>** — or clone and build it offline in one
command, below.

**Just want to run things?** → **[RUNNING.md](RUNNING.md)** — the plain-English, copy-paste
guide to the book, every starter script and every lab.

A self-contained learning path, written in plain English for a student starting from
*"I can program, and I once took a linear algebra course"* and ending at
*"I can build, train, debug and judge a PINN."*

**No special hardware needed.** Everything runs locally on **CPU + float64** — a configuration
every machine has: Mac, Windows or Linux, with or without a GPU — with **Colab/Kaggle** for the
rare job that needs one. Chapter 00 states the hardware requirements; chapter 04 maps every
kind of machine (Apple Silicon, Windows, Linux, NVIDIA/AMD/Intel GPU, browser-only) to its
lane. Everything is free unless marked `[paid]`. Every link was verified live in
**August 2026**.

```bash
make generate     # build dist/index.html from every chapter + every TS/JS source
make open         # ...and open it
```

No `make`? The build is plain Node, so this is the whole story on any platform:

```bash
node tools/build.mjs                      # or: npm run build
```

---

## No clone, no setup: the `pinn` binary

One zip per OS on the [Releases page](https://github.com/mariotoffia/pinn/releases) —
download, unzip, done:

```bash
unzip pinn-darwin-arm64.zip     # or -linux-amd64, -windows-amd64, …
xattr -d com.apple.quarantine ./pinn   # macOS only: the binary is unsigned
                                       # (Windows: SmartScreen → More info → Run anyway)

./pinn serve    # the whole learning path at http://127.0.0.1:8000 — offline, embedded
./pinn init     # materialise starter/ + notebooks/ into ./pinn-work
./pinn run 00   # starter scripts, run through uv (bootstrapped automatically)
./pinn lab 02   # marimo labs in your browser — CPU torch auto-installed once, cached
```

The binary embeds the hub, the starter kit and the labs. The one thing it cannot embed is
PyTorch, so `pinn run` / `pinn lab` hand dependency resolution to
[uv](https://docs.astral.sh/uv/) on first use — and both pin the **CPU-only** PyTorch wheel
index on Linux/Windows (the starter's `pyproject.toml` carries the pin, `pinn lab` sets it
for the notebook sandboxes), so you never accidentally pull the multi-GB CUDA build. After
that one download everything runs offline.

Every archive carries the binary, the licence and `RUNNING.md`, and each release lists
`SHA256SUMS.txt` so you can check what you downloaded.

Build it yourself: `make pinn` (needs Go ≥ 1.25 and Node; output in `dist/bin/`), or
`make package` to produce the full set of release zips.

---

## What is here

| | |
|---|---|
| **`content/`** | 18 markdown chapters, ~43,000 words, ~500 unique verified external links — ~50 of them one-click Colab labs |
| **`src/`** | The hub app in TypeScript — router, search, progress, resource filter, **guided tours** |
| **`tools/`** | The zero-dependency build: markdown renderer, LaTeX renderer, type stripper — plus `experiments/`, the scripts behind the tour's measured data |
| **`starter/`** | A runnable Python kit: NumPy backprop → autodiff → PINNs → inverse problems |
| **`notebooks/`** | Interactive **marimo labs** pairing with the tours (`uvx marimo edit --sandbox notebooks/...`) |
| **`cmd/pinn`** | The **single binary**: serves the hub and extracts + runs the kit and labs via uv |
| **`dist/`** | Build output (git-ignored). `dist/index.html` is one self-contained file that works offline, from `file://` |
| **`.github/`** | CI: cross-platform build, reproducibility check, lab execution, monthly link check, Pages deploy |

Nothing is fetched at runtime. No CDN, no web fonts, no network. The generated page opens on a
plane.

---

## The chapters

| # | Chapter | Why |
|---|---|---|
| 00 | Start Here | What the path is, and the hardware requirements that decide your setup |
| 01 | Math Refresher | The minimum that matters — and a CUT LIST of what to skip |
| 02 | Neural Networks from First Principles | Backprop by hand, then micrograd |
| 03 | **Automatic Differentiation** | **The hinge chapter. Everything after depends on `grad(grad(u))`** |
| 04 | Environment Setup | CPU, float64, Colab — a lane for every machine, and why consumer GPUs cannot run a PINN |
| 05 | The Craft of Training | Optimisers, L-BFGS, loss balancing, spectral bias, NTK |
| 06 | Reinforcement Learning | Zero to competent — and how much of it PINNs need: none |
| 07 | PDE Primer and Classical Baselines | What you are competing with, and how to generate ground truth |
| 08 | PINN Core | The method, stated precisely, with Burgers written out in full |
| 09 | **Why PINNs Fail** | **The most important chapter** |
| 10 | The PINN Recipe | A 10-step checklist with defaults you can apply today |
| 11 | Frameworks and Tools | What to install, and what is quietly dead |
| 12 | Worked Examples | The verified DeepXDE gallery, by PDE, with line counts |
| 13 | Neural Operators | DeepONet, FNO, foundation models — and the 2026 verdict on KANs |
| 14 | The 2026 Reality Check | When to use what, what is deployed, what to disbelieve |
| 15 | Reading List | The 12 papers to read in full, in order |
| 16 | Roadmap | A 14-week plan with weekly deliverables |
| 17 | Resource Index | Every link, in one filterable table |

---

## Make targets

```
make generate    build dist/index.html from content/*.md + src/*.ts + src/*.js
make open        build and open it in your browser
make serve       build and serve on http://localhost:8000
make watch       rebuild on every change
make check       syntax-check the tools, build, and verify the output
make ci          what CI runs: check + prove the build is byte-reproducible
make typecheck   run tsc over src/ (optional; needs npx + typescript)
make stats       chapter, word and link counts
make links       print every unique external URL
make pinn        build the single-binary launcher into dist/bin/ (needs Go 1.25+)
make pinn-all    cross-compile the launcher for macOS / Linux / Windows
make dev         create .venv at the repo root (starter kit + marimo + ruff) for your editor
make setup       create the starter-kit's own virtualenv, inside starter/
make run         run the starter-kit environment check
make clean       remove dist/
```

### How the build works

`tools/build.mjs` reads every `content/*.md`, renders it to HTML at build time
(`tools/markdown.mjs` + `tools/latex.mjs`), reads every `src/*.ts` and `src/*.js`, strips the
TypeScript annotations (`tools/striptypes.mjs`, or **esbuild** when it is resolvable),
concatenates everything into one IIFE, validates it with `node --check`, and inlines the
result — together with the CSS and the chapter data — into a single `dist/index.html`.

**No `npm install` required.** If you do want the faster, stricter path:

```bash
npm i -D esbuild typescript      # optional
make generate                    # now uses esbuild automatically
make typecheck                   # now runs tsc
```

If the built-in stripper ever produces invalid JavaScript, the build **fails with the Node
syntax error** instead of writing a broken page.

### Editing

- **Add a chapter:** drop a numbered `.md` file into `content/` with front matter
  (`title`, `subtitle`, `minutes`) and run `make generate`. It appears in the navigation
  automatically.
- **Change the app:** edit `src/*.ts` and run `make generate`.
- **Change the look:** edit `src/styles.css`.

---

## Python environment

The hub needs no Python at all. The **labs** and the **starter kit** do — one command sets up
the environment your editor should use:

```bash
make dev            # creates .venv/ with the starter kit (editable), marimo and ruff
```

VS Code picks `.venv` up automatically (reload the window, or *Python: Select Interpreter*).
That is also what silences the Ruff and Pylance warnings about a missing interpreter: linting
rules live in [`pyproject.toml`](pyproject.toml) at the repo root, so everyone gets the same
result. `make setup` still exists and does something different — it builds the starter kit's
own virtualenv *inside* `starter/`, which is what chapters 04 and 12 tell readers to do.

Then, for the interactive labs:

```bash
uvx marimo edit --sandbox notebooks/02_autodiff_lab.py     # or: .venv/bin/marimo edit ...
```

## Working in VS Code

`.vscode/` is committed on purpose — open the folder and the setup is already done:

1. **Open the repo** (`code .` or *File ▸ Open Folder*). VS Code offers the recommended
   extensions — accept. The list ([`.vscode/extensions.json`](.vscode/extensions.json)) is
   short and deliberate: Python, **Ruff** (the same rules CI runs, from `pyproject.toml`),
   **[marimo](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)**
   for the labs, YAML + GitHub Actions for the workflows, EditorConfig.
2. **Environment** — run `make dev` once, reload the window, done: `settings.json` already
   points Python and Ruff at `.venv/`.
3. **Build** — `Cmd/Ctrl+Shift+B` runs the default task (`node tools/build.mjs` →
   `dist/index.html`). *Terminal ▸ Run Task…* also has **open hub in browser** and
   **check build tools**.
4. **Labs, two ways** — *Run Task…* ▸ `lab: autodiff (marimo)` (or *spectral bias* /
   *oscillator PINN*) launches `uvx marimo edit` in a terminal; or open any
   `notebooks/*_lab.py` and start it from the marimo extension's editor button to run the
   same notebook inside VS Code.
5. **Debug** — *Run and Debug* ships two configs: **Debug hub build** (Node — step through
   the markdown/TS pipeline) and **Debug current Python file** (debugpy, uses `.venv`).

---

## The hub

- **Guided tours** — Brilliant-style stepped lessons on chapters 02, 03, 05, 08 and 09, with
  live animations that run in the page: a real PINN training in your browser, spectral bias
  happening before your eyes, gradient descent in an ill-conditioned valley, and playback of
  **measured** failure/rescue curves (vanilla β=30: 91% error; time-marched: 2.5% — produced
  by `tools/experiments/`). Quizzes never block: Next always works. Each tour ends in a
  hands-on lab: a local [marimo](https://marimo.io/) notebook from `notebooks/`, or a
  **one-click Colab** notebook that opens ready to run.
- **Search** the whole path (`/` or the Search button) — try `float64`, `L-BFGS`, `causal`,
  `weak baselines`.
- **Filter** all ~240 resource rows live on chapter 17.
- **Progress** — mark chapters as done; stored per browser in `localStorage`, wrapped so a
  browser that blocks storage still renders correctly.
- **Keyboard**: `/` search · `[` `]` previous/next chapter · `t` theme · `Esc` close.
- **Theme**: auto / light / dark.

---

## The starter kit

```bash
make setup                                   # or: cd starter && uv venv && uv pip install -e .
cd starter
python scripts/00_check_environment.py
```

Eight progressive scripts, all CPU, all float64. See `starter/README.md` for the measured
runtimes and the results you should expect — including the **Burgers ablation table**, the
single most valuable artifact of the whole path.

---

## The one-paragraph version of the field

A PINN uses a smooth neural network as a stand-in for the solution of a PDE, computes the
equation's derivatives exactly with automatic differentiation, and minimises the squared
equation error at randomly sampled points. The equation becomes an endless supply of free
training labels. For **inverse problems, data assimilation, awkward geometry and
high-dimensional PDEs**, that is a real capability that classical solvers do not offer
cheaply. As a *forward solver*, on problems a finite element method can handle, PINNs lose —
usually by orders of magnitude — and they lose **silently**: the loss goes down, the plot
looks smooth, the answer is wrong. Learn them for what they are good at, and learn the failure
modes at the same time as the method.

---

## Contributing

Corrections are welcome — especially **dead links** and **claims that have aged**.
The bar is stated in [CONTRIBUTING.md](CONTRIBUTING.md): every link checked before it is added,
every number measured or cited, plain English, and the skeptical stance kept intact.

If you touch `src/`, read the short "TypeScript subset" table in that file first — the build has
no dependencies, so a hand-written stripper has to understand every construct used.

## License

[Apache-2.0](LICENSE) — code, curriculum text, labs and all. See [NOTICE](NOTICE).

The path links to and briefly quotes third-party papers, libraries and courses; those remain the
property of their authors under their own licences, and none of them are redistributed here.
