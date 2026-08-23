# From Neural Networks to Physics-Informed Neural Networks

[![CI](https://github.com/mariotoffia/pinn/actions/workflows/ci.yml/badge.svg)](https://github.com/mariotoffia/pinn/actions/workflows/ci.yml)
[![Pages](https://github.com/mariotoffia/pinn/actions/workflows/pages.yml/badge.svg)](https://github.com/mariotoffia/pinn/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Dependencies](https://img.shields.io/badge/build%20dependencies-none-brightgreen)

**Read it now: <https://mariotoffia.github.io/pinn/>** — or clone and build it offline in one
command, below.

A self-contained learning path, written in plain English for a student starting from
*"I can program, and I once took a linear algebra course"* and ending at
*"I can build, train, debug and honestly judge a PINN."*

Written from an **Apple Silicon MacBook Pro** — **CPU-only local runs**, with **Colab/Kaggle**
for the rare job that needs a GPU — but not Mac-only: the core configuration (CPU, float64)
exists on every machine, and **Windows and Linux readers, with or without an NVIDIA GPU, are
fully covered** (chapter 04 §4.9). Everything is free unless marked `[paid]`. Every link was
verified live in **August 2026**.

```bash
make generate     # build index.html from every chapter + every TS/JS source
make open         # ...and open it
```

No `make`? The build is plain Node, so this is the whole story on any platform:

```bash
node tools/build.mjs --out index.html     # or: npm run build
```

---

## What is here

| | |
|---|---|
| **`content/`** | 18 markdown chapters, ~40,000 words, ~470 unique verified external links |
| **`src/`** | The hub app in TypeScript — router, search, progress, resource filter, **guided tours** |
| **`tools/`** | The zero-dependency build: markdown renderer, LaTeX renderer, type stripper — plus `experiments/`, the scripts behind the tour's measured data |
| **`starter/`** | A runnable Python kit: NumPy backprop → autodiff → PINNs → inverse problems |
| **`notebooks/`** | Interactive **marimo labs** pairing with the tours (`uvx marimo edit notebooks/...`) |
| **`index.html`** | Generated (git-ignored). One self-contained file that works offline, from `file://` |
| **`.github/`** | CI: cross-platform build, reproducibility check, lab execution, monthly link check, Pages deploy |

Nothing is fetched at runtime. No CDN, no web fonts, no network. The generated page opens on a
plane.

---

## The chapters

| # | Chapter | Why |
|---|---|---|
| 00 | Start Here | The honest framing, and four hardware facts that decide your whole setup |
| 01 | Math Refresher | The minimum that matters — and a CUT LIST of what to skip |
| 02 | Neural Networks from First Principles | Backprop by hand, then micrograd |
| 03 | **Automatic Differentiation** | **The hinge chapter. Everything after depends on `grad(grad(u))`** |
| 04 | Environment Setup | Apple Silicon, float64, Colab. Why MPS cannot run a PINN |
| 05 | The Craft of Training | Optimisers, L-BFGS, loss balancing, spectral bias, NTK |
| 06 | Reinforcement Learning | Zero to competent — and how much you actually need (spoiler: none, for PINNs) |
| 07 | PDE Primer and Classical Baselines | What you are competing with, and how to generate ground truth |
| 08 | PINN Core | The method, stated precisely, with Burgers written out in full |
| 09 | **Why PINNs Fail** | **The most important chapter** |
| 10 | The PINN Recipe | A 10-step checklist with defaults you can apply today |
| 11 | Frameworks and Tools | What to install, and what is quietly dead |
| 12 | Worked Examples | The verified DeepXDE gallery, by PDE, with line counts |
| 13 | Neural Operators | DeepONet, FNO, foundation models — and the 2026 verdict on KANs |
| 14 | The 2026 Reality Check | When to use what, what is deployed, what to disbelieve |
| 15 | Reading List | The 12 papers to actually read, in order |
| 16 | Roadmap | A 14-week plan with weekly deliverables |
| 17 | Resource Index | Every link, in one filterable table |

---

## Make targets

```
make generate    build index.html from content/*.md + src/*.ts + src/*.js
make open        build and open it in your browser
make serve       build and serve on http://localhost:8000
make watch       rebuild on every change
make check       syntax-check the tools, build, and verify the output
make ci          what CI runs: check + prove the build is byte-reproducible
make typecheck   run tsc over src/ (optional; needs npx + typescript)
make stats       chapter, word and link counts
make links       print every unique external URL
make setup       create the starter-kit virtualenv
make run         run the starter-kit environment check
make clean       remove index.html
```

### How the build works

`tools/build.mjs` reads every `content/*.md`, renders it to HTML at build time
(`tools/markdown.mjs` + `tools/latex.mjs`), reads every `src/*.ts` and `src/*.js`, strips the
TypeScript annotations (`tools/striptypes.mjs`, or **esbuild** when it is resolvable),
concatenates everything into one IIFE, validates it with `node --check`, and inlines the
result — together with the CSS and the chapter data — into a single `index.html`.

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

## The hub

- **Guided tours** — Brilliant-style stepped lessons on chapters 02, 03, 05, 08 and 09, with
  live animations that run in the page: a real PINN training in your browser, spectral bias
  happening before your eyes, gradient descent in an ill-conditioned valley, and playback of
  **measured** failure/rescue curves (vanilla β=30: 91% error; time-marched: 2.5% — produced
  by `tools/experiments/`). Quizzes never block: Next always works.
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

Corrections are genuinely welcome — especially **dead links** and **claims that have aged**.
The bar is stated in [CONTRIBUTING.md](CONTRIBUTING.md): every link checked before it is added,
every number measured or cited, plain English, and the skeptical stance kept intact.

If you touch `src/`, read the short "TypeScript subset" table in that file first — the build has
no dependencies, so a hand-written stripper has to understand every construct used.

## License

[Apache-2.0](LICENSE) — code, curriculum text, labs and all. See [NOTICE](NOTICE).

The path links to and briefly quotes third-party papers, libraries and courses; those remain the
property of their authors under their own licences, and none of them are redistributed here.
