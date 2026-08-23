# Contributing

Thanks for looking. This repository is a **learning path**, so the bar for changes is a little
unusual: correctness and honesty matter more than volume, and every external claim has to be
checkable.

## The one-minute version

```bash
git clone https://github.com/mariotoffia/pinn.git
cd pinn
make generate          # builds index.html - needs nothing but Node 18+
make open              # ...and opens it
```

There is **no install step and no backend**. `index.html` is a single self-contained file that
works from `file://`, offline.

## Ground rules for content

1. **Every external link must be checked before it is added**, and the note next to it must say
   what the reader gets. Dead or moved links are bugs — please open an issue with the
   "Broken or moved link" template.
2. **Numbers must be measured or cited.** The tours and the starter README quote results that
   were actually produced by the code in this repo (see `tools/experiments/` for the scripts
   behind the chapter 09 charts). If you add a number, say where it came from.
3. **Keep the editorial stance.** The path is deliberately skeptical: PINNs lose to classical
   solvers on forward problems, most published speedups compare against weak baselines, and the
   places PINNs genuinely win (inverse problems, data assimilation, high dimensions) are stated
   plainly. Please do not sand that down.
4. **Plain, simple English.** Short sentences. Every term explained the first time it appears.
   Same technical depth — simpler language, not simpler content.

## Working on the hub (`src/*.ts`)

The build has zero dependencies, which means `tools/striptypes.mjs` — a small hand-written
TypeScript stripper — has to understand every construct you use. **It handles a deliberate
subset.** Stay inside it:

| Do | Don't |
|---|---|
| `function f(x: string) {}` | `function f(cb: () => void) {}` — function-type annotations |
| `const x = y as HTMLAnchorElement` | `as const` |
| `let a; let b;` | `let a: T, b: U;` — annotated multi-declarators |
| `if (c) { x = 1 } else { x = 2 }` | a ternary whose true branch ends in `)` right before `:` |
| `const betaList = ...` | an identifier ending in `as` followed by a space (read as a cast) |

The bundle is concatenated into **one IIFE**, so top-level helper names must be unique across
all files in `src/` (that is why `touranim.ts` calls its DOM helper `mk` — `app.ts` already owns
`el`).

After any change: `make check` (syntax + build + `tools/verify.mjs`) and, if you have the
optional toolchain, `npm install && npm run typecheck` — which should stay at **zero errors**.

## Adding a chapter

Drop a numbered file into `content/` with front matter and run `make generate`:

```markdown
---
title: Short Title
subtitle: One line shown under the heading
minutes: 15
---
```

Navigation, search, the table of contents and the resource index pick it up automatically.
The markdown renderer is `tools/markdown.mjs` and supports exactly what the corpus uses — if
you need a construct it lacks, extend the renderer in the same PR and add a case to
`tools/verify.mjs`.

**One renderer gotcha worth knowing:** never write a URL containing raw parentheses (some DOIs
do). Percent-encode them (`%28` / `%29`) or the link will be truncated.

## Adding a guided tour

Tours are data: `src/tours.ts` holds the steps, `src/touranim.ts` the animations, and
`src/tourdata.ts` any measured data they replay. A step may carry a `quiz` (which must never
block progress — Next always works) and a `notebook` block, which is required to state three
things: **what it does**, **what you need first**, and **what success looks like**.

## Labs

The marimo labs in `notebooks/` must **always run**: the working solution lives in the cell and
the exercise is to replace it, with hints in fold-out accordions. CI executes every lab
end-to-end on Linux with CPU PyTorch, so a lab that only works on your machine will be caught.

```bash
uvx marimo edit notebooks/02_autodiff_lab.py
```

## Pull requests

- One topic per PR; small is good.
- Say what you verified, not just what you changed — CI runs the build, the reproducibility
  check, the typecheck, the Python checks, and (when labs change) the notebooks.
- `index.html` is a build artifact and is git-ignored. Never commit it.
