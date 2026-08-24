---
title: Neural Networks from First Principles
subtitle: Perceptron to backprop, derived by hand — then the two architecture ideas PINNs depend on
minutes: 23
---

# 02 — Neural Networks from First Principles

This chapter has one narrow, non-negotiable goal: **derive backpropagation by hand, then build a
tiny autograd engine from nothing.** ("Backpropagation", or backprop, is the algorithm that
computes how much each network weight contributed to the error. An "autograd engine" is the piece
of software that automates it.) If you skip this, PINNs will stay a black box forever — the whole
technique is a second, unusual use of the chain rule, and framework tutorials never explain it.

The second goal: **learn MLPs deeply, and everything else lightly.** An MLP (multi-layer
perceptron) is the plainest kind of neural network — layers of weights with a simple nonlinear
function between them. Nearly every PINN is a plain MLP: typically 3–6 hidden layers of 128–512
units with `tanh`, mapping coordinates `(x, t)` to a field value like temperature. CNNs, RNNs
and Transformers get a light pass here — just enough to read neural-operator papers later.

**Target: about 26 hours** for the core.

---

## 2.1 The visual first pass

- **3Blue1Brown — Neural Networks** — `video` `free` — https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
  The playlist has grown to 10 videos (LLM and attention chapters were added through 2026).
  **Watch videos 1–4:** what a network is, gradient descent, what backprop is really doing, and
  the backprop calculus. **~2 h.** The rest belongs to Section 2.6.

- **StatQuest — Neural Networks / Deep Learning (Josh Starmer)** — `video` `free` —
  https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1
  A gentler on-ramp than 3Blue1Brown: tiny steps, everything spelled out, almost no notation.
  If 3Blue1Brown moves too fast, start here and come back. If 3Blue1Brown felt fine, skip this.

- **TensorFlow Playground** — `interactive` `free` — https://playground.tensorflow.org/
  Do one specific exercise: switch to **Regression** mode and flip between `tanh` and `ReLU`
  while watching how *smooth* the learned function is. **That smoothness difference is exactly
  why PINNs use tanh.** ~1 h.

---

## 2.2 Backprop, derived by hand — the core of this chapter

- **Michael Nielsen — Neural Networks and Deep Learning** — `book` `free` — http://neuralnetworksanddeeplearning.com/
  Still the best written derivation of backprop anywhere, with runnable NumPy code for MNIST.
  Last substantive update 2019 — and nothing in Ch. 1–3 has aged.
  **Read Ch. 1** (perceptrons → sigmoid neurons → the MNIST network), **Ch. 2 with a pen in
  your hand** (the four fundamental equations of backprop, derived by you), **Ch. 3**
  (cross-entropy, softmax, regularisation, initialisation). **~7 h.**
  **Read Ch. 4 for pleasure** — ["A visual proof that neural nets can compute any function"](http://neuralnetworksanddeeplearning.com/chap4.html)
  is the "universal approximation" intuition that every PINN paper quietly leans on.
  **Skip Ch. 5–6** (dated framing; covered better elsewhere).

- **Andrej Karpathy — Neural Networks: Zero to Hero** — `course` `free` — https://karpathy.ai/zero-to-hero.html
  Playlist: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
  **Lecture 1, "building micrograd"** (https://youtu.be/VMj-3S1tku0, 2h25m) builds a working
  reverse-mode autodiff engine from nothing. This is *the* thing to internalise before PINNs,
  because a PINN is autodiff applied a second time — to the inputs instead of the weights.
  **~3 h, typing every line yourself.**
  **Lecture 4, "Becoming a Backprop Ninja"** (https://youtu.be/q8SA3rM6ckI) — hand-write every
  backward pass. **~3 h.**
  **Lecture 3, "Activations & Gradients, BatchNorm"** (https://youtu.be/P6sfmUTpUmc) is the best
  free material anywhere on why deep networks fail to train — directly relevant to why PINN
  training is hard. **Skip Lectures 6–8** (WaveNet, GPT, tokeniser) unless you are heading
  toward neural operators.
  Notebooks: https://github.com/karpathy/nn-zero-to-hero

- **karpathy/micrograd** — `repo` `free` — https://github.com/karpathy/micrograd
  About 100 lines of scalar autograd plus about 50 lines of a PyTorch-like API. **Read
  `engine.py` end to end in one sitting.** It will permanently demystify `loss.backward()`.
  **~1 h.**

- **CS231n — Backpropagation notes** — `notes` `free` — https://cs231n.github.io/optimization-2/
  Backprop explained as "local gradients flowing through gates", staged computation, and the
  vectorised matrix-derivative section. This is what makes hand-derived gradients fast instead
  of terrifying. **~2 h.**
  Companions: https://cs231n.github.io/optimization-1/ and https://cs231n.github.io/neural-networks-1/
  — loss landscapes, numerical vs analytic gradients, and **gradient checking**, which you
  *will* need on the day your PDE residual is wrong.

- **Foundations of Computer Vision, Ch. 14: Backpropagation (Torralba, Isola, Freeman)** —
  `book chapter` `free` — https://visionbook.mit.edu/backpropagation.html
  MIT Press 2024, free to read online. The cleanest modern treatment of backprop as dynamic
  programming over a computation graph — modular layers, branching graphs, and, importantly,
  **using backprop to optimise the *input* rather than the weights.** That last idea is one
  small step from how PINNs differentiate with respect to coordinates. **~2 h.**

- **Backprop Explainer** — `interactive` `free` — https://xnought.github.io/backprop-explainer/
  Step through forward and backward passes with live gradient values. Use it after Nielsen
  Ch. 2 to confirm your hand derivation matches. ~1 h.

---

## 2.3 One primary book, not four

You need exactly one modern main textbook. It should be **Prince**.

- **Understanding Deep Learning (Simon J.D. Prince)** — `book` `free PDF` — https://udlbook.github.io/udlbook/
  Repo with notebooks and slides: https://github.com/udlbook/udlbook
  The companion notebooks open **one-click in Colab**; for this chapter do
  [7.1 — backprop in a toy model](https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap07/7_1_Backpropagation_in_Toy_Model.ipynb "Fill-in-the-blank: hand-compute every derivative of a toy model")
  (fill-in-the-blank: you hand-compute every derivative) and
  [7.2 — backpropagation](https://colab.research.google.com/github/udlbook/udlbook/blob/main/Notebooks/Chap07/7_2_Backpropagation.ipynb "The full backprop algorithm, built on the 7.1 toy model").
  (CC BY-NC-ND: run them, do not redistribute modified copies.)
  Actively maintained — **v5.0.3, February 2026.** The PDF is free; the MIT Press print edition
  is paid.
  **Reading order for this path:** Ch. 2 → **Ch. 3 (shallow networks)** → **Ch. 4 (deep
  networks)** → Ch. 5 (loss functions) → Ch. 6 (fitting models) → **Ch. 7 (gradients and
  initialisation)** → Ch. 8, 9.
  **Ch. 3, 4 and 7 are the core for you** — Prince's analysis of what extra depth actually buys
  you is the best available. **~10 h.**
  Skim Ch. 10 (CNNs) and 12 (transformers). Skip Ch. 14–19. Read Ch. 20 ("why does deep
  learning work?") late.

- **Deep Learning: Foundations and Concepts (Bishop & Bishop)** — `book` `free online reader / paid print` —
  https://www.bishopbook.com/
  A free online version is on the authors' site; the hardback and the Springer PDF are paid.
  Springer's best-selling book of both 2024 and 2025. Errata and free exercise solutions for
  Ch. 2–10 are on the same page. **Use it as a lookup reference**, not cover to cover.
  ~10 h targeted.

- **Dive into Deep Learning (d2l.ai)** — `book/interactive` `free` — https://d2l.ai/
  Every section is an executable notebook, with parallel **PyTorch, JAX, TensorFlow and NumPy**
  code. **Run:** Ch. 2 (Preliminaries — including the calculus and automatic-differentiation
  sections), Ch. 3, **Ch. 5 (Multilayer Perceptrons)**, Ch. 12 (Optimization Algorithms). ~8 h.
  Pick the JAX track if you lean toward JAX-based PINN libraries.

- **Deep Learning (Goodfellow, Bengio & Courville)** — `book` `free HTML` — https://www.deeplearningbook.org/
  MIT Press 2016, frozen by the authors' own statement — and it shows its age. Still worth
  three chapters: **Ch. 6 (Deep Feedforward Networks)** — still the best formal treatment of
  universal approximation and the depth argument — **Ch. 8 (Optimization)**, and **Ch. 4
  (Numerical Computation: conditioning and ill-conditioned Hessians)**, which describes exactly
  the disease that afflicts PINN training. **Skip all of Part III.** ~6 h.

---

## 2.4 The famous courses — and which parts to take

- **MIT 6.S191: Introduction to Deep Learning** — `course` `free` — http://introtodeeplearning.com/
  **The 2026 edition ran March–May 2026** and is the best-maintained free intro course
  anywhere: nine lectures, all on YouTube, all slides public, three Colab labs. For this path,
  note that **Lecture 8 is "AI for Science"** — a rare direct on-ramp to exactly where you are
  heading. Lecture 9 covers large-scale parallel training.
  Playlist: https://www.youtube.com/playlist?list=PLkkuNyzb8LmxFutYuPA7B4oiMn6cjD6Rs
  Labs: https://github.com/MITDeepLearning/introtodeeplearning — Lab 1's PyTorch part opens
  straight in Colab: [PT_Part1_Intro.ipynb](https://colab.research.google.com/github/MITDeepLearning/introtodeeplearning/blob/master/lab1/PT_Part1_Intro.ipynb "MIT 6.S191 Lab 1: tensors, a dense layer from scratch, then autograd — in under an hour")
  — tensors → a dense layer from scratch → autograd, in under an hour.

- **UvA Deep Learning Tutorials (University of Amsterdam)** — `interactive` `free` —
  https://uvadlc-notebooks.readthedocs.io/en/latest/
  **Current: DL1 Fall 2025 and DL2 Spring 2026** — explicitly kept up to date. Notebooks in
  **both PyTorch (+Lightning) and JAX/Flax**, every one with a Colab badge and a YouTube
  recording. The best free notebook series in deep learning.
  **Priority: Tutorial 3 (Activation Functions) and Tutorial 4 (Optimization and
  Initialization).** Both matter directly for PINNs. ~6 h.

- **Stanford CS231n** — `course` `free notes, videos partly paywalled` — https://cs231n.stanford.edu/
  Spring 2026 is current; the slides are public but current lecture videos are
  students-only. The **notes at https://cs231n.github.io/ remain free** and are what you want.
  The only fully *public* video set is
  [Spring 2017](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv);
  Lecture 4 (backprop) and Lecture 6 (training neural networks I) are still excellent.

- **Stanford CS230** — `course` `free materials` — https://cs230.stanford.edu/
  The public video archive is [Autumn 2018](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb).
  **Lecture 8, "Career Advice / Reading Research Papers," is the most useful 80 minutes on this
  entire path** for someone whose goal is "read a PINN paper." Watch it before you open your
  first one. Cheat sheets (print the "Tips and Tricks" one): https://stanford.edu/~shervine/teaching/cs-230/

- **NYU Deep Learning (LeCun & Canziani)** — `course` `free` — https://atcold.github.io/NYU-DLFL22/
  (2021 edition with full written notes: https://atcold.github.io/NYU-DLSP21/). Built around
  **energy-based models** — an unusual and useful point of view: learning as minimising an
  energy is a close cousin of the variational PINN formulations (Deep Ritz). A reference
  library, not a study plan. One notebook *is* worth doing here, though:
  [spiral classification](https://colab.research.google.com/github/Atcold/NYU-DLSP21/blob/master/04-spiral_classification.ipynb "Train the same tiny net twice (linear, then ReLU) and watch the decision boundary bend — fifteen minutes")
  — train the same tiny net twice (linear, then ReLU) and watch the decision boundary bend.
  Fifteen minutes, and "what a hidden layer does" stops being abstract. (CC BY-NC-SA.)

- **Deep Learning Specialization, Course 1 — free on YouTube** — `video` `free` —
  https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0
  43 videos on the official DeepLearningAI channel covering "Neural Networks and Deep
  Learning." This gets you Andrew Ng's excellent forward/backward-propagation lectures
  **without paying for the full specialization** (which is `[paid]`, ~127 h, and last refreshed
  in April 2021).

---

## 2.5 The two architecture ideas PINNs depend on

### Activation functions — they matter more here than anywhere else in ML

The activation function is the nonlinearity between layers. A PINN loss contains `∂²u/∂x²` — a
second derivative. **ReLU's second derivative is zero almost everywhere**, so with ReLU the
physics term literally reads zero and teaches the network nothing. That is why the literature
defaults to `tanh`, and why the interesting alternatives are smooth ones: SiLU/GELU, and
**sine**, which can be differentiated forever without dying out.

- **UvA Tutorial 3: Activation Functions** — `interactive` `free` —
  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
  Implements and compares sigmoid, tanh, ReLU, LeakyReLU, ELU and Swish/SiLU, with
  **visualisations of gradient flow across layers and a count of dead neurons.** Read it while
  keeping the second-derivative question in mind. ~3 h.
- **SIREN: Implicit Neural Representations with Periodic Activation Functions** — `paper` `free` —
  https://www.vincentsitzmann.com/siren/ · https://arxiv.org/abs/2006.09661 · repo https://github.com/vsitzmann/siren
  Sitzmann et al., NeurIPS 2020 (oral). Networks with sine activations can **accurately
  represent their own first and second derivatives**, which ReLU and tanh MLPs cannot. The
  project page says this in the context of **solving PDEs** — this paper is a
  prerequisite for the modern PINN literature. Run the Poisson and
  Helmholtz demos: they are PINNs in everything but the name. **~5 h with the Colab.**

### Fourier features and spectral bias — the most-cited reason PINNs fail

"Spectral bias" means: neural networks learn the smooth, slowly-varying parts of a function
first, and the fine, fast-wiggling parts last — or never. The fix below appears in almost every
modern PINN.

- **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains** —
  `paper` `free` — https://bmild.github.io/fourfeat/ · https://arxiv.org/abs/2006.10739 ·
  repo https://github.com/tancik/fourier-feature-networks
  Tancik et al., NeurIPS 2020. Establishes spectral bias using neural-tangent-kernel analysis,
  and fixes it by first passing the inputs through a random sine/cosine mapping
  `γ(x) = [cos(Bx), sin(Bx)]ᵀ` with `Bᵢⱼ ~ N(0, σ²)`. The mapping gives you a **tunable knob
  (σ)** for how much high-frequency detail the network can learn. **~4 h with the Colab.**
  Run the included 1D/2D fitting demo with and without the mapping. Thirty minutes that make
  spectral bias permanently intuitive.

Chapter 05 covers the theory behind this (NTK, F-principle); Chapter 10 turns both ideas into
checklist items.

---

## 2.6 CNNs, RNNs, Transformers — the light pass

Just enough to read neural-operator papers. Do not build a career here.

- **CNN Explainer (Georgia Tech PoloClub)** — `interactive` `free` — https://poloclub.github.io/cnn-explainer/
  Interactive convolution/pooling walkthrough on a live model. The FNO (Chapter 13) borrows the
  "learned filter over a grid" idea, not the vision architecture zoo. ~1 h.
  Plus the convolution arithmetic in https://cs231n.github.io/convolutional-networks/ (~1 h).
- **The Illustrated Transformer (Jay Alammar)** — `article` `free` — https://jalammar.github.io/illustrated-transformer/
  Published 2018, updated 2025. The **positional encoding** section connects straight back to
  Fourier features. ~2 h.
- **Transformer Explainer (PoloClub)** — `interactive` `free` — https://poloclub.github.io/transformer-explainer/
  Runs a live GPT-2 small **in your browser** with interactive attention maps. ~1 h.
- **The Unreasonable Effectiveness of RNNs (Karpathy, 2015)** — `article` `free` —
  https://karpathy.github.io/2015/05/21/rnn-effectiveness/ — read it as *history*: what
  sequence models were before attention. ~1 h.

---

## 2.7 Visual tools worth 20–30 minutes each

- **NN-SVG** — https://alexlenail.me/NN-SVG/ — publication-ready architecture diagrams with SVG
  export. You will want this the first time you write up a PINN architecture.
- **Distill.pub archive** — https://distill.pub/ — the journal has been on pause since 2021, but
  the whole archive is free. Read **["Momentum: Why Momentum Really Works"](https://distill.pub/2017/momentum/)**
  — the best explanation anywhere of the conditioning problem that makes PINN losses hard to
  optimise. Also ["Feature Visualization"](https://distill.pub/2017/feature-visualization/)
  and, if you later use graph-based operators,
  ["Understanding Convolutions on Graphs"](https://distill.pub/2021/understanding-gnns/).
- **Welch Labs — Why Deep Learning Works Unreasonably Well** — https://www.youtube.com/watch?v=qx7hirqgfuU
  — a beautifully produced 2024 video on why hugely over-parameterised networks still
  generalise. An angle nothing else on this path covers, in half an hour.
- **LLM Visualization (Brendan Bycroft)** — https://bbycroft.net/llm — a 3D animated walkthrough
  of every matrix multiply in a transformer. Instructive about tensor shapes.
- **Spreadsheets Are All You Need** — https://spreadsheets-are-all-you-need.ai/ — GPT-2
  implemented in spreadsheet formulas; free browser version and Excel downloads. Every
  operation is a cell you can click. The anti-black-box. (The companion course is `[paid]`;
  the free material is the valuable part.)
- **Machine Learning Tokyo — Interactive Tools** — https://github.com/Machine-Learning-Tokyo/Interactive_Tools
  — a curated index of interactive ML/DL/math tools. Use as a directory.
- **A Visual Introduction to Machine Learning (R2D3)** — http://www.r2d3.us/visual-intro-to-machine-learning-part-1/
  — the clearest visual account of the bias–variance tradeoff you will find.

---

## CUT LIST

- **fast.ai Practical Deep Learning, in full.** Superb teaching, wrong direction for you. It
  optimises for fine-tuning large pretrained models on real datasets, fast. PINNs are tiny
  custom networks trained from scratch against a physics residual, with *no dataset at all*.
  Its top-down "train first, understand later" style actively conflicts with a path whose whole
  point is understanding the derivatives. **Take Lesson 3 and Lesson 5 of
  [Part 1](https://course.fast.ai/) and the from-scratch matmul/backprop lessons of
  [Part 2](https://course.fast.ai/Lessons/part2.html); leave the rest.**
- **DeepLearning.AI Deep Learning Specialization in full** `[paid]`. 127 hours and ~$25–30 per
  month for content last refreshed in April 2021, with Courses 3–5 almost entirely off this
  path. Watch Course 1 free on YouTube and spend the money on Bishop's hardback instead.
- **Goodfellow cover to cover.** From 2016, frozen, and Part III is now mostly a history of
  research directions that did not win. Its "deep learning bible" reputation is a decade out
  of date.
- **CS231n in full.** It is a *computer vision* course — object detection, video, 3D vision.
  Magnificent, and about 90% irrelevant to a coordinate-input MLP solving a PDE.
- **Learning TensorFlow.** The PINN world has settled on **PyTorch and JAX**. The original PINN
  code was TensorFlow 1, and you will occasionally *read* it. Do not learn the framework.
- **Anything that teaches backprop only through calling `.backward()`.** Nielsen Ch. 2 plus
  micrograd is the non-negotiable core of this chapter.
