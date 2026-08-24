---
title: Reinforcement Learning
subtitle: Zero to competent — and how much of it a PINN person needs
minutes: 23
---

# 06 — Reinforcement Learning

**Read the last section of this chapter first.** It tells you how much of this chapter to do.

Short version: **you need exactly zero reinforcement learning to build, train or debug a PINN.**
Reinforcement learning (RL) is about an *agent* learning to act — try something, get a reward,
adjust. A PINN is nothing like that: it has a fixed objective, uses the full batch, is
deterministic, and is differentiable end to end. No exploration, no delayed rewards, no
environment. Where RL earns its place is **controlling** systems governed by PDEs —
a valuable, and different, job.

---

## 6.1 The book

- **Reinforcement Learning: An Introduction, 2nd ed. (Sutton & Barto)** — `book` `free` —
  http://incompleteideas.net/book/the-book-2nd.html ·
  **PDF: http://incompleteideas.net/book/RLbook2020.pdf** (~73 MB; trimmed-margin version:
  http://incompleteideas.net/book/RLbook2020trimmed.pdf)
  The 2020 printing of the 2018 MIT Press second edition, posted free by the authors.
  **Ch. 1–8 (bandits → MDPs → dynamic programming → Monte Carlo → TD learning → n-step →
  planning) are the non-negotiable core.** Ch. 9–13 (function approximation, policy gradients)
  take you to the doorstep of deep RL. The same page hosts errata and teaching slides.
- **Shangtong Zhang — companion code** — `repo` `free` —
  https://github.com/ShangtongZhang/reinforcement-learning-an-introduction — MIT-licensed
  Python that reproduces nearly every figure in the book, chapter by chapter. Run
  `chapter06/cliff_walking.py` while reading Ch. 6, and TD learning clicks.
- **Denny Britz — reinforcement-learning** — `repo` `free` —
  https://github.com/dennybritz/reinforcement-learning — exercise notebooks in "here is the
  skeleton, you fill in the algorithm" format, aligned to both Sutton & Barto and David
  Silver. Older and less maintained, but the fill-in-the-blank format teaches better than
  reading finished code.

---

## 6.2 Lecture series — which to start with in 2026

- **DeepMind × UCL — Introduction to RL (David Silver, 2015)** — `video` `free` —
  https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ ·
  slides https://davidstarsilver.wordpress.com/teaching/
  Ten lectures that track Sutton & Barto almost exactly. 90% of it is timeless.
- **DeepMind × UCL RL Lecture Series (van Hasselt, Borsa, Hessel, 2021)** — `video` `free` —
  lecture 1: https://www.youtube.com/watch?v=TCCjZe0y4Qc · slides mirror:
  https://github.com/yjavaherian/deepmind-x-ucl-rl
  13 lectures. **One caveat:** DeepMind's official short link for this series is dead and
  there is no official playlist on their channel. The individual videos are still up, and the
  slides mirror is the reliable index.

> **Which first?** **Start with Silver 2015; use van Hasselt 2021 as a second pass.** Silver's
> series pairs with Sutton & Barto's chapter order — read a chapter, watch the matching
> lecture — and that pairing is worth more than recency. The 2021 series is stronger on
> function approximation and deep RL (lectures 7, 11, 12, 13 are the modern content Silver
> predates), but it has become harder to find. Neither covers the last five years.

- **Stanford CS234 (Emma Brunskill)** — `course` `free` — https://web.stanford.edu/class/cs234/ ·
  latest public playlist (Spring 2024):
  https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX
  More theory-minded than CS285 (regret bounds, exploration, offline evaluation). The Winter
  2026 syllabus adds RLHF basics. **A better first university course than CS285.**
- **UC Berkeley CS 285 — Deep RL (Sergey Levine)** — `course` `free` —
  https://rail.eecs.berkeley.edu/deeprlcourse/ · latest public videos (Fall 2023):
  https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps
  Graduate level. **The homeworks are the best part, and they are public**: HW1 Imitation,
  HW2 Policy Gradients, HW3 Q-Learning/Actor-Critic, **HW4 LLM RL** (new), HW5 Offline RL.
  Use the Fall 2023 playlist together with the current homeworks.
- **Hugging Face Deep RL Course** — `course` `free` —
  https://huggingface.co/learn/deep-rl-course/unit0/introduction — Colab-based; teaches SB3,
  RL Zoo, Sample Factory and CleanRL. **Its current state:** the pages describe
  themselves as in a "low-maintenance state", Unit 7 does not work, and the leaderboard is
  retired. The theory and hands-on units still work; treat the certificate as a bonus.
  **Skip Unit 7.** The hands-on notebooks open straight in Colab —
  [Unit 1](https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/notebooks/unit1/unit1.ipynb "Train and publish your first RL agent with Stable-Baselines3")
  (train and publish your first agent) and
  [Unit 4](https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/notebooks/unit4/unit4.ipynb "REINFORCE from scratch in PyTorch — the implement-it-yourself unit")
  (REINFORCE from scratch in PyTorch — the unit that maps onto §6.5's "implement it yourself").

---

## 6.3 Spinning Up — read the docs, do not run the code

- **Spinning Up in Deep RL — docs** — `docs` `free` — https://spinningup.openai.com/en/latest/
  The three-part [Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
  is still, in 2026, the clearest three-hour explanation of the RL problem, value functions and
  policy optimisation anywhere. Six algorithms are documented with pseudocode, derivation and
  code: VPG, TRPO, PPO, DDPG, TD3, SAC.
- **Key Papers in Deep RL** — https://spinningup.openai.com/en/latest/spinningup/keypapers.html —
  about 100 papers, annotated, across 13 areas. Still the best-organised RL reading list in
  existence, even though it stops around 2019.
- **"Spinning Up as a Deep RL Researcher"** — https://spinningup.openai.com/en/latest/spinningup/spinningup.html
- **The code repo** — https://github.com/openai/spinningup — **status: effectively
  unmaintained, and painful to install in 2026.** The README says "Status: Maintenance"; the
  install docs still specify `python=3.6`, OpenAI Gym (not Gymnasium), the dead `mujoco-py`,
  and Linux/macOS only. **Read the docs; use CleanRL for runnable versions of the same six
  algorithms.**

---

## 6.4 Libraries — with a CPU-only assessment

| Library | Status (Aug 2026) | CPU-only | Best for |
|---|---|---|---|
| **CleanRL** | Last tag v1.0.0 (2022); Gymnasium migration PR still open; pins Python <3.11 | Fine on CPU for classic control | **Reading and re-implementing.** The best teaching artifact in RL |
| **Stable-Baselines3** | **v2.8.0, April 2026**, ~13.4k ★, Python 3.10+, Gymnasium-native | **Excellent** — pure PyTorch | Baselines you can trust; "I want a working agent, not a lesson" |
| **SB3-Contrib** | Same API, experimental algorithms | Excellent | **MaskablePPO** is quietly the most useful thing here, for control problems with invalid actions |
| **RL Baselines3 Zoo** | Active | Excellent | Tuned hyperparameters for ~100 environments + Optuna integration |
| **SBX (SB3 on JAX)** | Active | Good | If you are already in JAX; maintainers cite ~20× speedups |
| **Gymnasium** | **v1.3.0, April 2026**, Python 3.10–3.14 | Excellent | **The** standard environment API |
| **PettingZoo** | Active | Excellent | Multi-agent — directly relevant to flow control |
| **TorchRL** | **v0.13.2, June 2026**, official PyTorch org | Works; docs are CUDA/Linux-first | Production PyTorch pipelines. Steepest learning curve (TensorDict) |
| **Tianshou** | **v2.0.1, April 2026**, ~10.8k ★, 40+ algorithms | Good | **Breadth** — if an algorithm exists, Tianshou has it |
| **RLlib (Ray)** | Active, but mid-migration to a "new API stack" | Runs on a laptop | **Wrong tool for learning** — most tutorials you will find target the old stack and no longer work |

Links: [CleanRL](https://github.com/vwxyzjn/cleanrl) ([docs](https://docs.cleanrl.dev/)) ·
[SB3](https://github.com/DLR-RM/stable-baselines3) ([docs](https://stable-baselines3.readthedocs.io/en/master/)) ·
[SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) ·
[RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) · [SBX](https://github.com/araffin/sbx) ·
[Gymnasium](https://gymnasium.farama.org/) · [PettingZoo](https://pettingzoo.farama.org/) ·
[TorchRL](https://docs.pytorch.org/rl/stable/index.html) · [Tianshou](https://tianshou.org/en/stable/) ·
[RLlib](https://docs.ray.io/en/latest/rllib/index.html)

**Use `gymnasium`, never `gym`, in new code.** The 5-value `step()` return
(`obs, reward, terminated, truncated, info`) is the visible difference — and the source of most
copy-pasted-code bugs.

---

## 6.5 What to implement, in order

| # | Algorithm | Why this step | Reference implementation |
|---|---|---|---|
| 1 | **Tabular Q-learning** | Value estimation with no network to hide behind. 40 lines on CliffWalking | Sutton & Barto §6.5 + `chapter06/cliff_walking.py` in [Zhang's repo](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) |
| 2 | **DQN** | Function approximation, replay buffer, target network — the three tricks that stop deep value-based RL from diverging | [`cleanrl/dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py) |
| 3 | **REINFORCE** | The policy-gradient theorem in its rawest form. High variance *on purpose* | [Spinning Up VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html) + [Lil'Log](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) |
| 4 | **A2C** | Adds a learned baseline (the critic). The variance drop is dramatic and visible | [SB3 `a2c.py`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py) |
| 5 | **PPO** | The workhorse. Clipped objective + GAE. If you learn one deep RL algorithm properly, this is it | [`cleanrl/ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) |
| 6 | **SAC** | Continuous control, off-policy, maximum entropy. **The right default for physical systems** | [`cleanrl/sac_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) |

- **The 37 Implementation Details of Proximal Policy Optimization** — `blog` `free` —
  https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ — **read this the
  moment you touch PPO.** It lists every undocumented trick (observation normalisation,
  advantage normalisation, orthogonal init with specific gains, value clipping, LR annealing,
  gradient clipping, Adam epsilon 1e-5) that separates "PPO that works" from "PPO that does
  not." It is also a masterclass in a bigger lesson: **implementation details beat algorithm
  choice** — a lesson that transfers straight to PINNs.
- **Policy Gradient Algorithms (Lil'Log)** — `blog` `free` —
  https://lilianweng.github.io/posts/2018-04-08-policy-gradient/ — REINFORCE → A2C/A3C → TRPO →
  PPO → DDPG → TD3 → SAC in one consistent notation. The most efficient way to see the family
  tree.
- **Debugging RL, Without the Agonizing Pain (Andy Jones)** — `blog` `free` —
  https://andyljones.com/posts/rl-debugging.html — **the RL version of Karpathy's recipe.** Its
  core contribution is a set of **probe environments** — trivial one-step problems with known
  correct answers — that isolate exactly which part of your agent is broken. Nothing else here
  will save you as many hours.

---

## 6.6 Environments that train on a laptop CPU in minutes

- **Gymnasium Classic Control** — https://gymnasium.farama.org/environments/classic_control/ —
  **CartPole-v1 solves with PPO in ~30 seconds on a CPU.** This is your unit test: if your
  implementation cannot solve CartPole, it is broken.
- **Gymnasium Box2D** — https://gymnasium.farama.org/environments/box2d/ — **LunarLander-v3 is
  the sweet spot**: a real test, ~5–15 minutes on CPU with SB3 PPO.
  `pip install "gymnasium[box2d]"` (needs swig).
- **Gymnasium Toy Text** — https://gymnasium.farama.org/environments/toy_text/ — FrozenLake,
  CliffWalking, Taxi. Tabular, instant. This is where step 1 happens.
- **MinAtar** — https://github.com/kenjyoung/MinAtar — miniature Atari on a 10×10 grid. The
  *hard* parts of Atari (sparse reward, exploration) at ~1/1000 of the compute. **The best free
  lunch on this list for CPU-only work.**
- **Gymnax** — https://github.com/RobertTLange/gymnax — classic control + MinAtar + bsuite in
  pure JAX, so the *entire* environment-plus-agent loop can be `jit`-ed and `vmap`-ed.
  **Caveat: last release April 2024** — works, but not evolving. See also
  [PureJaxRL](https://github.com/luchris429/purejaxrl).
- **Brax** — https://github.com/google/brax — differentiable physics in JAX with PPO/SAC
  included. **Two caveats:** only `brax/training` is actively maintained as of 0.13.0, and it
  is built for TPU/GPU — on a laptop CPU you lose the point. Keep it as the *concept demo* for
  hardware-accelerated differentiable simulation, the on-ramp to §6.8.

---

## 6.7 The bridge: where RL actually meets physics

### Plasma control — **MATURE, a real deployment**

- **Magnetic control of tokamak plasmas through deep reinforcement learning** — `paper` `free` —
  https://www.nature.com/articles/s41586-021-04301-9 — Degrave et al., *Nature* 2022.
  DeepMind + EPFL trained a control policy in a plasma simulator and deployed it on the **real
  TCV tokamak** (a fusion research reactor), driving 19 magnetic coils, including exotic plasma
  shapes. **The single most credible RL-for-physics result in existence** — simulation to
  reality, safety-critical, on real hardware.
- **Towards practical RL for tokamak magnetic control** — `paper` `free` —
  https://arxiv.org/abs/2307.11546 (journal version:
  https://www.sciencedirect.com/science/article/pii/S0920379624000140) — the follow-up:
  what broke, where the shape accuracy fell short, and how they cut training time. **Read this
  second — it is where the engineering reality lives.**
- **Magnetic control of WEST plasmas through deep RL** — https://hal.science/hal-04393963v2/document —
  an independent replication on a *different* tokamak. Replication is what turns an impressive
  demo into a mature technique.
- **TORAX** — https://github.com/google-deepmind/torax — DeepMind's differentiable JAX tokamak
  simulator. The natural sandbox.

### Flow control — **ACTIVE RESEARCH, not engineering**

- **Rabault et al., *JFM* 2019** — https://arxiv.org/abs/1808.07664 — the founding paper: PPO
  controlling two synthetic jets on a 2D cylinder at Re=100, about 8% drag reduction by
  suppressing vortex shedding.
- **Multi-agent RL on 3D cylinders entering turbulence (2025)** — https://www.nature.com/articles/s44172-025-00446-x
  — one agent per spanwise segment; the multi-agent framing is the answer to scaling (hence
  PettingZoo above).
- **Multi-agent RL drag reduction at Re_D = 3900 (2025)** — https://pmc.ncbi.nlm.nih.gov/articles/PMC12092499/ —
  a genuinely turbulent regime, ~9% drag reduction, open access with enough method detail to
  re-implement.

Real and reproducible — but confined to textbook geometries at modest Reynolds numbers, with the
cost of CFD training as the hard limit. **Nobody is flying an RL flow controller.**

### Adaptive mesh refinement — **PROMISING BUT EARLY**

- https://arxiv.org/abs/2103.01342 (Yang et al., AISTATS 2023) — poses mesh refinement as a
  sequential decision problem; the formulation paper.
- https://github.com/LLNL/marl-amr — multi-agent RL for adaptive mesh refinement from Lawrence
  Livermore, built on MFEM. **Actual runnable code from a national lab** — rare in this corner.
- https://arxiv.org/abs/2209.12351 (Foucart, Charous, Lermusiaux, *JCP* 2023) — handles the
  mesh-size-invariance problem properly and compares fairly against classical error
  estimators, **including the cases where RL loses.**

The gains over well-tuned classical error estimators are real but modest, and generalisation
across PDEs and geometries is unresolved.

### RL for solver parameters — **NICHE**

- https://arxiv.org/abs/2407.15872 — RL to choose multigrid cycle parameters.
- https://arxiv.org/abs/2106.01854 — RL for algebraic-multigrid coarsening.
- https://arxiv.org/abs/1902.10248 — **included deliberately as the counterexample:** Greenfeld
  et al. (ICML 2019) is plain *supervised* learning, not RL — and it works better. **For many
  "learn the solver parameter" problems, supervised learning on solver traces trains faster and
  works better. Always ask whether you need the sequential-decision framing at all.**

---

## 6.8 The most important paragraph in this chapter

- **Physics-based Deep Learning (Thuerey et al.)** — `book` `free` — https://www.physicsbaseddeeplearning.org/intro.html
  · repo https://github.com/tum-pbs/pbdl-book — **v0.3, the single best free resource on this
  bridge.** A fully executable Jupyter book. **It runs the *same* control problem (Burgers)
  through both a differentiable-physics solver and PPO, side by side, with code.** Read the
  [RL chapter](https://www.physicsbaseddeeplearning.org/reinflearn-intro.html) immediately after
  the differentiable-physics chapter — the comparison *is* the lesson.
- **Learning to Control PDEs with Differentiable Physics (Holl, Koltun, Thuerey, ICLR 2020)** —
  `paper` `free` — https://arxiv.org/abs/2001.07457 — the predictor-corrector scheme for
  long-horizon PDE control, and the clearest demonstration that **differentiable physics beats
  model-free RL by a wide margin whenever you can take gradients through the solver.**
- **ΦFlow** — https://github.com/tum-pbs/PhiFlow — the differentiable PDE framework behind both.
- **PDE Control Gym** — `paper+repo` `free` — https://arxiv.org/abs/2405.11401 ·
  https://github.com/lukebhan/PDEControlGym — Gymnasium-compatible environments for boundary
  control of hyperbolic and parabolic PDEs and Navier–Stokes, with both RL and classical
  control baselines. **The most useful practical artifact in this section.**
- **A Survey on Physics-Informed Reinforcement Learning** — https://arxiv.org/abs/2309.01909 —
  a map of how physics gets injected into RL (reward, transition model, policy architecture,
  constraints). Useful overview of a young, fragmented field.
- **Do Differentiable Simulators Give Better Policy Gradients? (Suh et al., ICML 2022)** —
  https://arxiv.org/abs/2202.00817 — **the necessary nuance:** gradients from a differentiable
  simulator can be *worse* than RL's estimates for chaotic or stiff systems.

> **If you can differentiate through your simulator, do that instead of RL.** Differentiable
> simulation gives you the true gradient of your objective with respect to the control — worth
> roughly 100–1000× in sample efficiency compared with a model-free policy gradient that must
> *estimate* that gradient from noisy returns. Model-free RL is the fallback for when the
> simulator is non-differentiable, discontinuous (contact, shocks, remeshing), chaotic over
> your horizon, or a black box you do not own. **A PINN person is, by definition, already
> fluent in autodiff-through-physics — so differentiable simulation is the shorter path for
> you, not the longer one. Knowing this is worth more than knowing PPO.**

### "RL to train PINNs" — **SPECULATIVE**

Three forms, in falling order of credibility: (1) **RL or bandits for adaptive collocation
sampling** — the most defensible, but it competes with much simpler residual-based adaptive
sampling (RAR/RAD in DeepXDE) that works fine and costs nothing; (2) **RL for adaptive loss
weighting** — almost always loses to the closed-form NTK/gradient-statistics rules in
Chapter 10; (3) **reformulating PDE solving itself as a decision process** — theoretically
interesting (it connects to the classical HJB link), essentially never competitive. **Know this
literature exists so you recognise it in a related-work section. Do not invest in it.**

---

## 6.9 Suggested order (the full RL track)

*Total ~78–95 h to real competence: you can define an MDP for a new problem, implement PPO and
SAC from scratch, debug them when they silently fail, and read an RL paper critically.*

1. **Spinning Up "Introduction to RL," Parts 1–3** — 3 h
2. **Sutton & Barto Ch. 1–6 with Zhang's code open** — 14 h — *deliverable: implement tabular
   Q-learning and SARSA on CliffWalking yourself, from the pseudocode, before looking at the
   repo.*
3. **Silver lectures 1–5** as a companion pass over the same material — 7 h — watch each
   *after* the matching chapter.
4. **Gymnasium API tour + write one environment of your own** — 4 h —
   https://gymnasium.farama.org/introduction/create_custom_env/ — *deliverable: wrap a 1D
   heat-equation or Burgers solver as a Gymnasium environment* (observation = the discretised
   field, action = boundary/source control, reward = negative tracking error). Do it now, while
   it is cheap — it is the artifact you will reuse.
5. **DQN: read `cleanrl/dqn.py` line by line, then re-implement from memory** — 9 h —
   *deliverable: average return above 475 on CartPole-v1.* Then read Andy Jones's debugging
   post (+2 h) and **build his probe environments** — they pay for themselves within a day.
6. **Sutton & Barto Ch. 9–10 + 13 → implement REINFORCE, then add a baseline** — 8 h —
   *deliverable: a plot of the return variance for REINFORCE with and without a baseline.*
   Watching that variance collapse is the point.
7. **PPO: read "The 37 Details" *first*, then `cleanrl/ppo.py`, then re-implement** — 12 h —
   *deliverable: LunarLander-v3 solved in under 15 minutes on CPU.* The hardest and most
   valuable single step.
8. **SAC on Pendulum-v1 and on your own PDE environment from step 4** — 8 h — the algorithm you
   would reach for in real physical control.
9. **Sanity-check against SB3 + RL Zoo** — 3 h — if SB3 does much better than your agent, diff
   the hyperparameters. That gap *is* the lesson.
10. **Spinning Up "Key Papers" — read six with real attention** — 10 h — DQN (Mnih 2015), TRPO,
    PPO, SAC, "Deep RL that Matters" (Henderson), and one model-based paper.
11. **CS285 selected lectures — model-based RL, offline RL, exploration** — 8 h — you do not
    need all 23. The model-based block matters most for physics (you *have* a model); the
    offline block matters when your simulator is expensive.
12. **The bridge: the PBDL RL chapter + PDE Control Gym** — 10 h — *deliverable: solve the same
    Burgers control problem two ways — model-free PPO and differentiable-physics gradient
    descent — and compare the sample counts.* The number you get will settle the question in
    your own head more convincingly than any paper.

---

## 6.10 What a PINN person needs from RL

**Read this before you spend 90 hours.**

- **Strictly zero RL is required to build, train or debug a PINN** — and be suspicious of any
  curriculum implying otherwise. Nothing in Sutton & Barto Ch. 1–8 will make your Navier–Stokes
  residual converge. If your goal is "get PINNs working," **Chapter 05 is 100% of the value and
  this chapter is 0%.**
- **What transfers is optimisation craft and debugging discipline — not RL algorithms.** The
  crossover: multi-objective loss balancing (GradNorm, PCGrad, and Kendall–Gal are *the same
  machinery* PINN papers rediscover for residual-vs-boundary weighting); building probe
  problems with known answers before trusting a big run; and the cultural knowledge that
  **implementation details beat algorithm choice.** You get all of that from Chapter 05 plus
  two blog posts.
- **RL is for *controlling* PDE-governed systems, not for *solving* PDEs.** If your job is
  "find u(x,t) satisfying this equation," RL is the wrong tool and always will be. If it is
  "I have a simulator, and I need a controller that keeps this plasma/flow/reactor in a desired
  state under disturbances, in real time," RL is a serious, deployed answer — TCV is the proof.
  **Know which job you have.**
- **If your physics is differentiable, use the gradients.** RL is the fallback, not the
  default.
- **Pick a budget deliberately:**

| Budget | What to do | Who it is right for |
|---:|---|---|
| **~10 h** | Spinning Up intro + the PBDL RL chapter + the tokamak *Nature* paper | **~80% of PINN people.** Enough to read RL-flavoured physics papers critically, and to call out an over-engineered RL solution in a review |
| **~35 h** | Add steps 1–5 and 7 above: tabular Q-learning, DQN, PPO, one custom PDE environment | You have a control problem and a simulator |
| **~90 h** | The full path | You intend to publish RL-for-physics work or ship a controller |
