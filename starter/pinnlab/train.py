"""The Adam -> L-BFGS training loop - recipe step 7.

Adam gets you into the basin; L-BFGS exploits curvature to actually converge, buying 2-4 orders
of magnitude on the residual. Rathore et al. (ICML 2024) establish that the sequence beats
either alone.

L-BFGS pitfalls, in the order they bite:
  1. it needs a `closure()` that re-evaluates the loss - unlike every other PyTorch optimiser,
     and forgetting zero_grad() inside it accumulates gradients across the line search;
  2. `line_search_fn` defaults to None (fixed step) and diverges on many PINN problems -
     you almost always want "strong_wolfe";
  3. it is a full-batch method: freeze the collocation set or the curvature pairs are garbage;
  4. it needs float64 - in float32 the differences of near-equal gradients are noise;
  5. `max_iter` counts INNER iterations per .step() call, not epochs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class TrainLog:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    terms: list[list[float]] = field(default_factory=list)
    extra: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    lbfgs_evals: int = 0

    def add(self, step: int, loss: float, terms: list[float], **extra) -> None:
        self.step.append(step)
        self.loss.append(loss)
        self.terms.append(terms)
        self.extra.append(extra)


def adam_then_lbfgs(
    params,
    loss_fn: Callable[[], tuple[torch.Tensor, list[torch.Tensor]]],
    *,
    adam_steps: int = 10_000,
    lbfgs_steps: int = 0,
    lr: float = 1e-3,
    decay_rate: float = 0.9,
    decay_every: int = 2000,
    log_every: int = 500,
    on_log: Callable[[int, float, list[float]], dict] | None = None,
    freeze_for_lbfgs: Callable[[], None] | None = None,
    verbose: bool = True,
) -> TrainLog:
    """Run Adam with exponential decay, then optionally polish with L-BFGS.

    `loss_fn` returns (total_loss, [term1, term2, ...]) and is responsible for its own sampling.
    `freeze_for_lbfgs` is called once before the L-BFGS phase - use it to fix the collocation set.

    Note: NO weight decay. The Expert's Guide explicitly warns that it degrades accuracy on
    forward problems, because a residual loss has no natural scale to shrink toward.
    """
    params = list(params)
    log = TrainLog()
    t0 = time.perf_counter()

    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=decay_rate)

    for step in range(adam_steps):
        opt.zero_grad(set_to_none=True)
        total, terms = loss_fn()
        total.backward()
        opt.step()
        sched.step()

        if step % log_every == 0 or step == adam_steps - 1:
            vals = [float(t.detach()) for t in terms]
            extra = on_log(step, float(total.detach()), vals) if on_log else {}
            log.add(step, float(total.detach()), vals, **extra)
            if verbose:
                msg = " ".join(f"{v:.3e}" for v in vals)
                tail = "  " + "  ".join(f"{k}={v:.3e}" for k, v in extra.items()) if extra else ""
                print(f"  adam {step:6d}  loss {float(total.detach()):.6e}  [{msg}]{tail}", flush=True)

    if lbfgs_steps > 0:
        if freeze_for_lbfgs is not None:
            freeze_for_lbfgs()
        if verbose:
            print("  --- switching to L-BFGS (collocation set frozen) ---", flush=True)

        opt2 = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=lbfgs_steps,
            history_size=100,
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
            line_search_fn="strong_wolfe",     # pitfall 2: the single most common fix
        )
        state = {"n": 0}

        def closure():
            opt2.zero_grad(set_to_none=True)   # pitfall 1
            total, terms = loss_fn()
            total.backward()
            state["n"] += 1
            if state["n"] % 100 == 0:
                vals = [float(t.detach()) for t in terms]
                extra = on_log(adam_steps + state["n"], float(total.detach()), vals) if on_log else {}
                log.add(adam_steps + state["n"], float(total.detach()), vals, **extra)
                if verbose:
                    msg = " ".join(f"{v:.3e}" for v in vals)
                    tail = "  " + "  ".join(f"{k}={v:.3e}" for k, v in extra.items()) if extra else ""
                    print(f"  lbfgs {state['n']:5d}  loss {float(total.detach()):.6e}  [{msg}]{tail}",
                          flush=True)
            return total

        opt2.step(closure)
        log.lbfgs_evals = state["n"]
        if verbose:
            print(f"  L-BFGS made {state['n']} function evaluations "
                  f"(few means it stopped early - check line_search_fn and float64)", flush=True)

    log.seconds = time.perf_counter() - t0
    return log


def relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """The number every PINN paper reports. Use a FRESH, dense grid, never the training points."""
    return float(torch.linalg.norm(pred - ref) / torch.linalg.norm(ref))
