"""Loss balancing - recipe steps 5 and 6.

A PINN loss is lambda_r*L_r + lambda_bc*L_bc + lambda_ic*L_ic + lambda_d*L_d, and choosing those
lambdas decides whether you get the answer or a plausible-looking artifact. The terms carry
different physical units and their gradient magnitudes can differ by orders of magnitude before
training even starts (which is why non-dimensionalisation is step 0, not a nicety).
"""

from __future__ import annotations

import math

import torch


class GradNormBalancer:
    """Gradient-norm loss balancing (Wang, Teng & Perdikaris; Expert's Guide step 6).

    Every `every` steps:
        lambda_i_hat = sum_j ||grad L_j|| / ||grad L_i||
        lambda_i     = alpha*lambda_i + (1-alpha)*lambda_i_hat

    The weights are detached - they scale the loss but must not be differentiated through.
    The guide finds this comparable to NTK-based balancing and recommends it first, because it
    is cheaper.
    """

    def __init__(self, n_terms: int, params, every: int = 1000, alpha: float = 0.9):
        self.lam = torch.ones(n_terms, dtype=torch.get_default_dtype())
        self.params = [p for p in params if p.requires_grad]
        self.every = every
        self.alpha = alpha
        self._step = 0

    def maybe_update(self, losses: list[torch.Tensor]) -> None:
        self._step += 1
        if self.every <= 0 or (self._step - 1) % self.every != 0:
            return
        norms = []
        for i, li in enumerate(losses):
            grads = torch.autograd.grad(li, self.params, retain_graph=True, allow_unused=True)
            n = torch.sqrt(sum((g.pow(2).sum() for g in grads if g is not None),
                               torch.zeros((), dtype=li.dtype)))
            norms.append(n.detach() + 1e-12)
        total = torch.stack(norms).sum()
        hat = torch.stack([total / n for n in norms])
        hat = hat / hat.mean()                       # keep the overall loss scale stable
        self.lam = self.alpha * self.lam + (1.0 - self.alpha) * hat

    def combine(self, losses: list[torch.Tensor]) -> torch.Tensor:
        return sum(w * li for w, li in zip(self.lam, losses))


class CausalWeighter:
    """Causal temporal weighting (Wang, Sankaran & Perdikaris 2022) - recipe step 5.

    Standard PINNs minimise the residual over the whole space-time domain SIMULTANEOUSLY, so the
    network tries to satisfy the equation at t=1 before it has learned the solution at t=0.1.
    That is anti-causal, and it is why vanilla PINNs cannot do chaotic dynamics.

        w_i = exp(-eps * sum_{k<i} L_r^k)

    Segment i is only weighted appreciably once all earlier segments have small residual. Three
    lines of code; often changes everything.

    Watch `last_weights`: if the later segments never activate, eps is too large; if they are all
    ~1 from the start, eps is too small and you are back to vanilla.

    Practical note: a FIXED eps genuinely needs tuning, and getting it wrong fails silently.
    Early in training the residuals are large, so the cumulative sum saturates and the late
    segments get weights like 1e-160 - the network never sees late times at all, and nothing
    warns you. This class therefore uses

        eps_eff = min(eps, -ln(min_weight) / sum_k L_r^k)

    so the LAST segment's weight can never fall below `min_weight` while the residuals are
    large, and `eps_eff` relaxes to your chosen `eps` once training has converged and the cap
    is no longer binding. Set `min_weight=0.0` to recover the paper's fixed-eps form exactly.
    """

    def __init__(self, n_segments: int = 32, eps: float = 1.0, min_weight: float = 1e-2):
        self.n = n_segments
        self.eps = eps
        self.min_weight = min_weight
        self.last_weights: torch.Tensor | None = None
        self.last_eps: float = 0.0

    def __call__(self, t: torch.Tensor, residual: torch.Tensor,
                 t_range: tuple[float, float]) -> torch.Tensor:
        t0, t1 = t_range
        idx = ((t.detach().flatten() - t0) / (t1 - t0) * self.n).long().clamp(0, self.n - 1)
        sq = residual.pow(2).flatten()

        seg = torch.zeros(self.n, dtype=sq.dtype, device=sq.device)
        cnt = torch.zeros(self.n, dtype=sq.dtype, device=sq.device)
        seg = seg.index_add(0, idx, sq)
        cnt = cnt.index_add(0, idx, torch.ones_like(sq))
        seg = seg / cnt.clamp(min=1.0)

        cum = torch.cat([torch.zeros(1, dtype=seg.dtype, device=seg.device),
                         torch.cumsum(seg.detach(), dim=0)[:-1]])
        eps = self.eps
        total = float(cum[-1])
        if self.min_weight > 0.0 and total > 0.0:
            eps = min(eps, -math.log(self.min_weight) / total)
        self.last_eps = eps
        w = torch.exp(-eps * cum)                    # detached: a weight, not a gradient path
        self.last_weights = w.detach()
        return (w * seg).mean()
