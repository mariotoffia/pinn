"""01 - An MLP with a hand-written backward pass. NumPy only, no autograd.

If you skip this, PINNs stay permanently opaque: the entire technique is a second, unusual
application of the chain rule, and framework tutorials never motivate it.

What this script does:
  * forward pass through a tanh MLP, storing what the backward pass needs
  * backward pass derived by hand (Nielsen's four equations)
  * a finite-difference gradient check that must pass before you trust anything
  * training on a toy 1D regression problem

Write your own version first, then diff against this one.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                       # noqa: E402
from pinnlab.plotting import lines       # noqa: E402


class NumpyMLP:
    """Layers: [in, h, ..., h, out]. tanh on hidden layers, linear output."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        # Glorot/Xavier, derived for tanh: var = 2/(fan_in + fan_out)
        self.W = [rng.normal(0, np.sqrt(2.0 / (a + b)), (b, a)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((b, 1)) for b in sizes[1:]]

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        """x: (in_dim, N). Returns output and the cache the backward pass needs."""
        a = x
        cache = {"a": [a], "z": []}
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ a + b
            a = np.tanh(z) if k < len(self.W) - 1 else z      # linear output layer
            cache["z"].append(z)
            cache["a"].append(a)
        return a, cache

    # --------------------------------------------------------------- backward
    def backward(self, cache, y):
        """The four backprop equations, by hand.

          BP1  delta_L   = dC/da_L  (*)  sigma'(z_L)        [linear output -> sigma' = 1]
          BP2  delta_l   = (W_{l+1}^T delta_{l+1}) (*) sigma'(z_l)
          BP3  dC/db_l   = delta_l
          BP4  dC/dW_l   = delta_l a_{l-1}^T

        Loss is mean squared error over N samples: C = (1/N) * sum ||a_L - y||^2
        """
        a, z = cache["a"], cache["z"]
        N = y.shape[1]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)

        delta = 2.0 * (a[-1] - y) / N                     # BP1 (output layer is linear)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = delta @ a[l].T                        # BP4
            gb[l] = delta.sum(axis=1, keepdims=True)      # BP3
            if l > 0:
                # BP2. d/dz tanh(z) = 1 - tanh(z)^2, and a[l] == tanh(z[l-1]) for hidden layers.
                delta = (self.W[l].T @ delta) * (1.0 - np.tanh(z[l - 1]) ** 2)
        return gW, gb

    def loss(self, x, y):
        out, _ = self.forward(x)
        return float(np.mean(np.sum((out - y) ** 2, axis=0)))


def gradient_check(net, x, y, eps=1e-6):
    """Central finite differences on every parameter. This is the non-negotiable step.

    In float64 the analytic and numerical gradients should agree to ~1e-9 relative. If they do
    not, your backward pass is wrong - and a wrong backward pass trains to a plausible, wrong
    answer rather than crashing.
    """
    _, cache = net.forward(x)
    gW, gb = net.backward(cache, y)

    worst = 0.0
    for tensors, grads, name in ((net.W, gW, "W"), (net.b, gb, "b")):
        for l, (P, G) in enumerate(zip(tensors, grads)):
            it = np.ndindex(*P.shape)
            for _ in range(min(40, P.size)):              # spot-check 40 entries per tensor
                idx = next(it)
                orig = P[idx]
                P[idx] = orig + eps
                lp = net.loss(x, y)
                P[idx] = orig - eps
                lm = net.loss(x, y)
                P[idx] = orig
                num = (lp - lm) / (2 * eps)
                den = max(abs(num), abs(G[idx]), 1e-12)
                worst = max(worst, abs(num - G[idx]) / den)
    print(f"  worst relative gradient error over sampled entries: {worst:.3e}")
    assert worst < 1e-6, "BACKWARD PASS IS WRONG - fix it before going any further"
    print("  gradient check PASSED")
    return worst


def main():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 200).reshape(1, -1)
    y = np.sin(3 * x) * np.exp(-x**2)

    net = NumpyMLP([1, 32, 32, 1], seed=0)

    print("gradient check (before training)")
    gradient_check(net, x[:, :16], y[:, :16])

    print("\ntraining with a hand-written Adam (chapter 05: it is just three EMAs)")
    mW = [np.zeros_like(W) for W in net.W]
    vW = [np.zeros_like(W) for W in net.W]
    mb = [np.zeros_like(b) for b in net.b]
    vb = [np.zeros_like(b) for b in net.b]
    lr0, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8

    for step in range(1, 8001):
        lr = lr0 * (0.9 ** (step // 1000))                # exponential decay, as in the recipe
        _, cache = net.forward(x)
        gW, gb = net.backward(cache, y)
        for l in range(len(net.W)):
            for P, g, m, v in ((net.W, gW[l], mW, vW), (net.b, gb[l], mb, vb)):
                m[l] = b1 * m[l] + (1 - b1) * g
                v[l] = b2 * v[l] + (1 - b2) * g * g
                mhat = m[l] / (1 - b1 ** step)
                vhat = v[l] / (1 - b2 ** step)
                P[l] -= lr * mhat / (np.sqrt(vhat) + eps)
        if step % 1000 == 0 or step == 1:
            print(f"  step {step:5d}  mse {net.loss(x, y):.6e}")

    pred, _ = net.forward(x)
    lines(
        x.ravel(),
        {"NumPy MLP (hand-written backprop)": pred.ravel(), "target": y.ravel()},
        "01_mlp_from_scratch.png",
        title="MLP trained with a backward pass you can read",
    )
    print(f"\nfinal mse {net.loss(x, y):.3e}")
    print("Next: scripts/02_autodiff_playground.py - the same idea, but differentiating the INPUT.")


if __name__ == "__main__":
    main()
