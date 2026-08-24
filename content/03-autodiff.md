---
title: Automatic Differentiation
subtitle: The hinge chapter — everything after this depends on grad(grad(u))
minutes: 20
---

# 03 — Automatic Differentiation

**This is the hinge of the entire path.** Everything before it is standard deep learning.
Everything after it depends on one skill: taking derivatives *of the network's output with
respect to its inputs*, and then differentiating *that* again with respect to the weights.

Automatic differentiation ("autodiff", or AD) is the machinery that makes this possible. It is
not symbolic algebra, and it is not the finite-difference trick of nudging inputs by a small
step. It computes exact derivatives by walking the chain rule through the program itself.

Do not skim this chapter. **Target: about 10 hours, including the exercise at the end.**

---

## 3.1 The one mental shift

In ordinary supervised learning, you differentiate **the loss with respect to the weights** θ:

```
loss = mse(net(x), y)
loss.backward()                 # ∂L/∂θ
```

In a PINN you *also* differentiate the **network output with respect to its inputs** — and you
do it **inside the loss**, so the weight gradient must then flow back through those input
derivatives too:

```
u    = net(x)                                                # forward
u_x  = grad(u, x, create_graph=True)                         # ∂u/∂x   — graph kept alive
u_xx = grad(u_x, x, create_graph=True)                       # ∂²u/∂x² — nested AD
loss = ((u_xx + f(x))**2).mean()                             # the physics
loss.backward()                                              # ∂L/∂θ through all of the above
```

That is three passes over the computation graph: `u → u_x → u_xx → θ`. The graph for `u_xx` is
roughly twice the size of the graph for `u_x`, which is already larger than the forward pass.
The flag `create_graph=True` is what keeps each intermediate derivative differentiable — forget
it, and your weight gradient is silently wrong (or zero).

**Everything is exact.** These are not finite differences. There is no truncation error, no grid,
no stability condition. All the error in a PINN comes from the network's limited ability to
represent the solution, plus the optimiser's failure to find the best weights. That exactness is
the strongest true claim the method has.

---

## 3.2 Forward mode vs reverse mode — and why it matters here

Autodiff comes in two flavours. **Reverse mode** (ordinary backprop) computes derivatives of one
output with respect to many inputs in one sweep. **Forward mode** does the opposite: one input,
many outputs, one sweep.

| | Cost | Efficient when | In a PINN |
|---|---|---|---|
| **Reverse mode** (backprop, VJP) | one sweep per **output** | few outputs, many inputs | ∂L/∂θ — thousands of weights, one scalar loss ✅ |
| **Forward mode** (JVP, dual numbers) | one sweep per **input** | few inputs, many outputs | ∂u/∂x — only 1–4 input coordinates ✅ |

A PINN contains both shapes in one loss. The input dimension is tiny (`x, y, z, t` — at most 4),
so **forward mode is the cheap direction for the PDE derivatives**, and reverse mode is right for
the weight gradient. That is exactly the combination `jacfwd(jacrev(...))` builds — and exactly
what naive chained `autograd.grad` calls fail to exploit.

For a `d`-dimensional Laplacian, `torch.diagonal(hessian(u)(x)).sum()` is the whole operator.

**Cost grows with the order of the PDE.** A 4th-order equation (biharmonic, Cahn–Hilliard,
Kuramoto–Sivashinsky) costs noticeably more per training point than a 2nd-order one — and the
memory for the nested graph usually runs out before the compute does.

---

## 3.3 Read these, in this order

1. **PyTorch — A Gentle Introduction to `torch.autograd`** — `tutorial` `free` —
   https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html — the shortest
   correct mental model of the backward graph. ~1 h.
2. **PyTorch — Autograd mechanics (the essential note)** — `docs` `free` —
   https://docs.pytorch.org/docs/stable/notes/autograd.html — how the graph is built and freed,
   how `requires_grad` propagates, in-place safety, **and the section on higher-order
   gradients.** Read this *before* writing a PINN, not after.
3. **JAX Autodiff Cookbook** — `notebook` `free` — https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
   — `jacfwd`/`jacrev`, JVP vs VJP, Hessians, **higher-order derivatives.** It carries a JAX
   label, but it is the best-written explanation of how forward and reverse mode compose,
   anywhere. **Read it even if you never write JAX.** ~2 h.
4. **Jacobians, Hessians, hvp, vhp — composing function transforms** — `tutorial` `free` —
   https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html — **the single most
   important PyTorch tutorial on this path.** It teaches the `torch.func` style for exactly
   what a PDE residual needs. ~1.5 h.
5. **Automatic Differentiation in Machine Learning: a Survey (Baydin et al., JMLR 2018)** — `paper`
   `free` — https://arxiv.org/abs/1502.05767 — fixes the vocabulary: autodiff is not symbolic
   math and not numerical differencing. Read §2–3 if you are short on time. ~1–2 h.
6. **PyTorch's official autograd-engine series** — `blog` `free` — the best free material on the
   internals. Read all three in order:
   [Overview of the Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) ·
   [How Graphs are Constructed](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/) ·
   [How Graphs are Executed](https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/)
7. **PyTorch internals (Edward Z. Yang)** — `blog` `free` — http://blog.ezyang.com/2019/05/pytorch-internals/
   — tensors, striding, dispatch, and where autograd sits in the stack. Written in 2019, but
   nothing has replaced it and the architecture is unchanged.

Also worth keeping open:
- **Forward-mode AD in PyTorch** — https://docs.pytorch.org/tutorials/intermediate/forward_ad_usage.html
- **Double Backward with Custom Functions** — https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
  — exactly the mechanics that break on Apple's MPS backend, and it teaches you to read the
  "derivative for X is not implemented" error.
- **Gradcheck mechanics** — https://docs.pytorch.org/docs/stable/notes/gradcheck.html —
  `gradgradcheck` is how you *prove* your PDE residual's second derivatives are correct, and
  **it needs float64.**
- **JAX — Advanced automatic differentiation** — https://docs.jax.dev/en/latest/advanced_autodiff.html
- **JAX — Custom derivative rules** — https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  — `custom_jvp` / `custom_vjp`, for when a physics term has a known exact derivative.
- **ETH Zürich — Tutorial 10: Coding Autodiff** — `notebook` `free` —
  [open in Colab](https://colab.research.google.com/github/camlab-ethz/AI_Science_Engineering/blob/main/Tutorial%2010%20-%20Coding%20Autodiff.ipynb)
  — from the ETH course in Chapter 12 §12.7: build reverse-mode autodiff yourself, then check
  it against PyTorch. The natural bridge from Chapter 02's micrograd to `grad(grad(u))`.
- **UvA — Introduction to JAX** — `notebook` `free` —
  [open in Colab](https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.ipynb)
  — if the JAX Cookbook above moves too fast, this is the gentler runnable on-ramp (MIT,
  maintained).

---

## 3.4 The two PyTorch idioms — know both, prefer the second

**Idiom A — `torch.autograd.grad` on a batch.** Works on whole *batches* and returns summed
derivatives. This is what nearly all PINN papers and older code use.

```python
x = x.requires_grad_(True)                                   # (N, 1)
u = net(x)                                                   # (N, 1)
u_x  = torch.autograd.grad(u.sum(),   x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
```

The `.sum()` trick works because each output row depends only on its own input row, so
`∂/∂xᵢ Σⱼ u(xⱼ) = ∂u(xᵢ)/∂xᵢ`. It is correct, it works everywhere — and it hides what is
actually going on.

**Idiom B — `torch.func` transforms on a single point, then `vmap`.** You write the physics for
*one* point — which is how the PDE is written on paper — and then vectorise it mechanically.
This composes better, looks like JAX, and is what the official tutorial teaches.

```python
from torch.func import functional_call, vmap, grad, jacrev, jacfwd

def u_fn(params, z):                  # z: (2,) -> scalar. ONE point.
    return functional_call(net, params, (z.unsqueeze(0),)).squeeze()

def laplacian(params, z):             # trace of the input-Hessian
    H = jacfwd(jacrev(u_fn, argnums=1), argnums=1)(params, z)
    return torch.diagonal(H).sum()

lap_batch = vmap(laplacian, in_dims=(None, 0))(params, pts)   # NOTE: in_dims, not in_axes
```

Verified `torch.func` transforms in PyTorch 2.13: `grad`, `grad_and_value`, `hessian`, `jacfwd`,
`jacrev`, `jvp`, `vjp`, `vmap`, `linearize`, `functional_call`, `stack_module_state`,
`functionalize`.

**JAX says the same thing more directly:**

```python
import jax; jax.config.update("jax_enable_x64", True)   # MUST be line 2
import jax.numpy as jnp
from jax import grad, jit, vmap, hessian

lap = lambda p, z: jnp.trace(hessian(u, argnums=1)(p, z))
loss = lambda p, pts: jnp.mean((vmap(lap, in_axes=(None, 0))(p, pts) - f(pts))**2)
g = jit(grad(loss))(params, pts)
```

A JAX PINN reads like the PDE itself. That clarity is why JAX is on this path, even
though PyTorch is the primary framework.

---

## 3.5 Five ways this will trip you up

1. **Forgetting `create_graph=True`.** The intermediate derivative then has no history, and your
   weight gradient is wrong — often *believably* wrong.
2. **`vmap` uses `in_axes` in JAX but `in_dims` in `torch.func`.** You will hit this. Probably
   twice.
3. **The output layer's bias gets an *exactly zero* gradient from a pure second-derivative
   loss** (a constant disappears under `∂²`), and PyTorch turns that into
   `RuntimeError: The differentiated Tensor at index N appears to not have been used in the graph`.
   Fix: pass `allow_unused=True`, or use `torch.func.grad` (which returns a zero tensor instead
   of an error). The problem vanishes as soon as you add boundary or data loss terms — which is
   exactly why it ambushes you on your very first residual-only test.
4. **Calling `torch.set_default_dtype(torch.float64)` after building your model does nothing to
   that model.** `nn.Linear` locks in the default dtype at construction time. Set it at the very
   top of your entry point.
5. **`jax.config.update("jax_enable_x64", True)` must run before *any* array is created — and it
   fails silently by default.** Arrays made before the flag stay float32 and later mix in
   quietly. Put it on line 2, and add
   `assert jnp.array(1.0).dtype == jnp.float64` right after.

---

## 3.6 The exercise that makes the whole path click

**Do this now, before reading a single PINN paper.**

> Build a 3-layer `tanh` MLP `u(x, t)` in float64 on the CPU. Using autodiff only, compute at a
> batch of random points:
>
> `r = ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x²`   with   `ν = 0.01/π`
>
> Then compute `mean(r²)` and call `.backward()`. Confirm that every weight has a finite
> gradient.

You have just written the residual of Burgers' equation, and its weight gradient — the core of a
PINN. The rest of the method is boundary conditions, sampling and optimisation.

Check your derivatives two independent ways:
- **Finite differences.** `(u(x+h) − 2u(x) + u(x−h))/h²` with `h = 1e-5` should agree with your
  `u_xx` to about 6 digits in float64. If it does not, your graph is wrong.
- **`torch.autograd.gradgradcheck`** on a small input — the rigorous check, and it needs float64.

`starter/scripts/02_autodiff_playground.py` does all of this, including a timing comparison
between Idiom A and Idiom B.

---

## 3.7 If autodiff still feels like magic

Go back and build it yourself. **Karpathy's micrograd lecture** (https://youtu.be/VMj-3S1tku0)
and the [~150-line repo](https://github.com/karpathy/micrograd) remove the magic permanently,
and they cost only 2–3 hours. Reverse-mode AD is nothing more than a topological sort of the
computation graph plus one local chain-rule step per node. That is the entire idea.
