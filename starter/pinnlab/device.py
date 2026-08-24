"""Device and dtype selection.

The short version, argued in chapter 04 of the learning path:

  * PyTorch's MPS backend has no float64 (Metal has no `double` type - a hardware limit).
  * MPS also cannot take second derivatives through `nn.Linear`
    (`aten::linear_backward` is not implemented; pytorch#98498, open since April 2023).
  * PINNs need float64: in float32 the residual noise floor is ~1e-7 and L-BFGS stops early.

So on a Mac, PINNs run on the CPU. This module makes that choice explicit rather than silent.
"""

from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch

#: The device every script in this kit uses. CUDA if you have it, CPU otherwise - never MPS.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup(dtype: torch.dtype = torch.float64, seed: int | None = 0) -> torch.device:
    """Set the global dtype and seed, and return the device to use.

    Call this as the FIRST thing in every entry point. `torch.set_default_dtype` is read by
    `nn.Linear` at construction time, so setting it after you build a module does nothing to
    that module - a silent float32 network consuming float64 inputs.
    """
    torch.set_default_dtype(dtype)
    if seed is not None:
        set_seed(seed)
    return DEVICE


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _try(fn):
    try:
        return True, fn()
    except Exception as exc:  # noqa: BLE001 - we are deliberately reporting any failure
        return False, f"{type(exc).__name__}: {str(exc)[:110]}"


def describe_environment() -> str:
    """Human-readable report of what this machine can and cannot do.

    Used by scripts/00_check_environment.py.
    """
    lines: list[str] = []
    add = lines.append

    add(f"python      {platform.python_version()}  ({platform.machine()}, {platform.system()})")
    add(f"torch       {torch.__version__}")
    add(f"numpy       {np.__version__}")
    add("")
    add(f"cuda available : {torch.cuda.is_available()}")
    add(f"mps built      : {torch.backends.mps.is_built()}")
    add(f"mps available  : {torch.backends.mps.is_available()}")
    add(f"chosen DEVICE  : {DEVICE}   <- deliberate: MPS lacks float64 and double-backward")
    add("")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    add("float64 support")
    for dev in devices:
        ok, info = _try(lambda d=dev: torch.ones(3, dtype=torch.float64, device=d).sum().item())
        add(f"  {str(dev):6s} : {'OK' if ok else 'FAIL -> ' + str(info)}")

    add("")
    add("second-order autograd through nn.Linear (THE PINN PATTERN)")
    for dev in devices:
        for dt in (torch.float64, torch.float32):
            if dev.type == "mps" and dt is torch.float64:
                add(f"  {str(dev):6s} / {str(dt):15s} : skipped (float64 unsupported on MPS)")
                continue
            ok, info = _try(lambda d=dev, t=dt: _second_order_check(d, t))
            add(f"  {str(dev):6s} / {str(dt):15s} : {'OK' if ok else 'FAIL -> ' + str(info)}")

    return "\n".join(lines)


def _second_order_check(device: torch.device, dtype: torch.dtype) -> str:
    """u -> u_x -> u_xx -> dLoss/dtheta. Three graph traversals; this is what a PINN does."""
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    ).to(device=device, dtype=dtype)
    x = torch.linspace(0, 1, 8, device=device, dtype=dtype).reshape(-1, 1).requires_grad_(True)
    u = net(x)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    loss = (u_xx + u).pow(2).mean()
    loss.backward()
    assert net[0].weight.grad is not None
    return str(u_xx.dtype)
