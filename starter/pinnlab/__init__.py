"""pinnlab - minimal, readable PINN building blocks.

Design rules for everything in this package:
  * float64 everywhere, CPU by default (see chapter 04 of the learning path)
  * no cleverness that hides the physics
  * every module is short enough to read in full
"""

from .device import DEVICE, describe_environment, set_seed, setup
from .derivatives import d, grad_wrt, laplacian
from .losses import CausalWeighter, GradNormBalancer
from .nets import MLP, FourierFeatures, PINN
from .sampling import latin_hypercube, rar_refine, sobol, uniform
from .train import TrainLog, adam_then_lbfgs

__all__ = [
    "DEVICE",
    "setup",
    "set_seed",
    "describe_environment",
    "d",
    "grad_wrt",
    "laplacian",
    "MLP",
    "PINN",
    "FourierFeatures",
    "uniform",
    "latin_hypercube",
    "sobol",
    "rar_refine",
    "GradNormBalancer",
    "CausalWeighter",
    "adam_then_lbfgs",
    "TrainLog",
]

__version__ = "0.1.0"
