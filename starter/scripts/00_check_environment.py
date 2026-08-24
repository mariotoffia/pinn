"""00 - Check what YOUR machine can do for PINN training.

Chapter 00 of the learning path claims:
  1. PINNs need float64 (FP64 is All You Need, NeurIPS 2025).
  2. PINNs need second derivatives through nn.Linear - and consumer GPUs often lack one or
     both: Apple's MPS backend has no float64 (Metal has no `double` type) and no
     double-backward (pytorch#98498).
  3. Therefore PINNs train on the CPU - or on a CUDA GPU, where both requirements are met.

Do not take that on faith. Run this. It takes ten seconds and settles it for your machine.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch                                    # noqa: E402
from pinnlab import describe_environment, setup  # noqa: E402


def precision_experiment() -> None:
    """Measure the float32 noise floor of a second derivative.

    Same weights, same inputs, two precisions. The disagreement IS the noise floor of your
    residual - and squaring it into an MSE gives the smallest loss float32 can physically
    resolve. This is why L-BFGS declares victory early in float32.
    """
    import copy

    torch.manual_seed(0)
    net64 = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    ).double()
    net32 = copy.deepcopy(net64).float()      # deepcopy, or .float() mutates net64 too

    def u_xx(net, dtype):
        x = torch.linspace(-1, 1, 512, dtype=dtype).reshape(-1, 1).requires_grad_(True)
        u = net(x)
        ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        return torch.autograd.grad(ux.sum(), x, create_graph=True)[0]

    a = u_xx(net64, torch.float64)
    b = u_xx(net32, torch.float32).double()
    rel_l2 = (torch.linalg.norm(a - b) / torch.linalg.norm(a)).item()
    abs_max = (a - b).abs().max().item()

    print()
    print("float32 vs float64 second derivative of the SAME network, same inputs")
    print(f"  relative L2 disagreement in u_xx : {rel_l2:.3e}")
    print(f"  max absolute disagreement        : {abs_max:.3e}")
    print(f"  -> squared into an MSE, float32 cannot resolve a residual below ~{abs_max**2:.1e}")
    print("  -> in practice L-BFGS stalls orders of magnitude before that, and reports success")


def main() -> None:
    device = setup()
    print("=" * 78)
    print("ENVIRONMENT REPORT")
    print("=" * 78)
    print(describe_environment())
    precision_experiment()

    print()
    print("=" * 78)
    print(f"VERDICT: train PINNs on {device} in {torch.get_default_dtype()}")
    if torch.backends.mps.is_available():
        print("  You are on Apple Silicon. If the two MPS lines above failed as predicted, you")
        print("  have now established empirically that your PINN belongs on the CPU.")
    print("=" * 78)


if __name__ == "__main__":
    main()
