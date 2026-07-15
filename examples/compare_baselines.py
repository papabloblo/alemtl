"""Compare baseline model forward contracts on synthetic batches.

Run from the repository root:

    python -m examples.compare_baselines
"""

import torch

from alemtl.models import CrossStitch, HardSharing, MMoE, MTAN, PLE, SingleTaskMLP, SoftSharing


def main() -> None:
    torch.manual_seed(3)
    X = torch.randn(3, 8, 4)
    models = {
        "soft": SoftSharing(3, input_dim=4, hidden=(8,), output_dim=1),
        "single": SingleTaskMLP(3, input_dim=4, hidden=(8,), output_dim=1),
        "hard": HardSharing(3, input_dim=4, trunk=(8,), head=(), output_dim=1),
        "mmoe": MMoE(3, input_dim=4, n_experts=2, expert_hidden=(8,), tower_hidden=(), output_dim=1),
        "cross": CrossStitch(3, input_dim=4, shared_dims=(8,), output_dim=1),
        "ple": PLE(
            3,
            input_dim=4,
            output_dim=1,
            n_layers=1,
            n_shared_experts=2,
            n_task_experts=1,
            expert_hidden=(8,),
            tower_hidden=(),
        ),
        "mtan": MTAN(3, input_dim=4, shared_dims=(8,), output_dim=1, tower_hidden=()),
    }

    for name, model in models.items():
        y = model(X)
        print(f"{name:>7}: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
