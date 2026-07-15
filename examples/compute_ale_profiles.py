"""Compute ALE profiles for a deterministic multitask model.

Run from the repository root:

    python -m examples.compute_ale_profiles
"""

import torch
from torch import nn

from alemtl.models import MultiTaskModel, SoftSharedModule
from alemtl.similarity import MultiTaskALE


def build_model() -> MultiTaskModel:
    model = MultiTaskModel(
        n_tasks=3,
        modules_layout={
            "encoder": {"shared": "hard", "module": lambda: nn.Identity()},
            "head": {"shared": "soft", "module": lambda: nn.Linear(2, 1)},
        },
        similarity_layers={"in": "encoder", "out": "head"},
    )

    head = model.model.get_submodule("head")
    if not isinstance(head, SoftSharedModule):
        raise TypeError("Expected a soft-shared head.")

    weights = torch.tensor([[2.0, -1.0], [2.1, -0.9], [-1.0, 2.0]])
    with torch.no_grad():
        for task, layer in enumerate(head.task_nets):
            layer.weight.copy_(weights[task].view(1, 2))
            layer.bias.zero_()
    return model


def make_batches(n_batches: int = 6, batch_size: int = 32):
    batches = []
    for _ in range(n_batches):
        X = torch.randn(3, batch_size, 2)
        y = torch.zeros(3, batch_size, 1)
        batches.append((X, y))
    return batches


def main() -> None:
    torch.manual_seed(5)
    ale = MultiTaskALE(
        model=build_model(),
        dataloader=make_batches(),
        n_tasks=3,
        n_features_out=1,
        num_intervals=10,
        n_guess=64,
    )
    ale.update()
    curves = ale(centered=True, cumulative=True, std=1.0)

    print("ALE curves:", tuple(curves.shape))
    print("First task, first feature, first five points:")
    print(curves[0, 0, :5])


if __name__ == "__main__":
    main()
