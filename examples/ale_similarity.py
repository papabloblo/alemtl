"""Compute ALE curves and task similarity for a small multitask model.

Run from the repository root:

    python -m examples.ale_similarity

The example builds a deterministic model with three task-specific linear heads.
Tasks 0 and 1 have similar head weights, while task 2 uses a different
relationship. ALE curves are computed from synthetic batches and then compared
with the Frechet-based ``MultitaskSimilarity`` helper.
"""

import torch
from torch import nn

from alemtl.models.multitask_model import MultiTaskModel
from alemtl.models.multitask_model import SoftSharedModule
from alemtl.similarity.ale import MultiTaskALE
from alemtl.similarity.similarity import MultitaskSimilarity


def build_linear_model(n_tasks: int, input_dim: int) -> MultiTaskModel:
    """Create an identity encoder plus one linear head per task."""

    modules_layout = {
        "encoder": {
            "shared": "hard",
            "module": lambda: nn.Identity(),
        },
        "head": {
            "shared": "soft",
            "module": lambda: nn.Linear(input_dim, 1),
        },
    }

    model = MultiTaskModel(
        n_tasks=n_tasks,
        modules_layout=modules_layout,
        similarity_layers={"in": "encoder", "out": "head"},
    )

    head = model.model.get_submodule("head")
    if not isinstance(head, SoftSharedModule):
        raise TypeError("Expected the example head to be soft-shared.")

    weights = torch.tensor(
        [
            [2.0, -1.0],
            [2.1, -0.9],
            [-1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    biases = torch.tensor([0.1, 0.0, -0.2], dtype=torch.float32)

    with torch.no_grad():
        for task, task_head in enumerate(head.task_nets):
            task_head.weight.copy_(weights[task].unsqueeze(0))
            task_head.bias.copy_(biases[task].view(1))

    return model


def make_task_first_batches(
    *,
    n_tasks: int,
    n_batches: int,
    batch_size: int,
    input_dim: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create batches shaped ``(n_tasks, batch, features)``."""

    batches = []
    for _ in range(n_batches):
        shared_features = torch.randn(batch_size, input_dim)
        X = shared_features.unsqueeze(0).repeat(n_tasks, 1, 1)
        X = X + 0.05 * torch.randn_like(X)
        y = torch.zeros(n_tasks, batch_size, 1)
        batches.append((X, y))
    return batches


def main() -> None:
    torch.manual_seed(7)

    n_tasks = 3
    input_dim = 2
    model = build_linear_model(n_tasks=n_tasks, input_dim=input_dim)
    batches = make_task_first_batches(
        n_tasks=n_tasks,
        n_batches=8,
        batch_size=32,
        input_dim=input_dim,
    )

    ale = MultiTaskALE(
        model=model,
        dataloader=batches,
        n_tasks=n_tasks,
        n_features_out=1,
        num_intervals=12,
        n_guess=64,
    )
    ale.update()

    curves = ale(centered=True, cumulative=True, std=1.0)
    print("ALE curves shape:", tuple(curves.shape))
    print("  layout: (tasks, features, intervals, x_plus_outputs)")

    similarity = MultitaskSimilarity(
        ale_curves=ale,
        centered=True,
        cumulative=True,
        std=1.0,
    )
    similarity.compute()
    scores, task_pairs = similarity.tasks_groups()

    print("\nPairwise task similarity matrix:")
    print(similarity.scores)

    print("\nPer-feature task similarity tensor shape:")
    print(tuple(similarity.similarity_tasks_features.shape))

    print("\nNearest task for each task:")
    for score, pair in zip(scores, task_pairs):
        task, nearest = pair.tolist()
        print(f"  task {task} -> task {nearest} (score={score.item():.4f})")

    first_curve = curves[0, 0]
    print("\nFirst five ALE points for task 0, feature 0:")
    print(first_curve[:5])


if __name__ == "__main__":
    main()
