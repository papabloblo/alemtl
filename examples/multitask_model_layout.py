"""Build and inspect a ``MultiTaskModel``.

Run from the repository root:

    python -m examples.multitask_model_layout

The example uses a hard-shared feature extractor and one soft-shared head per
task. Inputs follow the project convention ``(n_tasks, batch, n_features)``.
"""

import torch
from torch import nn

from alemtl.models.multitask_model import MultiTaskModel


def build_model(n_tasks: int, input_dim: int, hidden_dim: int, output_dim: int) -> MultiTaskModel:
    """Create a small multitask network from named layer specifications."""

    modules_layout = {
        "encoder": {
            "shared": "hard",
            "module": lambda: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ),
        },
        "head": {
            "shared": "soft",
            "module": lambda: nn.Linear(hidden_dim, output_dim),
        },
    }

    similarity_layers = {
        "in": "encoder",
        "out": "head",
    }

    return MultiTaskModel(
        n_tasks=n_tasks,
        modules_layout=modules_layout,
        similarity_layers=similarity_layers,
        same_parameters=True,
    )


def main() -> None:
    torch.manual_seed(42)

    n_tasks = 3
    batch_size = 8
    input_dim = 4
    hidden_dim = 16
    output_dim = 1

    model = build_model(
        n_tasks=n_tasks,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    X = torch.randn(n_tasks, batch_size, input_dim)
    y_pred = model(X)

    print(model)
    print("Full model output shape:", tuple(y_pred.shape))

    encoded = model.forward_input_similarity(X)
    similarity_output = model.forward_similarity(encoded)
    print("Similarity input shape:", tuple(encoded.shape))
    print("Similarity output shape:", tuple(similarity_output.shape))

    task_parameters = model.get_soft_shared_parameters_by_task(task=0)
    print("Task 0 soft-shared parameter vector:", tuple(task_parameters.shape))

    single_task_model = model.model_by_task(task=0)["model"]
    single_task_input = torch.randn(batch_size, input_dim)
    single_task_output = single_task_model(single_task_input)
    print("Task 0 standalone view output shape:", tuple(single_task_output.shape))


if __name__ == "__main__":
    main()
