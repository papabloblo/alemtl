"""Tiny training run with ALE-Frechet task grouping.

Run from the repository root:

    python -m examples.train_alefrechet
"""

import torch
from torch import nn

from alemtl.models import MultiTaskModel
from alemtl.similarity import MultiTaskALE, MultitaskSimilarity
from alemtl.training import MultiTaskLoss, MultiTaskTrainer, mae_loss

SEED = 11
N_TASKS = 3
N_FEATURES = 2
N_OUTPUTS = 1
TRAIN_BATCHES = 4
VALIDATION_BATCHES = 2
TEST_BATCHES = 2
BATCH_SIZE = 16
NUM_INTERVALS = 8
N_GUESS = 32
LEARNING_RATE = 1e-2
L2_PENALTY = 1e-4


def make_batches(
    n_tasks: int = N_TASKS,
    n_batches: int = TRAIN_BATCHES,
    batch_size: int = BATCH_SIZE,
):
    """Create synthetic task-first regression batches."""

    weights = torch.tensor([[2.0, -1.0], [2.1, -0.9], [-1.0, 2.0]])
    batches = []
    for _ in range(n_batches):
        x = torch.randn(n_tasks, batch_size, N_FEATURES)
        y = torch.einsum("tbf,tf->tb", x, weights).unsqueeze(-1)
        batches.append((x, y))
    return batches


def build_model() -> MultiTaskModel:
    """Build a small hard-trunk/soft-head multi-task model."""

    return MultiTaskModel(
        n_tasks=N_TASKS,
        modules_layout={
            "trunk": {
                "shared": "hard",
                "module": lambda: nn.Sequential(nn.Linear(N_FEATURES, 8), nn.ReLU()),
            },
            "head": {"shared": "soft", "module": lambda: nn.Linear(8, N_OUTPUTS)},
        },
        similarity_layers={"in": "trunk", "out": "head"},
        same_parameters=True,
    )


def main() -> None:
    torch.manual_seed(SEED)

    model = build_model()
    train_batches = make_batches(n_batches=TRAIN_BATCHES)
    validation_batches = make_batches(n_batches=VALIDATION_BATCHES)
    test_batches = make_batches(n_batches=TEST_BATCHES)

    ale = MultiTaskALE(
        model=model,
        dataloader=validation_batches,
        n_tasks=N_TASKS,
        n_features_out=N_OUTPUTS,
        num_intervals=NUM_INTERVALS,
        n_guess=N_GUESS,
    )
    similarity = MultitaskSimilarity(ale)
    loss = MultiTaskLoss(
        model=model,
        loss_fn=nn.MSELoss(reduction="none"),
        errors_fn={"MAE": mae_loss},
        l2_penalty=L2_PENALTY,
    )

    trainer = MultiTaskTrainer(
        model=model,
        train_dataloader=train_batches,
        validation_dataloader=validation_batches,
        test_dataloader=test_batches,
        optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
        loss=loss,
        ale=ale,
        multitask_similarity=similarity,
        ale_each_epochs=1,
        similarity_each_epochs=1,
        keep_similarity_epochs=1,
        print_each_epochs=2,
        print_limit_epochs=2,
        logging_dir="",
        learning_type="ALE-Frechet soft sharing",
        dataset_name="synthetic linear tasks",
        architecture="hard trunk + soft task heads",
        n_intervals_ale=NUM_INTERVALS,
        learning_rate=LEARNING_RATE,
        train_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        ale_batch_size=BATCH_SIZE,
        l2penalty=L2_PENALTY,
        seed=SEED,
    )
    trainer.train(epochs=2, max_batches=2)

    print("Final task similarity matrix:")
    print(similarity.scores.detach().cpu())


if __name__ == "__main__":
    main()
