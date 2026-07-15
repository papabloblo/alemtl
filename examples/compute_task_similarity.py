"""Compute task similarity from ALE profiles.

Run from the repository root:

    python -m examples.compute_task_similarity
"""

import torch

from alemtl.similarity import MultiTaskALE, MultitaskSimilarity

from .compute_ale_profiles import build_model, make_batches

SEED = 5
N_TASKS = 3
N_FEATURES_OUT = 1
NUM_INTERVALS = 10
N_GUESS = 64


def main() -> None:
    torch.manual_seed(SEED)

    ale = MultiTaskALE(
        model=build_model(),
        dataloader=make_batches(),
        n_tasks=N_TASKS,
        n_features_out=N_FEATURES_OUT,
        num_intervals=NUM_INTERVALS,
        n_guess=N_GUESS,
    )
    ale.update()

    similarity = MultitaskSimilarity(ale)
    similarity.compute()
    scores, pairs = similarity.tasks_groups()

    print("Task similarity matrix:")
    print(similarity.scores)
    print("Nearest task pairs:")
    for score, pair in zip(scores, pairs):
        task, nearest = pair.tolist()
        print(f"  task {task} -> task {nearest}, score={score.item():.4f}")


if __name__ == "__main__":
    main()
