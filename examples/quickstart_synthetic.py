"""Quickstart example using a synthetic multi-task regression dataset.

Run from the repository root:

    python -m examples.quickstart_synthetic
"""

import numpy as np
import pandas as pd
import torch

from alemtl.data.dataset import MultitaskDataset
from alemtl.data.dataloader import MultitaskDataloader


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Generate imbalanced regression data with one relationship per task."""
    rng = np.random.default_rng(seed)
    task_config = {
        "task_a": {"size": 80, "weights": (2.0, -1.0), "bias": 0.5},
        "task_b": {"size": 120, "weights": (-0.5, 1.5), "bias": -1.0},
        "task_c": {"size": 50, "weights": (1.0, 1.0), "bias": 2.0},
    }

    frames = []
    for task, config in task_config.items():
        features = rng.normal(size=(config["size"], 2))
        noise = rng.normal(scale=0.2, size=config["size"])
        target = (
            features @ np.asarray(config["weights"])
            + config["bias"]
            + noise
        )
        frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "feature_1": features[:, 0],
                    "feature_2": features[:, 1],
                    "target": target,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    data = generate_synthetic_data()
    dataset = MultitaskDataset(
        data=data,
        task_id="task",
        target_names="target",
        dtype=torch.float32,
    )

    print(dataset)
    print("Rows per task:", dataset.get_task_counts_by_original())

    features, targets = dataset[0]
    print("Single balanced sample:")
    print("  features shape:", tuple(features.shape))
    print("  targets shape: ", tuple(targets.shape))

    dataloader = MultitaskDataloader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    batch_features, batch_targets = next(iter(dataloader))
    print("MultitaskDataloader batch (tasks, batch, features):")
    print("  features shape:", tuple(batch_features.shape))
    print("  targets shape: ", tuple(batch_targets.shape))


if __name__ == "__main__":
    main()
