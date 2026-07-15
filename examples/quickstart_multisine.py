"""Quickstart with a synthetic multi-sine multitask regression dataset.

Run from the repository root:

    python -m examples.quickstart_multisine
"""

import math

import pandas as pd
import torch

from alemtl.data import MultitaskDataloader, MultitaskDataset


def make_multisine_frame(n_per_task: int = 96) -> pd.DataFrame:
    """Create three related sine-regression tasks."""

    rows = []
    x = torch.linspace(0, 2 * math.pi, steps=n_per_task)
    task_specs = {
        "low_freq": (1.0, 0.0),
        "phase_shift": (1.0, 0.6),
        "high_freq": (2.0, 0.2),
    }

    for task, (frequency, phase) in task_specs.items():
        y = torch.sin(frequency * x + phase)
        for xi, yi in zip(x, y):
            rows.append(
                {
                    "task": task,
                    "x": float(xi),
                    "sin_x": float(torch.sin(xi)),
                    "cos_x": float(torch.cos(xi)),
                    "target": float(yi),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    data = make_multisine_frame()
    dataset = MultitaskDataset(data, task_id="task", target_names="target")
    dataloader = MultitaskDataloader(dataset, batch_size=16, shuffle=True, num_workers=0)

    X, y = next(iter(dataloader))
    print(dataset)
    print("Task counts:", dataset.get_task_counts_by_original())
    print("Feature batch shape:", tuple(X.shape))
    print("Target batch shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
