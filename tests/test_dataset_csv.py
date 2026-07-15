import os

import pandas as pd
import torch

from alemtl.data.dataset import MultitaskDatasetCsv


def _write_csv(path):
    pd.DataFrame(
        {
            "task": ["b", "a", "b", "a", "b"],
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [10, 20, 30, 40, 50],
            "target": [100, 200, 300, 400, 500],
        }
    ).to_csv(path, index=False)


def test_csv_dataset_loads_rows_on_demand_and_balances_tasks(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    dataset = MultitaskDatasetCsv(
        csv_path,
        task_id="task",
        target_names=["target"],
        chunksize=2,
        y_dtype=torch.long,
    )

    assert dataset.n_tasks == 2
    assert dataset.feature_names_ == ["feature_1", "feature_2"]
    assert dataset.get_task_counts_by_original() == {"a": 2, "b": 3}
    assert len(dataset) == 3
    assert dataset.X == []
    assert dataset.Y == []

    x_0, y_0 = dataset[0]
    x_2, y_2 = dataset[2]

    torch.testing.assert_close(
        x_0,
        torch.tensor([[2, 20], [1, 10]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        y_0,
        torch.tensor([[200], [100]], dtype=torch.long),
    )
    torch.testing.assert_close(
        x_2,
        torch.tensor([[2, 20], [5, 50]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        y_2,
        torch.tensor([[200], [500]], dtype=torch.long),
    )
    dataset.close()


def test_csv_dataset_accepts_target_name_alias_and_removes_cache(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    dataset = MultitaskDatasetCsv(
        csv_path,
        task_id="task",
        target_name="target",
        chunksize=2,
        cache_dir=tmp_path,
    )
    generated_cache = dataset.cache_dir

    assert os.path.isdir(generated_cache)
    assert dataset.n_targets == 1

    dataset.close()
    assert not os.path.exists(generated_cache)
