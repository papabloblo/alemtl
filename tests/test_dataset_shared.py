import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from alemtl.data.dataset import (
    MultitaskDataset,
    MultitaskDatasetImg,
)


def _tabular_frame():
    return pd.DataFrame(
        {
            "task": ["b", "a", "b", "a", "b"],
            "feature": [1, 2, 3, 4, 5],
            "target": [10, 20, 30, 40, 50],
        }
    )


@pytest.mark.parametrize("dataset_class", [MultitaskDataset])
def test_dataframe_names_share_balancing_and_storage(dataset_class):
    dataset = dataset_class(
        _tabular_frame(),
        task_id="task",
        target_names="target",
        y_dtype=torch.long,
    )

    assert dataset.get_task_counts_by_original() == {"a": 2, "b": 3}
    assert len(dataset) == 3

    features, targets = dataset[2]
    torch.testing.assert_close(
        features,
        torch.tensor([[2], [5]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        targets,
        torch.tensor([[20], [50]], dtype=torch.long),
    )


def test_dataframe_and_image_use_shared_validation():
    frame = _tabular_frame()
    frame.loc[0, "target"] = np.nan

    with pytest.raises(ValueError, match="NaNs found in features/targets"):
        MultitaskDataset(frame, "task", "target")


def test_image_dataset_only_customizes_feature_loading(tmp_path):
    for filename, value in (("a.png", 10), ("b1.png", 20), ("b2.png", 30)):
        Image.new("RGB", (4, 4), (value, value, value)).save(
            tmp_path / filename
        )

    annotations = pd.DataFrame(
        {
            "img_file": ["a.png", "b1.png", "b2.png"],
            "task_id": ["a", "b", "b"],
            "target": [1, 2, 3],
        }
    )
    dataset = MultitaskDatasetImg(
        img_dir=tmp_path,
        img_data=annotations,
        y_dtype=torch.long,
    )

    images, targets = dataset[1]
    assert images.shape == (2, 3, 4, 4)
    torch.testing.assert_close(
        targets,
        torch.tensor([[1], [3]], dtype=torch.long),
    )
