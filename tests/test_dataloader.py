import pytest
import torch
from torch.utils.data import TensorDataset

from alemtl.data.dataloader import MultitaskDataloader, create_dataloaders


def test_multitask_dataloader_permute_and_device_argument():
    dataset = TensorDataset(
        torch.randn(5, 2, 3),
        torch.randn(5, 2, 1),
    )
    dataloader = MultitaskDataloader(
        dataset,
        batch_size=4,
        device="cpu",
    )

    X, y = next(iter(dataloader))

    assert X.shape == (2, 4, 3)
    assert y.shape == (2, 4, 1)
    assert X.device.type == "cpu"


def test_multitask_dataloader_can_leave_batch_first():
    dataset = TensorDataset(
        torch.randn(5, 2, 3),
        torch.randn(5, 2, 1),
    )
    dataloader = MultitaskDataloader(dataset, batch_size=4, permute=False)

    X, y = next(iter(dataloader))

    assert X.shape == (4, 2, 3)
    assert y.shape == (4, 2, 1)


def test_create_dataloaders_requires_all_splits():
    dataset = TensorDataset(torch.randn(5, 2, 3), torch.randn(5, 2, 1))

    with pytest.raises(KeyError, match="ale"):
        create_dataloaders(
            {"train": dataset, "test": dataset},
            batch_size_train=2,
            batch_size_ale=2,
            batch_size_test=2,
        )
