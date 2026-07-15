from __future__ import annotations

from typing import Iterator, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import MultitaskDataset


class MultitaskDataloader:
    """
    Wrapper around torch.utils.data.DataLoader that permutes batch shape from
    (batch, tasks, ...) to (tasks, batch, ...), convenient for multitask models.

    Parameters
    ----------
    dataset : Dataset
        A dataset that yields (X, Y) where each is shaped (B, T, ...), i.e.,
        batch-first with a task axis in dim=1. `X` may be None for certain datasets.
    permute : bool, default=True
        If True, permutes (B, T, ...) -> (T, B, ...). If False, leaves batches as-is.
    *args, **kwargs :
        Passed through to `torch.utils.data.DataLoader` (e.g., batch_size, shuffle,
        num_workers, pin_memory, collate_fn, etc.)

    Attributes
    ----------
    dataloader : DataLoader
        The underlying PyTorch DataLoader.
    permute : bool
        Whether permutation is applied on iteration.
    """

    def __init__(
        self,
        dataset: Dataset,
        *args,
        permute: bool = True,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> None:
        self._dataset = dataset
        self.device = None if device is None else torch.device(device)
        self._dataloader = DataLoader(dataset=dataset, *args, **kwargs)
        self.permute = permute
        self._iter: Optional[Iterator] = None

    # -------- iteration protocol --------
    def __iter__(self) -> "MultitaskDataloader":
        self._iter = iter(self._dataloader)
        return self

    def __next__(self) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self._iter is None:
            self._iter = iter(self._dataloader)

        X, Y = next(self._iter)
        X = self._move_to_device(X)
        Y = self._move_to_device(Y)

        if not self.permute:
            return X, Y
        return self._maybe_permute(X, "X"), self._maybe_permute(Y, "Y")

    # -------- passthroughs / niceties --------
    def __len__(self) -> int:
        return len(self._dataloader)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(permute={self.permute}, "
            f"device={self.device}, dataloader={self._dataloader})"
        )

    # -------- internal helpers --------
    def _move_to_device(self, value):
        if self.device is None or value is None:
            return value
        if torch.is_tensor(value):
            return value.to(self.device)
        return value

    @staticmethod
    def _maybe_permute(t: Optional[torch.Tensor], which: str) -> Optional[torch.Tensor]:
        """
        Permute (B, T, ...) -> (T, B, ...) if t is not None.

        Raises
        ------
        TypeError
            If `t` is not a Tensor (and not None).
        ValueError
            If `t.dim() < 2` (no batch+task axes to swap).
        """
        if t is None:
            return None
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{which} must be a Tensor or None, got {type(t)!r}")
        if t.dim() < 2:
            raise ValueError(
                f"{which} must have at least 2 dims (B,T,...) to permute; got shape {tuple(t.shape)}"
            )
        # swap first two axes, keep the rest unchanged
        dims = (1, 0, *range(2, t.dim()))
        return t.permute(dims)


def create_dataloaders(
    datasets: Mapping[str, MultitaskDataset],
    batch_size_train: int,
    batch_size_ale: int,
    batch_size_test: int,
    num_workers: int = 1,
    device: str | torch.device | None = None,
) -> dict:
    """
    Creates and returns DataLoaders for training, ALE (Accumulated Local Effects), and testing.

    Args:
        datasets (dict): Dictionary containing datasets with keys 'train', 'ale', and 'test'.
        batch_size_train (int): Batch size for training DataLoader.
        batch_size_ale (int): Batch size for ALE DataLoader.
        batch_size_test (int): Batch size for testing DataLoader.
        num_workers (int, optional): Number of worker processes. Defaults to NUM_WORKERS.
        device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        dict: A dictionary containing DataLoaders for training, ALE, and testing.
    """

    required = {"train", "ale", "test"}
    missing = required - set(datasets)
    if missing:
        raise KeyError(f"Missing datasets: {sorted(missing)}")

    return {
        "train": MultitaskDataloader(
            datasets["train"],
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            device=device,
        ),
        "ale": MultitaskDataloader(
            datasets["ale"],
            batch_size=batch_size_ale,
            shuffle=True,
            num_workers=num_workers,
            device=device,
        ),
        "test": MultitaskDataloader(
            datasets["test"],
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            device=device,
        ),
    }
