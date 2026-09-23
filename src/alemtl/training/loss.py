"""Loss, regularization, and metric helpers for multitask training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional

import torch
from torch import Tensor

from ..models.multitask_model import MultiTaskModel


MetricFn = Callable[[Tensor, Tensor], Tensor]


def _reduce_per_task(values: Tensor, n_tasks: int, name: str) -> Tensor:
    """Reduce a task-first tensor to one scalar per task."""

    if values.dim() == 0:
        return values.expand(n_tasks)
    if values.size(0) != n_tasks:
        raise ValueError(f"{name} must be task-first with size 0 == {n_tasks}, got shape {tuple(values.shape)}.")
    if values.dim() == 1:
        return values
    return values.mean(dim=tuple(range(1, values.dim())))


class MultiTaskLoss:
    """Task-wise loss wrapper plus optional soft-sharing penalty.

    Parameters
    ----------
    model:
        Model exposing ``n_tasks``, ``device``, and ``get_param_groups``.
    loss_fn:
        Callable loss. It should return either a task-first tensor shaped like
        ``(n_tasks, batch, ...)``, one value per task ``(n_tasks,)``, or a scalar.
    errors_fn:
        Optional mapping of metric names to callables. Each metric receives
        ``(real, pred)`` and should return a task-first tensor or one value per
        task.
    l2_penalty:
        Regularization strength consumed by the trainer.

    Notes
    -----
    ``update_tasks_groups`` receives task pairs from the similarity module. The
    penalty is the L2 distance between each pair's explicit soft-shared
    parameter vectors. It returns one penalty value per group.
    """

    def __init__(
        self,
        model: MultiTaskModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        errors_fn: Optional[Mapping[str, MetricFn]] = None,
        l2_penalty: float | Tensor = 1e-3,
    ) -> None:
        self.tasks_groups: Optional[Tensor | list[list[int]]] = None
        self.l2_penalty = l2_penalty
        self.loss_fn = loss_fn
        self.model = model
        self.errors_dict = dict(errors_fn or {})

    @property
    def loss_aggregation(self) -> str:
        return "rmse" if self.loss_fn is rmse_loss else "mean"

    @property
    def errors_aggregation(self) -> dict[str, str]:
        return {name: "rmse" if metric is rmse_loss else "mean"
                for name, metric in self.errors_dict.items()}

    @torch.no_grad()
    def epoch_values(self, real: Tensor, pred: Tensor, loss: Tensor,
                     errors: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Return additive batch means for epoch tracking, separate from backprop.

        Built-in RMSE uses mean squared error over all observations/outputs
        per task; the tracker takes the root after aggregating the epoch.
        Other callbacks must return batch means for sample weighting to apply.
        """
        if self.loss_aggregation != "rmse" and "rmse" not in self.errors_aggregation.values():
            return loss, errors
        # Accumulate in at least float32 even when forward uses mixed precision.
        dtype = torch.float64 if pred.dtype == torch.float64 else torch.float32
        squared = (pred.to(dtype) - real.to(dtype)).square()
        mse = _reduce_per_task(squared, self.model.n_tasks, "squared error")
        return (
            mse if self.loss_aggregation == "rmse" else loss,
            {name: mse if self.errors_aggregation[name] == "rmse" else value
             for name, value in errors.items()},
        )

    def update_tasks_groups(self, tasks_groups: Optional[Tensor | list[list[int]]]) -> None:
        """Set task groups used by :meth:`penalty`.

        ``tasks_groups`` is expected to contain rows like ``(task, peer_task)``.
        Passing ``None`` disables the penalty.
        """

        self.tasks_groups = tasks_groups

    def penalty(self) -> Tensor:
        """Return L2 distances between grouped task-specific parameters."""

        if self.tasks_groups is None:
            return torch.zeros(self.model.n_tasks, device=self.model.device)

        groups = self._normalize_task_groups(self.tasks_groups)
        if not groups:
            return torch.zeros(0, device=self.model.device)

        parameter_groups = self.model.get_param_groups(groups)
        if not parameter_groups:
            return torch.zeros(0, device=self.model.device)

        penalties = []
        for group, parameters in zip(groups, parameter_groups):
            if parameters.size(0) != 2:
                raise ValueError(f"Each task group must contain exactly two tasks, got {group}.")
            penalties.append(torch.linalg.vector_norm(parameters[0] - parameters[1], ord=2))
        return torch.stack(penalties).to(self.model.device)

    def _normalize_task_groups(self, tasks_groups: Tensor | list[list[int]]) -> list[list[int]]:
        if torch.is_tensor(tasks_groups):
            tasks_groups = tasks_groups.detach().cpu().tolist()

        normalized = []
        for group in tasks_groups:
            if len(group) != 2:
                raise ValueError(f"Each task group must have exactly two task indices, got {group}.")
            pair = [int(group[0]), int(group[1])]
            for task in pair:
                if not 0 <= task < self.model.n_tasks:
                    raise IndexError(f"Task index must be in [0, {self.model.n_tasks}), got {task}.")
            normalized.append(pair)
        return normalized

    def loss_per_task(self, real: Tensor, pred: Tensor) -> Tensor:
        """Compute one loss scalar per task."""

        return _reduce_per_task(self.loss_fn(real, pred), self.model.n_tasks, "loss")

    @torch.no_grad()
    def errors_per_task(self, real: Tensor, pred: Tensor) -> dict[str, Tensor]:
        """Compute configured metrics as one tensor per metric."""

        return {
            key: _reduce_per_task(metric(real, pred), self.model.n_tasks, key)
            for key, metric in self.errors_dict.items()
        }


def accuracy(y_pred: Tensor, y: Tensor, threshold: float = 0.5) -> Tensor:
    """Binary accuracy per output column after sigmoid thresholding."""

    pred_labels = torch.sigmoid(y_pred) >= threshold
    return (pred_labels == y.bool()).float().mean(dim=0)


def accuracy2(y_pred: Tensor, y: Tensor, threshold: float = 0.5) -> Tensor:
    """Backward-compatible alias for :func:`accuracy`."""

    return accuracy(y_pred, y, threshold=threshold)


def confusion_matrix(y_pred: Tensor, y: Tensor, threshold: float = 0.5) -> dict[str, Tensor]:
    """Binary confusion-matrix counts per output column."""

    pred_labels = torch.sigmoid(y_pred) >= threshold
    target = y.bool()

    return {
        "true_positives": (pred_labels & target).sum(dim=0).float(),
        "true_negatives": ((~pred_labels) & (~target)).sum(dim=0).float(),
        "false_positives": (pred_labels & (~target)).sum(dim=0).float(),
        "false_negatives": ((~pred_labels) & target).sum(dim=0).float(),
    }


def confusion_matrix2(y_pred: Tensor, y: Tensor, threshold: float = 0.5) -> dict[str, Tensor]:
    """Backward-compatible alias for :func:`confusion_matrix`."""

    return confusion_matrix(y_pred, y, threshold=threshold)


def f1_score(y_pred: Tensor, y: Tensor, threshold: float = 0.5) -> Tensor:
    """Binary F1 score per output column."""

    counts = confusion_matrix(y_pred, y, threshold)
    numerator = 2 * counts["true_positives"]
    denominator = numerator + counts["false_positives"] + counts["false_negatives"]
    return numerator / denominator.clamp_min(1e-12)


def mape_loss(real: Tensor, pred: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean absolute percentage error reduced over the batch axis."""

    denominator = real.abs().clamp_min(eps)
    return (real - pred).abs().div(denominator).mean(dim=1)


def mae_loss(real: Tensor, pred: Tensor) -> Tensor:
    """Mean absolute error reduced over the batch axis."""

    return (pred - real).abs().mean(dim=1)


def rmse_loss(real: Tensor, pred: Tensor) -> Tensor:
    """Root mean squared error reduced over the batch axis."""

    return torch.sqrt(((pred - real) ** 2).mean(dim=1))
