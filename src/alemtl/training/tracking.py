"""Tracking utilities for training metrics, timings, ALE, and similarity."""

from __future__ import annotations

import os
import time
from typing import Optional

import torch


class ElapsedTime:
    """Accumulate elapsed wall-clock times for one training phase."""

    def __init__(self, id: str) -> None:
        self.id = id
        self._start_time: Optional[float] = None
        self.elapsed_times: list[float] = []

    def start(self) -> None:
        self._start_time = time.time()

    def end(self) -> None:
        if self._start_time is None:
            return
        self.elapsed_times.append(time.time() - self._start_time)
        self._start_time = None

    def total_time(self, minutes: bool = False, avg: bool = False) -> float:
        if not self.elapsed_times:
            return 0.0
        value = sum(self.elapsed_times) / len(self.elapsed_times) if avg else sum(self.elapsed_times)
        return value / 60 if minutes else value

    def last_epoch(self, minutes: bool = False) -> float:
        value = self.elapsed_times[-1] if self.elapsed_times else 0.0
        return value / 60 if minutes else value

    def print_epoch(self) -> None:
        print(
            f"[{self.id}] Last epoch: {self.last_epoch(minutes=True):.2f} minutes | "
            f"All epochs: {self.total_time(minutes=True):.2f} minutes"
        )


class TrackALE_Similarity:
    """Store sparse epoch-indexed ALE or similarity tensors."""

    def __init__(self, device: str | torch.device = "cpu", keep_epochs: Optional[int] = None) -> None:
        self.device = torch.device(device)
        self.keep_epochs = keep_epochs
        self.track: dict[int, torch.Tensor] = {}

    def update(self, epoch: int, result: torch.Tensor) -> None:
        if result is None:
            raise ValueError("result must be provided when updating ALE/similarity tracking.")

        self.track[int(epoch)] = result.detach().to(self.device).clone()
        self._prune()

    def _prune(self) -> None:
        if self.keep_epochs is None or self.keep_epochs <= 0:
            return
        while len(self.track) > self.keep_epochs:
            del self.track[min(self.track.keys())]

    def for_save(self) -> dict[int, torch.Tensor]:
        return {epoch: result.cpu() for epoch, result in self.track.items()}


class TrackMetrics:
    """Accumulate sample-weighted metrics; RMSE inputs must be batch MSEs."""

    def __init__(
        self,
        errors_names: list[str],
        device: str | torch.device = "cpu",
        keep_epochs: Optional[int] = None,
        *,
        loss_aggregation: str = "mean",
        errors_aggregation: Optional[dict[str, str]] = None,
    ) -> None:
        self.loss_aggregation = loss_aggregation
        self.errors_aggregation = dict(errors_aggregation or {})
        if any(mode not in {"mean", "rmse"} for mode in [loss_aggregation, *self.errors_aggregation.values()]):
            raise ValueError("Metric aggregation must be mean or rmse.")
        self.device = torch.device(device)
        self.keep_epochs = keep_epochs
        self.errors_names = list(errors_names)

        self.epoch_numbers: list[int] = []
        self.loss_per_task: list[Optional[torch.Tensor]] = []
        self.loss_penalty: list[Optional[torch.Tensor]] = []
        self.l2penalty: list[Optional[torch.Tensor]] = []
        self.errors: dict[str, list[Optional[torch.Tensor]]] = {name: [] for name in self.errors_names}
        self.batch_counts: list[int] = []
        self.sample_counts: list[int] = []

    def new_epoch(self, epoch: Optional[int] = None) -> None:
        """Start accumulating a new epoch."""

        if epoch is None:
            epoch = self.epoch_numbers[-1] + 1 if self.epoch_numbers else 0

        self.epoch_numbers.append(int(epoch))
        self.loss_per_task.append(None)
        self.loss_penalty.append(None)
        self.l2penalty.append(None)
        self.batch_counts.append(0)
        self.sample_counts.append(0)
        for key in self.errors:
            self.errors[key].append(None)

        self._prune()

    def _prune(self) -> None:
        if self.keep_epochs is None or self.keep_epochs <= 0:
            return
        while len(self.epoch_numbers) > self.keep_epochs:
            self.epoch_numbers.pop(0)
            self.loss_per_task.pop(0)
            self.loss_penalty.pop(0)
            self.l2penalty.pop(0)
            self.batch_counts.pop(0)
            self.sample_counts.pop(0)
            for key in self.errors:
                self.errors[key].pop(0)

    def _require_epoch(self) -> None:
        if not self.epoch_numbers:
            raise RuntimeError("new_epoch() must be called before update().")

    def _epoch_index(self, epoch: int) -> int:
        if epoch < 0:
            return epoch
        if epoch in self.epoch_numbers:
            return self.epoch_numbers.index(epoch)
        raise IndexError(f"Epoch {epoch} is not retained. Retained epochs: {self.epoch_numbers}.")

    def _to_tracking_tensor(self, value: Optional[torch.Tensor | float]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().to(self.device).clone()
        return torch.as_tensor(value, device=self.device).clone()

    def _accumulate(
        self,
        store: list[Optional[torch.Tensor]],
        value: Optional[torch.Tensor | float],
    ) -> None:
        tensor = self._to_tracking_tensor(value)
        if tensor is None:
            return

        if store[-1] is None:
            store[-1] = tensor
        else:
            store[-1] = store[-1] + tensor

    def _epoch_mean(self, values: list[Optional[torch.Tensor]], epoch: int, aggregation: str = "mean") -> torch.Tensor:
        idx = self._epoch_index(epoch)
        value = values[idx]
        count = self.sample_counts[idx]
        if value is None or count <= 0:
            return torch.tensor(0.0, device=self.device)
        mean = value / count
        return mean.clamp_min(0).sqrt() if aggregation == "rmse" else mean

    def update(
        self,
        loss_per_task: torch.Tensor,
        errors_per_task: dict[str, torch.Tensor],
        loss_penalty: torch.Tensor,
        l2_penalty: torch.Tensor | float | None = None,
        *,
        batch_size: int = 1,
    ) -> None:
        """Accumulate one batch of metrics for the current epoch."""

        self._require_epoch()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        missing = set(self.errors) - set(errors_per_task)
        if missing:
            raise KeyError(f"Missing error metrics: {sorted(missing)}.")

        self._accumulate(self.loss_per_task, loss_per_task.detach() * batch_size)
        self._accumulate(self.loss_penalty, loss_penalty)
        self._accumulate(self.l2penalty, l2_penalty)
        self.batch_counts[-1] += 1
        self.sample_counts[-1] += batch_size

        for key in self.errors:
            self._accumulate(self.errors[key], errors_per_task[key].detach() * batch_size)

    def loss_epoch(self, epoch: int) -> torch.Tensor:
        value = self._epoch_mean(self.loss_per_task, epoch, self.loss_aggregation)
        return value.mean() if value.ndim > 0 else value

    def last_loss(self) -> torch.Tensor:
        return self.loss_epoch(-1)

    def info(self, epoch: int) -> dict[str, torch.Tensor]:
        errors = {key: self._epoch_mean(values, epoch, self.errors_aggregation.get(key, "mean")).mean() for key, values in self.errors.items()}
        errors["LOSS"] = self.loss_epoch(epoch)
        return errors

    def for_save(self) -> dict:
        return {
            "epochs": torch.tensor(self.epoch_numbers, dtype=torch.long),
            "sample_counts": torch.tensor(self.sample_counts, dtype=torch.long),
            "loss_per_task": self.to_cpu_tensor(self.loss_per_task, self.sample_counts, self.loss_aggregation),
            "loss_penalty": self.to_cpu_tensor(self.loss_penalty, self.batch_counts),
            "l2penalty": self.to_cpu_tensor(self.l2penalty, self.batch_counts),
            "errors": {
                error: self.to_cpu_tensor(values, self.sample_counts, self.errors_aggregation.get(error, "mean"))
                for error, values in self.errors.items()
            },
        }

    @staticmethod
    def to_cpu_tensor(
        values: list[Optional[torch.Tensor]],
        batch_counts: list[int],
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Convert accumulated epoch sums into mean tensors on CPU."""

        if not values:
            return torch.empty(0, dtype=torch.float32)

        first = next((value for value in values if value is not None), None)
        if first is None:
            return torch.zeros(len(values), dtype=torch.float32)

        zero = torch.zeros_like(first, device=first.device)
        means = [
            zero if value is None or count <= 0 else value / count
            for value, count in zip(values, batch_counts)
        ]
        result = torch.stack(means).cpu()
        return result.clamp_min(0).sqrt() if aggregation == "rmse" else result


class Tracker:
    """Coordinate all training metrics, timings, checkpoints, and saved artifacts."""

    METRIC_PHASES = ("train", "validation", "test")

    def __init__(
        self,
        errors_names: list[str],
        device: str | torch.device = "cpu",
        keep_epochs: Optional[int] = None,
        path: Optional[str] = None,
        config_info: Optional[dict] = None,
        *,
        loss_aggregation: str = "mean",
        errors_aggregation: Optional[dict[str, str]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.keep_epochs = keep_epochs
        self.current_epoch = -1
        self.path = path
        self.config_info = config_info

        self.track = {
            "train": {
                "timing": ElapsedTime("train"),
                "metrics": TrackMetrics(
                    errors_names, device=self.device, keep_epochs=self.keep_epochs,
                    loss_aggregation=loss_aggregation, errors_aggregation=errors_aggregation,
                ),
            },
            "validation": {
                "timing": ElapsedTime("validation"),
                "metrics": TrackMetrics(
                    errors_names, device=self.device, keep_epochs=self.keep_epochs,
                    loss_aggregation=loss_aggregation, errors_aggregation=errors_aggregation,
                ),
            },
            "test": {
                "timing": ElapsedTime("test"),
                "metrics": TrackMetrics(
                    errors_names, device=self.device, keep_epochs=self.keep_epochs,
                    loss_aggregation=loss_aggregation, errors_aggregation=errors_aggregation,
                ),
            },
            "ale": {
                "timing": ElapsedTime("ale"),
                "metrics": TrackALE_Similarity(device=self.device, keep_epochs=self.keep_epochs),
            },
            "similarity": {
                "timing": ElapsedTime("similarity"),
                "metrics": TrackALE_Similarity(device=self.device, keep_epochs=self.keep_epochs),
            },
        }

        self.best_epoch: int = -1
        self.best_model_parameters: Optional[dict[str, torch.Tensor]] = None
        self.best_val_loss_value: Optional[float] = None
        self._best_train_metrics: Optional[dict[str, torch.Tensor]] = None
        self._best_val_metrics: Optional[dict[str, torch.Tensor]] = None

    # Start and end methods ------------------------------------
    def start_epoch(self) -> None:
        self.current_epoch += 1

    def _epoch_for_new_metrics(self) -> int:
        return max(self.current_epoch, 0)

    def start_train(self) -> None:
        self.track["train"]["metrics"].new_epoch(self._epoch_for_new_metrics())
        self.track["train"]["timing"].start()

    def end_train(self) -> None:
        self.track["train"]["timing"].end()

    def start_validation(self) -> None:
        self.track["validation"]["metrics"].new_epoch(self._epoch_for_new_metrics())
        self.track["validation"]["timing"].start()

    def end_validation(self) -> None:
        self.track["validation"]["timing"].end()

    def start_test(self) -> None:
        self.track["test"]["metrics"].new_epoch(0)
        self.track["test"]["timing"].start()

    def end_test(self) -> None:
        self.track["test"]["timing"].end()

    def start_ale(self) -> None:
        self.track["ale"]["timing"].start()

    def end_ale(self, epoch: Optional[int] = None, ale: Optional[torch.Tensor] = None) -> None:
        if epoch is not None:
            self.track["ale"]["metrics"].update(epoch, ale)
        self.track["ale"]["timing"].end()

    def start_similarity(self) -> None:
        self.track["similarity"]["timing"].start()

    def end_similarity(self, epoch: Optional[int] = None, similarity: Optional[torch.Tensor] = None) -> None:
        if epoch is not None:
            self.track["similarity"]["metrics"].update(epoch, similarity)
        self.track["similarity"]["timing"].end()

    def update_metrics(
        self,
        phase: str,
        loss_per_task: torch.Tensor,
        errors_per_task: dict[str, torch.Tensor],
        penalty: torch.Tensor,
        l2_penalty: Optional[torch.Tensor | float] = None,
        *,
        batch_size: int = 1,
    ) -> None:
        if phase not in self.METRIC_PHASES:
            raise KeyError(f"Unknown metric phase {phase!r}. Valid phases: {list(self.METRIC_PHASES)}.")
        self.track[phase]["metrics"].update(loss_per_task, errors_per_task, penalty, l2_penalty, batch_size=batch_size)

    # Early stopping methods ------------------------------------
    def last_val_loss(self) -> torch.Tensor:
        return self.track["validation"]["metrics"].last_loss()

    def best_val_loss(self) -> Optional[float]:
        return self.best_val_loss_value

    @staticmethod
    def _clone_state_dict(params: dict) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in params.items()}

    def update_best_model(self, params: dict, epoch: int, best_loss_value: float) -> None:
        self.best_epoch = int(epoch)
        self.best_val_loss_value = float(best_loss_value)
        self.best_model_parameters = self._clone_state_dict(params)
        self._best_train_metrics = self.track["train"]["metrics"].info(epoch)
        self._best_val_metrics = self.track["validation"]["metrics"].info(epoch)

    def best_model(self) -> dict:
        if self.best_epoch < 0:
            return {
                "epoch": -1,
                "train_metrics": {},
                "val_metrics": {},
                "test_metrics": self.track["test"]["metrics"].info(-1)
                if self.track["test"]["metrics"].epoch_numbers
                else {},
                "state_dict": self.best_model_parameters,
            }

        test_metrics = (
            self.track["test"]["metrics"].info(-1)
            if self.track["test"]["metrics"].epoch_numbers
            else {}
        )
        return {
            "epoch": self.best_epoch,
            "train_metrics": self._best_train_metrics or {},
            "val_metrics": self._best_val_metrics or {},
            "test_metrics": test_metrics,
            "state_dict": self.best_model_parameters,
        }

    # Save methods ----------------------------------------------
    def _ensure_path(self) -> None:
        if not self.path:
            raise ValueError("Tracker.path is not set. Provide a directory before saving.")
        os.makedirs(self.path, exist_ok=True)

    def save_times(self) -> None:
        self._ensure_path()
        times = {mode: item["timing"].elapsed_times for mode, item in self.track.items()}
        torch.save(times, os.path.join(self.path, "times.pth"))

    def save_metrics(self) -> None:
        self._ensure_path()
        metrics = {mode: item["metrics"].for_save() for mode, item in self.track.items()}
        torch.save(metrics, os.path.join(self.path, "metrics.pth"))

    def save_best_model(self) -> None:
        self._ensure_path()
        torch.save(self.best_model(), os.path.join(self.path, "best_model.pth"))

    def save_config(self) -> None:
        if self.config_info is None:
            return
        self._ensure_path()
        torch.save(self.config_info, os.path.join(self.path, "config.pth"))

    def save(self) -> None:
        self.save_times()
        self.save_metrics()
        self.save_best_model()
        self.save_config()

    # Load methods are intentionally not implemented yet.
