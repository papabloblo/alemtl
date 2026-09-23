"""
Utilities for training, evaluating, and monitoring a multi-task PyTorch model.

This module provides:

- A user-friendly printer/banner helper (``_MultiTaskTrainerPrint``) to
  display execution settings, per-epoch metric tables, and timing summaries.
- The main trainer class (``MultiTaskTrainer``) that encapsulates a standard
  train/test loop with optional ALE curve computation and task-similarity
  tracking, early stopping, checkpointing via an external ``Tracker`` object,
  and optional LR scheduling.

Typical usage example:

.. code-block:: python

    trainer = MultiTaskTrainer(
        model, train_loader, test_loader, optimizer, loss,
        ale=ale_helper, multitask_similarity=sim_helper, scheduler=sched,
        early_stopping_epochs=20, print_each_epochs=1, ...
    )
    trainer.train(epochs=100)
"""

import time
import torch
from contextlib import nullcontext

from ..data.dataloader import MultitaskDataloader
from ..models.multitask_model import MultiTaskModel
from ..similarity.ale import MultiTaskALE
from ..similarity.similarity import MultitaskSimilarity
from .loss import MultiTaskLoss
from .tracking import Tracker


class _MultiTaskTrainerPrint:
    """Pretty-printer and banners for training sessions.

    This helper formats and prints run configuration, per-epoch metric tables,
    and timing summaries for the different tracked phases (train, test, ALE,
    similarity, etc.).

    Args:
        learning_type: Short description of the training regime (e.g. "MTL").
        dataset_name: Human-readable dataset identifier.
        total_epochs: Total number of epochs to train (shown in headers).
        n_tasks: Number of prediction tasks in the model.
        logging_dir: Path where external tracking artifacts are stored.
        seed: Random seed used for the run.
        early_stopping_rounds: Patience used for early stopping.
        n_intervals_ale: Number of intervals used for ALE computation.
        learning_rate: Optimizer learning rate (for display only).
        similarity_each_epochs: Frequency (epochs) to compute similarity.
        train_batch_size: Train DataLoader batch size.
        test_batch_size: Test DataLoader batch size.
        ale_batch_size: Batch size used during ALE computation.
        l2penalty: L2 regularization coefficient used in the loss.
        architecture: Text label for the model architecture.
    """

    def __init__(
            self,
            learning_type: str,
            dataset_name: str,
            total_epochs: int,
            n_tasks: int,
            logging_dir: str,
            seed: int,
            early_stopping_rounds: int,
            n_intervals_ale: int,
            learning_rate: float,
            similarity_each_epochs: int | None,
            train_batch_size: int,
            test_batch_size: int,
            ale_batch_size: int,
            l2penalty: float,
            architecture: str
    ):

        self.current_epoch = 0
        self.dataset_name = dataset_name
        self.learning_type = learning_type
        self.total_epochs = total_epochs
        self.logging_dir = logging_dir
        self.n_tasks = n_tasks
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.n_intervals_ale = n_intervals_ale
        self.learning_rate = learning_rate
        self.similarity_each_epochs = similarity_each_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.ale_batch_size = ale_batch_size
        self.l2penalty = l2penalty
        self.architecture = architecture

        self.info, self.max_char, self.separator, self.blank_line = self._create_info()

    def _create_info(self):
        """Builds the formatted info dictionary and line helpers.

        Returns:
           tuple[dict[str, str], int, str, str]: A tuple containing:
               - info: Mapping of info keys to padded printable strings.
               - max_char: The computed maximum line width.
               - separator: Horizontal line string of width ``max_char``.
               - blank_line: A bordered blank line with width ``max_char``.
        """
        info = dict(
            execution_information="EXECUTION INFORMATION",
            learning_type=f"Learning type: {self.learning_type}",
            architecture=f"Architecture: {self.architecture}",
            dataset=f"Dataset: {self.dataset_name}",
            num_epochs=f"Number of epochs: {self.total_epochs}",
            batch_size_train=f"Train batch size: {self.train_batch_size}",
            batch_size_test=f"Test batch size: {self.test_batch_size}",
            batch_size_ale=f"ALE batch size: {self.ale_batch_size}",
            similarity_each_epochs=f"Compute similarity each epochs: {self.similarity_each_epochs}",
            learning_rate=f"Learning rate: {self.learning_rate}",
            l2penalty=f"L2 penalty: {self.l2penalty}",
            n_intervals_ale=f"Number of ALE intervals: {self.n_intervals_ale}",
            early_stopping_rounds=f"Early stopping rounds: {self.early_stopping_rounds}",
            seed=f"Seed: {self.seed}",
            n_tasks=f"Number of tasks: {self.n_tasks}",
            log_dir=f"Logging directory: {self.logging_dir}",
            execution_date=f"Execution datetime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        max_char = max(len(x) for x in info.values()) + 4
        separator = "-" * max_char
        blank_line = "|" + " " * (max_char - 2) + "|"

        info = {key: "| " + value + " " * (max_char - 4 - len(value)) + " |" for key, value in
                info.items()}

        return info, max_char, separator, blank_line

    def info_model(self):
        """
        Prints a banner with the execution configuration.
        """
        self.info, self.max_char, self.separator, self.blank_line = self._create_info()
        print()
        print(self.separator)
        print(self.blank_line)

        print(self.info['execution_information'])

        print(self.blank_line)
        print(self.separator)

        print(self.info['learning_type'])

        print(self.blank_line)

        print(self.info['architecture'])
        print(self.info['dataset'])

        print(self.blank_line)

        print(self.info['n_tasks'])
        print(self.info['num_epochs'])
        print(self.info['batch_size_train'])
        print(self.info['batch_size_ale'])
        print(self.info['batch_size_test'])
        print(self.info['similarity_each_epochs'])

        print(self.blank_line)

        print(self.info['learning_rate'])
        print(self.info['l2penalty'])

        print(self.blank_line)

        print(self.info['n_intervals_ale'])
        print(self.info['early_stopping_rounds'])

        print(self.blank_line)

        print(self.info['seed'])

        print(self.separator)

        print(self.info['log_dir'])
        print(self.info['execution_date'])

        print(self.separator)
        print()

    def start_epoch_banner(self, epoch: int, total_epochs: int) -> None:
        """Prints a centered banner announcing the start of an epoch.

        Args:
            epoch: Zero-based epoch index to show.
            total_epochs: Total number of epochs for the session.
        """
        print_info = f"EPOCH {epoch + 1} OF {total_epochs}"
        spaces = self.max_char - len(print_info) - 2
        print()
        print(self.separator)
        print('|' + " " * (spaces // 2) + print_info + " " * (spaces - spaces // 2) + '|')
        print(self.separator)

    def best_model_banner(self, best_model) -> None:
        """Prints a banner announcing the best model found during training.

        Args:
        """

        print(f"BEST MODEL FOUND AT EPOCH {best_model['epoch'] + 1}")

        metric_names = list(best_model['train_metrics'].keys())
        for metric in metric_names:
            print(f'{metric}:')
            print(f'\tTrain: {best_model["train_metrics"][metric]:.4f}')
            print(f'\tValidation: {best_model["val_metrics"][metric]:.4f}')
            print(f'\tTest: {best_model["test_metrics"][metric]:.4f}')

    @staticmethod
    def epoch_table(
            track,
            metric_names: list[str],
            from_epoch: int,
            to_epoch: int,
            similarity_each_epochs: int | None = None
    ):
        """Prints a compact table with train/test metrics for a range of epochs.

        Args:
            track: Tracker internal state (``Tracker.track``).
            metric_names: Ordered list of metric names to display. Typically
                ``['LOSS', <error_1>, <error_2>, ...]``.
            from_epoch: Inclusive starting epoch index to print.
            to_epoch: Inclusive ending epoch index to print.
        """
        epoch_width = 8
        value_width = 10
        metric_width = 2 * value_width + 3

        top_separator = (
            "+"
            + "-" * (epoch_width + 2)
            + "+"
            + "+".join("-" * (metric_width + 2) for _ in metric_names)
            + "+"
        )
        separator = (
            "+"
            + "-" * (epoch_width + 2)
            + "+"
            + "+".join(
                "+".join("-" * (value_width + 2) for _ in range(2))
                for _ in metric_names
            )
            + "+"
        )
        title_width = len(top_separator) - 4
        header = f"| {'':^{epoch_width}} |" + "".join(
            f" {metric:^{metric_width}} |" for metric in metric_names
        )
        header2 = f"| {'EPOCH':^{epoch_width}} |" + "".join(
            f" {'Train':^{value_width}} | {'Validation':^{value_width}} |"
            for _ in metric_names
        )

        print(top_separator)
        print(f"| {'METRICS':^{title_width}} |")
        print(top_separator)
        print(header)
        print(separator)
        print(header2)
        print(separator)
        for ep in range(from_epoch, to_epoch + 1):
            if (
                    ep > from_epoch
                    and similarity_each_epochs is not None
                    and (ep + 1) % similarity_each_epochs == 0
            ):
                print(separator)
            print(f"| {ep + 1:^{epoch_width}} |", end="")
            mtr_train = track['train']['metrics'].info(ep)
            mtr_val = track['validation']['metrics'].info(ep)
            for m in metric_names:
                print(
                    f" {float(mtr_train[m]):^{value_width}.4f} "
                    f"| {float(mtr_val[m]):^{value_width}.4f} |",
                    end="",
                )
            print()
        print(separator)
        print()

    @staticmethod
    def times(track, epoch, total_epochs: int):
        """Prints elapsed, average, and estimated times for tracked phases.

        Args:
            track: Tracker internal state (``Tracker.track``).
            epoch: Current zero-based epoch index.
            total_epochs: Total number of epochs planned.
        """
        header = f"|  {' ':^25}  |  CURRENT  EPOCH  |  {'ALL EPOCHS':^14}  |"
        separator = "-" * len(header)
        print("+-----------------+");
        print("|  ELAPSED TIMES  |")
        print(separator);
        print(header);
        print(separator)
        steps = [track[k]['timing'] for k in track.keys()]
        for st in steps:
            print(f"|  {st.id:^25}  |  {st.last_epoch(minutes=True):^6.2f} minutes  "
                  f"|  {st.total_time(minutes=True):^6.2f} minutes  |")
        print(separator)
        total_current = sum(st.last_epoch(minutes=True) for st in steps)
        total_all = sum(st.total_time(minutes=True) for st in steps)
        print(f'|  {"TOTAL TIME":^25}  |  {total_current:^6.2f} minutes  |  {total_all:^6.2f} minutes  |')
        avg_time = total_all / (epoch + 1)
        est_time = total_all * (total_epochs / (epoch + 1))
        print(f'|  {"AVERAGE TIME":^25}  |  {" ":14}  |  {avg_time:^6.2f} minutes  |')
        print(f'|  {"ESTIMATED TIME":^25}  |  {" ":14}  |  {est_time:^6.2f} minutes  |')
        print(separator)


class MultiTaskTrainer:
    """End-to-end trainer for multi-task PyTorch models.

    This class runs the canonical train/test loop, tracks metrics and times via
    ``Tracker``, optionally computes ALE curves and task-feature similarity, and
    supports early stopping, checkpoints (via ``Tracker.save()``), and a step
    LR scheduler.

    Args:
        model: The multi-task model to train/evaluate.
        train_dataloader: Dataloader used for training.
        test_dataloader: Dataloader used for evaluation.
        optimizer: Optimizer used during training.
        loss: Multi-task loss object that also exposes per-task errors
            and regularization penalty.
        ale: Optional ALE computer. If provided and scheduled, ALE curves
            will be updated and recorded.
        multitask_similarity: Optional similarity helper to compute task
            similarity and produce task groups for the loss.
        scheduler: Optional learning rate scheduler stepped once per epoch.
        maximize_loss: If True, "improvement" means a larger loss; otherwise
            smaller is better (default).
        early_stopping_epochs: Patience (epochs) with no improvement before
            stopping early. Use ``float('inf')`` to disable.
        print_each_epochs: Frequency to print metrics tables and times.
        ale_each_epochs: Frequency (epochs) to compute ALE. If ``None``, skip.
        similarity_each_epochs: Frequency (epochs) to compute similarity.
        keep_similarity_epochs: Number of epochs after a similarity computation
            to keep the inferred task groups in the loss; after that, groups
            are cleared (if supported by the loss).
        track_device: Device for the ``Tracker`` tensors/aggregations.
        track_epochs: Optional limit of epochs kept in memory by ``Tracker``.
        save_results_each: Frequency (epochs) to checkpoint via ``Tracker.save()``.
        logging_dir: Path displayed in the printer banner (for context only).
        learning_type: Display label for training regime (printer banner).
        dataset_name: Display dataset name (printer banner).
        n_intervals_ale: Number of intervals for ALE plots (printer banner).
        learning_rate: Learning rate to display (printer banner).
        train_batch_size: Train batch size to display (printer banner).
        test_batch_size: Test batch size to display (printer banner).
        ale_batch_size: ALE batch size to display (printer banner).
        l2penalty: L2 penalty coefficient to display (printer banner).
        architecture: Model architecture label to display (printer banner).
        seed: Random seed to display (printer banner).
    """

    def __init__(
        self,
        model: MultiTaskModel,
        train_dataloader: MultitaskDataloader,
        validation_dataloader: MultitaskDataloader,
        test_dataloader: MultitaskDataloader,
        optimizer: torch.optim.Optimizer,
        loss: MultiTaskLoss,
        ale: MultiTaskALE | None = None,
        multitask_similarity: MultitaskSimilarity | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        maximize_loss: bool = False,
        early_stopping_epochs: int | float = float("inf"),
        print_each_epochs: int = 1,
        ale_each_epochs: int | None = None,
        similarity_each_epochs: int | None = None,
        keep_similarity_epochs: int = 0,
        track_device: str = "cpu",
        track_epochs: int | None = None,
        save_results_each: int | None = None,
        logging_dir: str = "",
        learning_type: str = "",
        dataset_name: str = "",
        n_intervals_ale: int = 0,
        learning_rate: float = 0.0,
        train_batch_size: int = 0,
        test_batch_size: int = 0,
        ale_batch_size: int = 0,
        l2penalty: float = 0.0,
        architecture: str = "",
        seed: int = 0,
        print_limit_epochs: int = 5,
        config_info: dict | None = None,
        amp: bool = False,
        amp_dtype: str = "float16",
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.ale = ale
        self.multitask_similarity = multitask_similarity

        self.early_stopping_epochs = early_stopping_epochs
        self.maximize_loss = bool(maximize_loss)
        self.ale_each_epochs = ale_each_epochs
        self.similarity_each_epochs = similarity_each_epochs
        self.keep_similarity_epochs = max(0, int(keep_similarity_epochs))
        self.print_each_epochs = max(1, int(print_each_epochs))
        self.print_limit_epochs = max(1, int(print_limit_epochs))
        self.save_results_each = save_results_each
        self.logging_dir = logging_dir

        self.tracking = Tracker(
            errors_names=list(self.loss.errors_dict.keys()),
            device=track_device,
            keep_epochs=track_epochs,
            path=logging_dir,
            config_info=config_info,
            loss_aggregation=self.loss.loss_aggregation,
            errors_aggregation=self.loss.errors_aggregation,
        )

        self.printer = _MultiTaskTrainerPrint(
            learning_type=learning_type,
            dataset_name=dataset_name,
            total_epochs=0,
            n_tasks=self.model.n_tasks,
            logging_dir=logging_dir,
            seed=seed,
            early_stopping_rounds=early_stopping_epochs,
            n_intervals_ale=n_intervals_ale,
            learning_rate=learning_rate,
            similarity_each_epochs=similarity_each_epochs,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            ale_batch_size=ale_batch_size,
            l2penalty=l2penalty,
            architecture=architecture,
        )

        self._best_seen: float | None = None
        self.amp_dtype = str(amp_dtype).lower()
        self.amp = self._amp_enabled(amp)

    def _amp_enabled(self, requested: bool) -> bool:
        return bool(requested and torch.cuda.is_available() and str(self.model.device).startswith("cuda"))

    def _amp_context(self):
        if not self.amp:
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _validate_epochs(epochs: int | None) -> int:
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        return epochs

    @staticmethod
    def _max_batches(dataloader, max_batches: int | None) -> int:
        total = len(dataloader)
        if max_batches is None:
            return total
        if max_batches < 0:
            raise ValueError("max_batches must be non-negative or None.")
        return min(int(max_batches), total)

    def _is_improvement(self, current_loss: float) -> bool:
        if self._best_seen is None:
            return True
        return current_loss > self._best_seen if self.maximize_loss else current_loss < self._best_seen

    def _should_stop_early(self, epoch: int) -> bool:
        if self.tracking.best_epoch < 0:
            return False
        return (epoch - self.tracking.best_epoch) >= self.early_stopping_epochs

    def _safe_save(self) -> None:
        if self.logging_dir:
            self.tracking.save()

    def train(self, epochs: int = None, max_batches: int | None = None) -> None:
        """Run the training loop with validation, optional ALE, and similarity."""

        epochs = self._validate_epochs(epochs)
        self.printer.total_epochs = epochs
        self.printer.info_model()

        for epoch in range(epochs):
            self.tracking.start_epoch()

            self.train_step(max_batches)
            self.validation_step(max_batches)
            self.ale_step(epoch)
            self.similarity_step(epoch)

            if ((epoch + 1) % self.print_each_epochs) == 0:
                self._print_progress(epoch, epochs)

            current_loss = self._last_validation_loss()
            if self._is_improvement(current_loss):
                self._best_seen = current_loss
                self.tracking.update_best_model(self.model.state_dict(), epoch, best_loss_value=current_loss)
            elif self._should_stop_early(epoch):
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"No improvement for {self.early_stopping_epochs} epochs."
                )
                break

            if self.save_results_each is not None and ((epoch + 1) % self.save_results_each == 0):
                self._safe_save()

            self._step_scheduler(current_loss)

        self.test_step()
        self._safe_save()
        if self.tracking.best_epoch >= 0:
            self.printer.best_model_banner(self.tracking.best_model())

    def _print_progress(self, epoch: int, epochs: int) -> None:
        metric_names = ["LOSS"] + list(self.loss.errors_dict.keys())
        from_epoch = max(0, epoch - (self.print_limit_epochs - 1))
        self.printer.start_epoch_banner(epoch, epochs)
        self.printer.epoch_table(
            self.tracking.track,
            metric_names,
            from_epoch,
            epoch,
            self.similarity_each_epochs,
        )
        self.printer.times(self.tracking.track, epoch, epochs)

    def _last_validation_loss(self) -> float:
        value = self.tracking.last_val_loss()
        return float(value.item()) if torch.is_tensor(value) else float(value)

    def _step_scheduler(self, current_loss: float) -> None:
        if self.scheduler is None:
            return
        try:
            self.scheduler.step(current_loss)
        except TypeError:
            self.scheduler.step()

    # Steps ----------------------------------------------
    def train_step(self, max_batches: int | None = None) -> None:
        """Run one training phase."""

        self.tracking.start_train()
        try:
            self.model.train()
            self._step("train", self.train_dataloader, self.model, max_batches=max_batches)
        finally:
            self.tracking.end_train()

    def test_val_step(self, phase: str, model, dataloader, max_batches: int | None = None) -> None:
        """Run one evaluation phase without gradient tracking."""

        with torch.no_grad():
            model.eval()
            self._step(phase, dataloader, model, max_batches=max_batches)

    def validation_step(self, max_batches: int | None = None) -> None:
        self.tracking.start_validation()
        try:
            self.test_val_step("validation", self.model, self.validation_dataloader, max_batches=max_batches)
        finally:
            self.tracking.end_validation()

    def test_step(self, max_batches: int | None = None) -> None:
        self.tracking.start_test()
        try:
            if self.tracking.best_model_parameters is not None:
                self.model.load_state_dict(self.tracking.best_model_parameters)
            self.test_val_step("test", self.model, self.test_dataloader, max_batches=max_batches)
        finally:
            self.tracking.end_test()

    def _step(
        self,
        phase: str,
        dataloader: MultitaskDataloader,
        model: MultiTaskModel,
        max_batches: int | None = None,
    ) -> None:
        """Run one train/validation/test phase over a dataloader."""

        if phase not in {"train", "validation", "test"}:
            raise ValueError("phase must be 'train', 'validation', or 'test'.")

        is_train = phase == "train"
        batches_to_process = self._max_batches(dataloader, max_batches)
        if batches_to_process == 0:
            return

        for batch_idx, (X, y) in enumerate(dataloader):
            if batch_idx >= batches_to_process:
                break

            X = X.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with self._amp_context():
                y_pred = model(X)
                loss_per_task = self.loss.loss_per_task(y, y_pred)
                penalty = self.loss.penalty()
                errors_per_task = self.loss.errors_per_task(y, y_pred)
                l2_penalty = self.loss.l2_penalty
                total_loss = self._total_train_loss(loss_per_task, penalty, l2_penalty) if is_train else None

            epoch_loss, epoch_errors = self.loss.epoch_values(y, y_pred, loss_per_task, errors_per_task)
            self.tracking.update_metrics(
                phase, epoch_loss, epoch_errors, penalty, l2_penalty, batch_size=y.size(1),
            )

            if is_train:
                total_loss.backward()
                self.optimizer.step()

    def _total_train_loss(self, loss_per_task: torch.Tensor, penalty: torch.Tensor, l2_penalty) -> torch.Tensor:
        mean_loss = loss_per_task.mean()
        weighted_penalty = self._similarity_weighted_penalty(penalty)
        l2 = l2_penalty.item() if torch.is_tensor(l2_penalty) else float(l2_penalty)
        return mean_loss + l2 * weighted_penalty.sum()

    def _similarity_weighted_penalty(self, penalty: torch.Tensor) -> torch.Tensor:
        if self.multitask_similarity is None:
            return penalty

        similarity, _ = self.multitask_similarity.tasks_groups()
        similarity = similarity.to(device=penalty.device, dtype=penalty.dtype)
        if similarity.numel() != penalty.numel():
            raise RuntimeError(
                "Similarity scores and penalty vector must have the same length, "
                f"got {similarity.numel()} and {penalty.numel()}."
            )
        return penalty * similarity

    @torch.no_grad()
    def ale_step(self, epoch: int) -> None:
        """Optionally computes and records ALE curves at a scheduled frequency.

        If ``ale_each_epochs`` is set and ``ale`` is provided, this recomputes
        intervals and effects from the current model and stores the curves in
        the ``Tracker``. Effects from previous epochs are discarded.

        Args:
            epoch: Zero-based current epoch index.
        """
        self.tracking.start_ale()
        try:
            if self._is_scheduled(epoch, self.ale_each_epochs) and self.ale is not None:
                self.ale.recompute()
                self.tracking.end_ale(epoch, self.ale())
            else:
                self.tracking.end_ale()
        except Exception:
            self.tracking.end_ale()
            raise

    @torch.no_grad()
    def similarity_step(self, epoch: int) -> None:
        """Optionally computes and records task-feature similarity.

        Computes similarity at the configured frequency; if the loss exposes
        ``update_tasks_groups(groups)``, it will receive the new grouping and
        subsequently be cleared according to ``keep_similarity_epochs``.

        Args:
            epoch: Zero-based current epoch index.
        """
        self.tracking.start_similarity()
        try:
            if self._is_scheduled(epoch, self.similarity_each_epochs) and self.multitask_similarity is not None:
                self.multitask_similarity.compute()
                _, groups = self.multitask_similarity.tasks_groups()
                self._update_loss_task_groups(groups)
                self.tracking.end_similarity(epoch, self.multitask_similarity.similarity_tasks_features)
            else:
                self._clear_stale_similarity_groups(epoch)
                self.tracking.end_similarity()
        except Exception:
            self.tracking.end_similarity()
            raise

    # Helpers ----------------------------------------------
    @staticmethod
    def _is_scheduled(epoch: int, frequency: int | None) -> bool:
        return frequency is not None and frequency > 0 and ((epoch + 1) % frequency == 0)

    def _update_loss_task_groups(self, groups) -> None:
        if hasattr(self.loss, "update_tasks_groups"):
            self.loss.update_tasks_groups(groups)

    def _clear_stale_similarity_groups(self, epoch: int) -> None:
        if self.similarity_each_epochs is None or self.similarity_each_epochs <= 0:
            return
        if self.keep_similarity_epochs <= 0:
            self._update_loss_task_groups(None)
            return

        epochs_since_similarity = (epoch + 1) % self.similarity_each_epochs
        if epochs_since_similarity >= self.keep_similarity_epochs:
            self._update_loss_task_groups(None)

    def _start_epoch(self, epoch: int, total_epochs: int) -> None:
        """Starts timing/trackers for an epoch and prints the epoch banner.

        Args:
            epoch: Zero-based current epoch index.
            total_epochs: Total number of epochs planned.
        """
        self.tracking.start_epoch()
        self.printer.start_epoch_banner(epoch, total_epochs)

    def get_best_model(self) -> dict:
        """Returns the best model metrics and parameters recorded by the tracker.

        Returns:
            dict: State dictionary of the best model seen during training.
        """
        return self.tracking.best_model()
