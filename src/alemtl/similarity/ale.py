"""Accumulated Local Effects helpers for multitask similarity.

This module computes ALE curves for the similarity slice of a
:class:`~alemtl.models.multitask_model.MultiTaskModel`. The curves can then be
compared across tasks by :mod:`alemtl.similarity.similarity`.

ALE answers a local question: for samples that fall inside one feature interval,
how much does the model output change when that feature is moved from the left
edge of the interval to the right edge, while all other features stay fixed?
Those local differences are averaged per interval and optionally accumulated,
centered, standardized, and smoothed.

Shape contract
--------------
The code follows the project-wide task-first convention:

* dataloader batch features: ``(n_tasks, batch, n_features)``
* similarity input features: ``(n_tasks, batch, n_similarity_features)``
* perturbations for one task: ``(batch, n_similarity_features, n_similarity_features)``
* task similarity output for perturbations: ``(2 * batch, n_similarity_features, n_outputs)``
* returned curves: ``(n_tasks, n_similarity_features, num_intervals, 1 + n_outputs)``

The last curve channel stores the x-grid in channel ``0`` and ALE values in
channels ``1:``. For a single-output model this means each curve point is
``(x, y)`` and can be fed directly to the Frechet-distance similarity helper.

Main classes
------------
``Intervals``
    Builds and updates per-feature interval boundaries, maps input values to
    interval indices, and returns left/right interval bounds.

``MultiTaskALE``
    Drives the full ALE workflow over a dataloader and a ``MultiTaskModel``.

Minimal example
---------------
The example below is intentionally small and uses an in-memory iterable instead
of the project dataloader. Real training code should pass the ``ale`` split
created by :func:`alemtl.data.dataloader.create_dataloaders`.

.. code-block:: python

    import torch
    from torch import nn

    from alemtl.models.multitask_model import MultiTaskModel
    from alemtl.similarity.ale import MultiTaskALE

    layout = {
        "encoder": {
            "shared": "hard",
            "module": lambda: nn.Sequential(nn.Linear(3, 8), nn.ReLU()),
        },
        "head": {
            "shared": "soft",
            "module": lambda: nn.Linear(8, 1),
        },
    }

    model = MultiTaskModel(
        n_tasks=2,
        modules_layout=layout,
        similarity_layers={"in": "encoder", "out": "head"},
    )

    batches = [
        (torch.randn(2, 16, 3), torch.randn(2, 16, 1))
        for _ in range(4)
    ]

    ale = MultiTaskALE(
        model=model,
        dataloader=batches,
        n_tasks=2,
        num_intervals=10,
        n_features_out=1,
    )
    ale.update()
    curves = ale(centered=True, cumulative=True)
    print(curves.shape)  # (2, 3, 10, 2)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader

from ..models.multitask_model import MultiTaskModel

try:
    from scipy.interpolate import UnivariateSpline
except ImportError:  # pragma: no cover - exercised only when scipy is absent.
    UnivariateSpline = None


class Intervals:
    """Feature-wise interval boundaries for ALE.

    ``Intervals`` is a small utility used by :class:`MultiTaskALE`. It
    discretizes each input feature independently and keeps a boundary vector per
    feature. Boundaries are stored as a list, not a single rectangular tensor,
    because low-cardinality features may produce fewer intervals than
    ``num_intervals``.

    Boundary semantics
    ------------------
    For a feature boundary vector ``b`` of length ``L``, valid interval indices
    are ``0`` through ``L - 2``. Index ``j`` represents:

    ``b[j] < x <= b[j + 1]``

    The first and last boundaries are expanded by ``epsilon`` so values exactly
    equal to the observed minimum or maximum still fall into a valid bin.

    Parameters
    ----------
    X:
        Two-dimensional tensor with shape ``(n_samples, n_features)``. This is
        usually one task slice of the similarity-input representation.
    num_intervals:
        Target number of intervals for quantile-based features. The actual
        number of intervals can be smaller if a feature has few unique values or
        duplicate quantile boundaries.
    device:
        Device where boundary tensors and returned tensors are stored.
    epsilon:
        Small margin subtracted from the first boundary and added to the last
        boundary. It also controls how far boundaries are expanded by
        :meth:`update`.

    Attributes
    ----------
    intervals:
        List of length ``n_features``. Each entry is a one-dimensional boundary
        tensor.
    n_features:
        Number of feature columns in ``X``.
    num_intervals:
        Requested target interval count.
    device:
        Torch device where this object stores tensors.
    epsilon:
        Boundary expansion margin.

    Examples
    --------
    Basic interval construction and lookup:

    .. code-block:: python

        import torch
        from alemtl.similarity.ale import Intervals

        X = torch.tensor([
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [3.0, 40.0],
        ])

        intervals = Intervals(X, num_intervals=2)
        print(intervals.n_features)  # 2
        print(intervals.n_intervals_per_feature())

        idx = intervals.compute_index_intervals(X)
        print(idx.shape)  # (4, 2, 1)

        bounds = intervals.bounds(X, indices_left=idx)
        print(bounds.shape)  # (4, 2, 2)

    Updating boundaries for later batches:

    .. code-block:: python

        next_batch = torch.tensor([
            [-1.0, 5.0],
            [4.0, 50.0],
        ])
        intervals.update(next_batch)
        counts = intervals.cardinality(next_batch)
    """

    def __init__(
        self,
        X: torch.Tensor,
        num_intervals: int = 30,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-4,
    ) -> None:
        """Initialize per-feature interval boundaries.

        The constructor validates ``X``, moves it to ``device``, creates the
        initial boundary vectors, and caches the first/last boundaries so future
        calls to :meth:`update` can cheaply expand only the extremes.

        Raises
        ------
        ValueError
            If ``X`` is not two-dimensional, has zero samples, or
            ``num_intervals < 1``.
        """
        self.device = torch.device(device)
        self.num_intervals = int(num_intervals)
        self.epsilon = float(epsilon)

        if self.num_intervals < 1:
            raise ValueError("num_intervals must be >= 1")

        X = self._validate_input(X).to(self.device)
        self.intervals = self._create(X)
        self.n_features = len(self.intervals)

        self._first_cache = torch.stack([bounds[0] for bounds in self.intervals]).to(self.device)
        self._last_cache = torch.stack([bounds[-1] for bounds in self.intervals]).to(self.device)

    @staticmethod
    def _validate_input(X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 2:
            raise ValueError(f"X must have shape (n_samples, n_features), got {tuple(X.shape)}.")
        if X.size(0) == 0:
            raise ValueError("X must contain at least one sample.")
        return X

    @torch.no_grad()
    def _create(self, X: torch.Tensor) -> list[Tensor]:
        """Create monotone boundary vectors for every feature.

        Features with ``unique_values <= num_intervals`` use their sorted unique
        values directly. Higher-cardinality features use an evenly spaced
        quantile grid. Duplicate quantile boundaries are removed with
        ``unique(sorted=True)`` so constant or repeated features remain valid.
        """

        quantiles = torch.linspace(0, 1, steps=self.num_intervals + 1, device=self.device)
        intervals: list[Tensor] = []

        for feature in range(X.size(1)):
            values = X[:, feature]
            unique_values = values.unique(sorted=True)

            if unique_values.numel() <= self.num_intervals:
                boundaries = torch.cat([unique_values[:1], unique_values]).clone()
            else:
                boundaries = values.quantile(quantiles, interpolation="higher").unique(sorted=True)

            if boundaries.numel() == 1:
                boundaries = boundaries.repeat(2)

            boundaries[0] = boundaries[0] - self.epsilon
            boundaries[-1] = boundaries[-1] + self.epsilon
            intervals.append(boundaries)

        return intervals

    @torch.no_grad()
    def update(self, X: torch.Tensor) -> None:
        """Expand first/last boundaries to cover new data.

        Only the outer boundaries are changed. Internal quantile boundaries stay
        fixed so accumulated ALE bins remain comparable across batches.

        Parameters
        ----------
        X:
            New batch with shape ``(n_samples, n_features)``.

        Examples
        --------
        .. code-block:: python

            intervals = Intervals(torch.randn(100, 3), num_intervals=5)
            intervals.update(torch.randn(32, 3) * 2.0)
        """

        X = self._validate_input(X).to(self.device)
        if X.size(1) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.size(1)}.")

        batch_min = X.amin(dim=0) - self.epsilon
        batch_max = X.amax(dim=0) + self.epsilon

        new_first = torch.minimum(self._first_cache, batch_min)
        new_last = torch.maximum(self._last_cache, batch_max)

        left_updates = (new_first != self._first_cache).nonzero(as_tuple=False).flatten()
        right_updates = (new_last != self._last_cache).nonzero(as_tuple=False).flatten()

        for feature in left_updates.tolist():
            self.intervals[feature][0] = new_first[feature]
        for feature in right_updates.tolist():
            self.intervals[feature][-1] = new_last[feature]

        if left_updates.numel():
            self._first_cache[left_updates] = new_first[left_updates]
        if right_updates.numel():
            self._last_cache[right_updates] = new_last[right_updates]

    def compute_boolean_mask(self, X: torch.Tensor) -> list[Tensor]:
        """Return per-feature interval membership masks.

        Returns
        -------
        list[torch.Tensor]
            One tensor per feature. For feature ``f`` the tensor has shape
            ``(n_samples, n_intervals_f)`` and ``mask[i, j]`` is ``True`` when
            sample ``i`` falls into interval ``j`` for feature ``f``.

        Notes
        -----
        This is convenient for inspection and cardinality checks. The ALE update
        loop uses :meth:`compute_index_intervals`, which is more compact for
        scatter/index-add accumulation.
        """

        X = self._validate_input(X).to(self.device)
        if X.size(1) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.size(1)}.")

        return [
            (X[:, feature].unsqueeze(-1) > self.intervals[feature][:-1])
            & (X[:, feature].unsqueeze(-1) <= self.intervals[feature][1:])
            for feature in range(self.n_features)
        ]

    def compute_index_intervals(self, X: torch.Tensor) -> torch.Tensor:
        """Return left interval indices with shape ``(n_samples, n_features, 1)``.

        The returned index ``idx[i, f, 0]`` is the left boundary index for
        sample ``i`` and feature ``f``. It is clamped to valid interval bins, so
        values outside the initial data range after boundary updates still map
        to a valid accumulator position.

        Examples
        --------
        .. code-block:: python

            X = torch.tensor([[0.0], [1.0], [2.0]])
            intervals = Intervals(X, num_intervals=2)
            idx = intervals.compute_index_intervals(torch.tensor([[2.0]]))
            assert idx.shape == (1, 1, 1)
        """

        X = self._validate_input(X).to(self.device)
        if X.size(1) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.size(1)}.")

        n_samples = X.size(0)
        by_feature = X.transpose(0, 1).contiguous()
        indices = []

        for feature in range(self.n_features):
            boundaries = self.intervals[feature]
            left_idx = torch.bucketize(by_feature[feature], boundaries, right=False) - 1
            left_idx = left_idx.clamp_(0, boundaries.numel() - 2)
            indices.append(left_idx.view(n_samples, 1))

        return torch.cat(indices, dim=1).unsqueeze(-1)

    @torch.no_grad()
    def bounds(self, X: torch.Tensor, indices_left: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return left/right bounds with shape ``(n_samples, n_features, 2)``.

        ``out[i, f, 0]`` is the left boundary for sample ``i`` and feature
        ``f``. ``out[i, f, 1]`` is the corresponding right boundary. Passing
        precomputed ``indices_left`` avoids duplicate bucketization.
        """

        X = self._validate_input(X).to(self.device)
        if indices_left is None:
            indices_left = self.compute_index_intervals(X)
        else:
            indices_left = indices_left.to(self.device)

        n_samples, n_features, _ = indices_left.shape
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}.")

        out = torch.empty((n_samples, n_features, 2), device=self.device, dtype=X.dtype)
        for feature in range(self.n_features):
            boundaries = self.intervals[feature]
            idx = indices_left[:, feature, 0].clamp(0, boundaries.numel() - 2)
            out[:, feature, 0] = boundaries[idx]
            out[:, feature, 1] = boundaries[idx + 1]

        return out

    def cardinality(self, X: torch.Tensor) -> list[Tensor]:
        """Count samples per interval for every feature.

        This method first calls :meth:`update`, then computes interval
        membership counts. It returns variable-length tensors because different
        features can have different numbers of effective intervals.
        """

        self.update(X)
        masks = self.compute_boolean_mask(X)
        return [mask.sum(dim=0) for mask in masks]

    def n_intervals_per_feature(self) -> torch.Tensor:
        """Return the number of intervals available for each feature.

        Returns
        -------
        torch.Tensor
            One-dimensional ``long`` tensor of shape ``(n_features,)``.
        """

        return torch.tensor(
            [bounds.numel() - 1 for bounds in self.intervals],
            device=self.device,
            dtype=torch.long,
        )


class MultiTaskALE:
    """Accumulated Local Effects estimator for a :class:`MultiTaskModel`.

    The estimator operates on the model's configured similarity slice:

    1. Run ``model.model_similarity_input`` on each dataloader batch.
    2. Build left/right perturbations for each feature interval.
    3. Run each task's ``model_similarity`` view on the perturbations.
    4. Accumulate average local effects into fixed ``(task, feature, interval,
       output)`` tensors.

    The class is stateful. Construction initializes interval boundaries from a
    small sample of the dataloader. Calling :meth:`update` accumulates local
    effects and counts into ``g_ale_per_feature`` and ``cardinality``. Calling
    the object, for example ``ale()``, converts those accumulators into curves.

    Parameters
    ----------
    model:
        A :class:`MultiTaskModel` exposing ``model_similarity_input`` and
        ``model_by_task(task)["model_similarity"]``.
    dataloader:
        Iterable yielding feature/target batches or feature-only batches. When
        a tuple/list is yielded, the first element is used as features. Feature
        tensors must follow the project task-first contract:
        ``(n_tasks, batch, n_features)``.
    n_tasks:
        Number of tasks represented in each batch and in ``model``.
    shared_input_data:
        Set to ``True`` when every task receives the same input data. In this
        mode one shared :class:`Intervals` object is created and cardinality is
        counted once per batch, while effects are still accumulated per task.
    n_features_out:
        Number of model outputs produced by the similarity slice. For scalar
        regression this is usually ``1``.
    num_intervals:
        Fixed accumulator width. Returned curves always use this many interval
        points, padding low-cardinality feature grids with repeated right
        edges.
    device:
        Device where computations and accumulators are stored.
    epsilon:
        Margin passed to :class:`Intervals` for boundary expansion.
    n_guess:
        Approximate number of transformed samples used at construction time to
        initialize interval boundaries.

    Attributes
    ----------
    g_ale_per_feature:
        Tensor with shape ``(n_tasks, n_features_in, num_intervals,
        n_features_out)``. It stores summed local effects before division by
        cardinality.
    cardinality:
        Tensor with shape ``(n_tasks, n_features_in, num_intervals)``. In
        ``shared_input_data=True`` mode, only row ``0`` is used for counts and
        is shared by every task during readout.
    intervals:
        List of :class:`Intervals` objects. Length is ``n_tasks`` for ordinary
        multitask data and ``1`` for shared-input data.
    n_features_in:
        Number of features entering the similarity slice.

    Examples
    --------
    Standard task-specific inputs:

    .. code-block:: python

        import torch
        from torch import nn

        from alemtl.models.multitask_model import MultiTaskModel
        from alemtl.similarity.ale import MultiTaskALE

        n_tasks = 2
        layout = {
            "encoder": {
                "shared": "hard",
                "module": lambda: nn.Sequential(nn.Linear(4, 6), nn.ReLU()),
            },
            "head": {
                "shared": "soft",
                "module": lambda: nn.Linear(6, 1),
            },
        }
        model = MultiTaskModel(
            n_tasks=n_tasks,
            modules_layout=layout,
            similarity_layers={"in": "encoder", "out": "head"},
        )

        dataloader = [
            (torch.randn(n_tasks, 32, 4), torch.randn(n_tasks, 32, 1))
            for _ in range(5)
        ]

        ale = MultiTaskALE(model, dataloader, n_tasks=n_tasks, num_intervals=8)
        ale.update()
        curves = ale(centered=True, cumulative=True, std=1.0)
        assert curves.shape == (n_tasks, 4, 8, 2)

    Shared input data:

    .. code-block:: python

        shared_model = MultiTaskModel(
            n_tasks=3,
            modules_layout=layout,
            similarity_layers={"in": "encoder", "out": "head"},
            shared_input_data=True,
        )
        shared_batch = torch.randn(16, 4)
        dataloader = [(shared_batch, torch.randn(3, 16, 1))]

        ale = MultiTaskALE(
            shared_model,
            dataloader,
            n_tasks=3,
            shared_input_data=True,
            num_intervals=8,
        )
        ale.update()
        curves = ale()
    """

    def __init__(
        self,
        model: MultiTaskModel,
        dataloader: DataLoader,
        n_tasks: int,
        shared_input_data: bool = False,
        n_features_out: int = 1,
        num_intervals: int = 30,
        device: str | torch.device = "cpu",
        epsilon: float = 1e-4,
        n_guess: int = 100,
    ) -> None:
        """Initialize interval boundaries and zero-filled ALE accumulators.

        Initialization consumes batches from ``dataloader`` until roughly
        ``n_guess`` transformed samples have been collected. The dataloader must
        therefore be re-iterable if you plan to call :meth:`update` afterwards.
        Lists, tuples, PyTorch dataloaders, and the project
        ``MultitaskDataloader`` satisfy this requirement.

        Raises
        ------
        ValueError
            If task/output/interval/sample counts are not positive.
        RuntimeError
            If the dataloader yields no batches or the model similarity input
            has an incompatible shape.
        """
        self.model = model
        self.dataloader = dataloader
        self.n_tasks = int(n_tasks)
        self.shared_input_data = bool(shared_input_data)
        self.n_features_out = int(n_features_out)
        self.num_intervals = int(num_intervals)
        self.device = torch.device(device)
        self.epsilon = float(epsilon)
        self.n_guess = int(n_guess)

        if self.n_tasks <= 0:
            raise ValueError("n_tasks must be positive.")
        if self.n_features_out <= 0:
            raise ValueError("n_features_out must be positive.")
        if self.num_intervals < 1:
            raise ValueError("num_intervals must be >= 1.")
        if self.n_guess <= 0:
            raise ValueError("n_guess must be positive.")

        self.g_ale_per_feature, self.cardinality, self.intervals, self.n_features_in = self._reinit()

    # --------------------------- setup helpers ---------------------------

    @staticmethod
    def _batch_features(batch) -> torch.Tensor:
        """Extract feature tensors from dataloader batches.

        The project dataloaders yield ``(X, Y)``. Some tests and examples yield
        only ``X``. This helper accepts both forms and always returns features.
        """
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    def _similarity_input(self, X: torch.Tensor) -> torch.Tensor:
        """Run the model prefix before the configured similarity slice.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(n_tasks, batch, n_features_in)``.

        Notes
        -----
        If ``shared_input_data=True`` and the model prefix returns a raw
        ``(batch, features)`` tensor, it is expanded to the task-first shape.
        """
        features = self.model.model_similarity_input(X.to(self.device)).to(self.device)
        if features.dim() == 2 and self.shared_input_data:
            features = features.unsqueeze(0).expand(self.n_tasks, -1, -1)
        if features.dim() != 3:
            raise RuntimeError(
                "model_similarity_input must return shape (n_tasks, batch, n_features); "
                f"got {tuple(features.shape)}."
            )
        if features.size(0) != self.n_tasks:
            raise RuntimeError(f"Expected {self.n_tasks} task slices, got {features.size(0)}.")
        return features

    def _reinit(self) -> tuple[Tensor, Tensor, list[Intervals], int]:
        """Create fresh intervals and zero-filled accumulators.

        Used by initialization and :meth:`reset`. The caller installs the
        returned state on this object.
        """
        intervals, n_features_in = self._initialize_intervals()

        effects = torch.zeros(
            (self.n_tasks, n_features_in, self.num_intervals, self.n_features_out),
            device=self.device,
        )
        counts = torch.zeros(
            (self.n_tasks, n_features_in, self.num_intervals),
            device=self.device,
            dtype=torch.long,
        )

        return effects, counts, intervals, n_features_in

    def _initialize_intervals(self) -> tuple[list[Intervals], int]:
        """Build interval objects from a sampled similarity-input tensor.

        Returns
        -------
        tuple[list[Intervals], int]
            The interval objects and the number of features entering the
            similarity slice.
        """
        X_sample = self._X_sample()
        n_features_in = X_sample.size(-1)

        if self.shared_input_data:
            return [
                Intervals(
                    X_sample[0],
                    num_intervals=self.num_intervals,
                    device=self.device,
                    epsilon=self.epsilon,
                )
            ], n_features_in

        return [
            Intervals(
                X_sample[task],
                num_intervals=self.num_intervals,
                device=self.device,
                epsilon=self.epsilon,
            )
            for task in range(self.n_tasks)
        ], n_features_in

    @torch.no_grad()
    def _X_sample(self) -> Tensor:
        """Collect transformed samples to seed interval boundaries.

        The returned tensor has shape ``(n_tasks, n_sampled, n_features_in)``.
        Sampling happens after ``model.model_similarity_input`` so intervals are
        constructed in the exact feature space where ALE perturbations will be
        applied.
        """

        samples = []
        seen = 0

        for batch in self.dataloader:
            X = self._batch_features(batch)
            transformed = self._similarity_input(X)
            samples.append(transformed.detach())
            seen += transformed.size(1)
            if seen >= self.n_guess:
                break

        if not samples:
            raise RuntimeError("Dataloader produced no samples to initialize ALE intervals.")

        return torch.cat(samples, dim=1)

    # --------------------------- update ---------------------------

    @torch.no_grad()
    def reset(self) -> None:
        """Discard effects/counts and rebuild intervals for the current model.

        Samples the dataloader again, including the current model prefix when
        explaining latent features. No local effects are computed. Use
        :meth:`recompute` for a complete fresh explanation in evaluation mode.
        """
        self.g_ale_per_feature, self.cardinality, self.intervals, self.n_features_in = self._reinit()

    @torch.no_grad()
    def recompute(self, max_batches: Optional[int] = None) -> None:
        """Replace the accumulated ALE with an explanation of the current model.

        Rebuilds intervals, clears effects and counts, then processes at most
        ``max_batches`` batches (all batches when ``None``). Interval sampling
        still uses ``n_guess`` samples, independently of this limit. The
        dataloader must be re-iterable.

        Computation runs in evaluation mode, so dropout and batch-normalization
        updates do not alter the explanation. Each module's original training
        flag is restored even if computation fails. Read the result with
        ``ale(...)`` as usual. Unlike :meth:`update`, this method never mixes
        effects from earlier model states.
        """
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be non-negative or None.")

        training_modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            self.reset()
            self.update(max_batches=max_batches)
        finally:
            for module, training in training_modes:
                module.training = training

    @torch.no_grad()
    def update(self, max_batches: Optional[int] = None) -> None:
        """Accumulate ALE local effects over batches from the dataloader.

        Parameters
        ----------
        max_batches:
            Optional limit on processed batches. Use this for quick previews or
            tests. ``None`` processes the entire dataloader.

        Notes
        -----
        Updates ``g_ale_per_feature`` and ``cardinality`` in place. Repeated
        calls keep accumulating; they do not reset previous values. Use this
        only while the model is unchanged. Use :meth:`recompute` after model
        parameters change, or :meth:`reset` to rebuild empty accumulators and
        intervals without computing effects.

        Examples
        --------
        .. code-block:: python

            ale.update(max_batches=2)  # quick estimate
            preview = ale(centered=True, cumulative=True)

            ale.update()  # adds more effects on top of the preview pass
            curves = ale()
        """

        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be non-negative or None.")

        processed = 0
        interval_count = self.num_intervals

        for batch in self.dataloader:
            if max_batches is not None and processed >= max_batches:
                break
            processed += 1

            X = self._similarity_input(self._batch_features(batch))
            n_tasks, n_samples, n_features = X.shape
            if n_tasks != self.n_tasks or n_features != self.n_features_in:
                raise RuntimeError(
                    "Unexpected similarity-input shape "
                    f"{tuple(X.shape)}; expected ({self.n_tasks}, batch, {self.n_features_in})."
                )

            feature_ids_flat = (
                torch.arange(n_features, device=self.device)
                .unsqueeze(0)
                .expand(n_samples, n_features)
                .reshape(-1)
            )

            if self.shared_input_data:
                intervals = self.intervals[0]
                intervals.update(X[0])
                indices = intervals.compute_index_intervals(X[0])
                X_left, X_right = self._X_perturbation(X[0], intervals, indices)
                self.cardinality[0] += self._counts_from_indices(indices, interval_count)

                for task in range(self.n_tasks):
                    self._accumulate_task_effects(
                        task=task,
                        X_left=X_left,
                        X_right=X_right,
                        indices=indices,
                        feature_ids_flat=feature_ids_flat,
                    )
                continue

            for task in range(self.n_tasks):
                intervals = self.intervals[task]
                intervals.update(X[task])
                indices = intervals.compute_index_intervals(X[task])
                X_left, X_right = self._X_perturbation(X[task], intervals, indices)
                self.cardinality[task] += self._counts_from_indices(indices, interval_count)
                self._accumulate_task_effects(
                    task=task,
                    X_left=X_left,
                    X_right=X_right,
                    indices=indices,
                    feature_ids_flat=feature_ids_flat,
                )

    def _counts_from_indices(self, indices: Tensor, interval_count: int) -> Tensor:
        """Convert ``(N, F, 1)`` interval indices into ``(F, I)`` counts."""
        idx = indices.squeeze(-1).long()
        return torch_functional.one_hot(idx, num_classes=interval_count).sum(dim=0).to(torch.long)

    def _accumulate_task_effects(
        self,
        *,
        task: int,
        X_left: Tensor,
        X_right: Tensor,
        indices: Tensor,
        feature_ids_flat: Tensor,
    ) -> None:
        """Forward one task's perturbations and add effects to accumulators.

        ``X_left`` and ``X_right`` each have shape ``(N, F, F)``. They are
        concatenated along the sample axis and evaluated in one forward pass.
        The first ``N`` outputs are left-bound predictions and the second ``N``
        outputs are right-bound predictions. Their difference is scatter-added
        into the fixed ``(F, I, O)`` accumulator for ``task``.
        """
        n_samples, n_features, _ = X_left.shape
        X_cat = torch.cat([X_left, X_right], dim=0)
        Y_cat = self.model.model_by_task(task)["model_similarity"](X_cat)

        if Y_cat.dim() != 3:
            raise RuntimeError(f"model_similarity must return shape (2N, F, O), got {tuple(Y_cat.shape)}.")
        if Y_cat.size(0) != 2 * n_samples or Y_cat.size(1) != n_features:
            raise RuntimeError(
                "model_similarity returned incompatible shape "
                f"{tuple(Y_cat.shape)} for perturbation shape {tuple(X_cat.shape)}."
            )
        if Y_cat.size(-1) != self.n_features_out:
            raise RuntimeError(f"Expected {self.n_features_out} outputs, got {Y_cat.size(-1)}.")

        Y_left, Y_right = Y_cat[:n_samples], Y_cat[n_samples:]
        Y_diff = Y_right - Y_left

        idx_flat = indices.reshape(-1).long()
        bins_flat = feature_ids_flat * self.num_intervals + idx_flat
        y_flat = Y_diff.reshape(-1, self.n_features_out)

        accum_flat = torch.zeros(
            n_features * self.num_intervals,
            self.n_features_out,
            device=self.device,
            dtype=Y_diff.dtype,
        )
        accum_flat.index_add_(0, bins_flat, y_flat)
        self.g_ale_per_feature[task] += accum_flat.view(
            n_features,
            self.num_intervals,
            self.n_features_out,
        )

    # --------------------------- perturbation ---------------------------

    def _X_perturbation(
        self,
        X: torch.Tensor,
        interval: Intervals,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create left/right perturbation tensors with shape ``(N, F, F)``.

        For each original sample, this creates one perturbed copy per feature.
        In copy ``f``, only feature ``f`` is replaced by the left or right
        interval boundary. All other features remain equal to the original
        sample.

        Example for ``F=3``:

        ``X_left[i, 2]`` is sample ``i`` with only feature ``2`` moved to that
        feature's left interval boundary.
        """

        X = X.to(self.device)
        indices = indices.to(self.device)
        bounds = interval.bounds(X, indices_left=indices)

        n_features = X.size(-1)
        feature_ids = torch.arange(n_features, device=self.device)

        X_left = X.unsqueeze(1).expand(-1, n_features, -1).clone()
        X_right = X_left.clone()
        X_left[:, feature_ids, feature_ids] = bounds[:, :, 0]
        X_right[:, feature_ids, feature_ids] = bounds[:, :, 1]

        return X_left, X_right

    # --------------------------- readout ---------------------------

    def _get_ale(
        self,
        centered: bool = True,
        cumulative: bool = True,
        std: Optional[float] = None,
    ) -> torch.Tensor:
        """Return ALE values with shape ``(T, F, I, O)``.

        Parameters
        ----------
        centered:
            If ``True``, subtract each feature/output curve's mean along the
            interval axis. Centering is common for ALE because local effects are
            identifiable up to an additive constant.
        cumulative:
            If ``True``, cumulative-sum the averaged local effects along the
            interval axis. This produces ordinary ALE curves. If ``False``, the
            method returns non-cumulative average local effects per interval.
        std:
            Optional target standard deviation for each feature/output curve.
            Set to ``None`` to keep the original scale. A positive number such
            as ``1.0`` standardizes curves before cross-task comparison.
        """

        curves = []

        for task in range(self.n_tasks):
            counts = self.cardinality[0] if self.shared_input_data else self.cardinality[task]
            denom = counts.unsqueeze(-1).clamp(min=1).to(self.g_ale_per_feature.dtype)
            avg = torch.nan_to_num(self.g_ale_per_feature[task] / denom, nan=0.0)

            if cumulative:
                avg = avg.cumsum(dim=1)
            if centered:
                avg = avg - avg.mean(dim=1, keepdim=True)
            if std is not None:
                if std <= 0:
                    raise ValueError(f"std must be positive, got {std}.")
                scale = avg.std(dim=1, keepdim=True).clamp(min=1e-12)
                avg = (avg / scale) * std

            curves.append(torch.nan_to_num(avg, nan=0.0))

        return torch.stack(curves, dim=0)

    def __call__(
        self,
        centered: bool = True,
        cumulative: bool = True,
        std: Optional[float] = 1.0,
        spline: bool = False,
        spline_smooth: float = 1.0,
    ) -> torch.Tensor:
        """Return curves with shape ``(T, F, I, 1 + O)``.

        The final axis stores the interval right-edge grid in channel ``0`` and
        ALE output values in channels ``1:``.

        Parameters
        ----------
        centered:
            Passed to :meth:`_get_ale`.
        cumulative:
            Passed to :meth:`_get_ale`.
        std:
            Passed to :meth:`_get_ale`. Defaults to ``1.0`` because the
            downstream similarity metric compares curve geometry and often
            benefits from comparable curve scales.
        spline:
            If ``True``, smooth output channels with
            :meth:`smooth_ale_curves`. Requires SciPy.
        spline_smooth:
            Smoothing factor used by ``scipy.interpolate.UnivariateSpline``.

        Examples
        --------
        .. code-block:: python

            ale.update()
            curves = ale(centered=True, cumulative=True, std=1.0)
            x = curves[0, 0, :, 0]
            y = curves[0, 0, :, 1]

            raw_local_effects = ale(centered=False, cumulative=False, std=None)
        """

        values = self._get_ale(centered=centered, cumulative=cumulative, std=std)
        grid = self._interval_grid_right_edges()
        curves = torch.cat([grid.unsqueeze(-1), values], dim=-1)

        if spline:
            curves = self.smooth_ale_curves(curves, spline_smooth=spline_smooth)

        return curves

    def _interval_grid_right_edges(self) -> Tensor:
        """Build a fixed right-edge grid with shape ``(T, F, I)``.

        Features with fewer than ``num_intervals`` effective intervals are
        padded by repeating their last right edge. This keeps the returned ALE
        tensor rectangular while preserving monotonic x-values.
        """

        n_interval_sets = 1 if self.shared_input_data else self.n_tasks
        grids = []

        for task in range(n_interval_sets):
            intervals = self.intervals[task]
            feature_edges = []

            for feature in range(self.n_features_in):
                boundaries = intervals.intervals[feature]
                if boundaries.numel() < self.num_intervals + 1:
                    pad = boundaries[-1].repeat(self.num_intervals + 1 - boundaries.numel())
                    fixed_boundaries = torch.cat([boundaries, pad], dim=0)
                else:
                    fixed_boundaries = boundaries[: self.num_intervals + 1]
                feature_edges.append(fixed_boundaries[1:].unsqueeze(0))

            grids.append(torch.cat(feature_edges, dim=0).unsqueeze(0))

        grid = torch.cat(grids, dim=0).to(self.device)
        if self.shared_input_data:
            grid = grid.expand(self.n_tasks, -1, -1)
        return grid

    @staticmethod
    def smooth_ale_curves(ale_tensor: Tensor, spline_smooth: float = 1.0) -> Tensor:
        """Spline-smooth ALE output channels along the interval axis.

        Parameters
        ----------
        ale_tensor:
            Tensor with shape ``(n_tasks, n_features, num_intervals,
            1 + n_outputs)``. Channel ``0`` must contain the x-grid and is not
            smoothed.
        spline_smooth:
            Base smoothing factor. The implementation scales it by output
            variance and interval count before passing it to SciPy. Use ``0``
            for interpolation-like behavior, or larger values for smoother
            curves.

        Returns
        -------
        torch.Tensor
            A clone of ``ale_tensor`` with smoothed output channels. If one
            curve has a constant x-grid or SciPy cannot fit a spline for that
            curve, the original values for that curve are kept.

        Raises
        ------
        ImportError
            If SciPy is not installed.
        """

        if UnivariateSpline is None:
            raise ImportError("scipy is required for spline=True.")

        n_tasks, n_features, _, n_channels = ale_tensor.shape
        device = ale_tensor.device
        dtype = ale_tensor.dtype
        smoothed = ale_tensor.clone()

        for task in range(n_tasks):
            for feature in range(n_features):
                x = smoothed[task, feature, :, 0].detach().cpu().numpy()
                if np.allclose(x.max(), x.min()):
                    continue

                order = np.argsort(x)
                xs = np.asarray(x)[order]

                for channel in range(1, n_channels):
                    y = smoothed[task, feature, :, channel].detach().cpu().numpy()
                    ys = np.asarray(y)[order]

                    try:
                        smooth_factor = float(spline_smooth) * (np.var(ys) * max(len(xs), 2))
                        smooth_factor = max(smooth_factor, 1e-12)
                        degree = min(3, max(1, len(xs) - 1))
                        spline_model = UnivariateSpline(xs, ys, s=smooth_factor, k=degree)
                        yhat = spline_model(xs)
                        inverse = np.empty_like(order)
                        inverse[order] = np.arange(order.size)
                        y_smooth = yhat[inverse]
                    except Exception:
                        y_smooth = y

                    smoothed[task, feature, :, channel] = torch.tensor(
                        y_smooth,
                        device=device,
                        dtype=dtype,
                    )

        return smoothed
