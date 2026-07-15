"""Task similarity from multitask ALE curves.

The public workflow is:

1. Build/update a :class:`alemtl.similarity.ale.MultiTaskALE` instance.
2. Pass it to :class:`MultitaskSimilarity`.
3. Call :meth:`MultitaskSimilarity.compute`.
4. Use :meth:`MultitaskSimilarity.tasks_groups` to get nearest-task pairs for
   the parameter-sharing regularizer.

ALE curves are expected to have shape ``(T, F, I, 1 + O)``:

* ``T``: number of tasks
* ``F``: number of similarity-input features
* ``I``: number of ALE intervals
* ``O``: number of model outputs
* channel ``0``: x-grid
* channels ``1:``: ALE values

For single-output models each feature curve is a sequence of ``(x, y)`` points.
For multi-output models each output channel is compared separately and then
aggregated per feature.
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal, Optional

import torch

from .ale import MultiTaskALE

SimilarityReduction = Literal["mean", "sum", "max"]
SimilarityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _precompute_indices(m: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return anti-diagonal DP indices for the discrete Frechet recurrence."""

    diag_indices = []
    for diag in range(2, 2 * m - 1):
        i_start = max(1, diag - (m - 1))
        i_end_exclusive = min(m, diag)
        if i_start < i_end_exclusive:
            i_idx = torch.arange(i_start, i_end_exclusive, device=device)
            j_idx = diag - i_idx
            diag_indices.append((i_idx, j_idx))
    return diag_indices


def _normalize_curves(curve0: torch.Tensor, curve1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if curve0.shape != curve1.shape:
        raise ValueError(f"curve0 and curve1 must have the same shape, got {curve0.shape} and {curve1.shape}.")
    if curve0.dim() not in (2, 3):
        raise ValueError("curves must have shape (m, 2) or (batch, m, 2).")
    if curve0.size(-1) != 2:
        raise ValueError(f"curves must have last dimension size 2, got {curve0.size(-1)}.")

    squeeze_back = curve0.dim() == 2
    if squeeze_back:
        curve0 = curve0.unsqueeze(0)
        curve1 = curve1.unsqueeze(0)
    return curve0, curve1, squeeze_back


def discrete_frechet_distance_vectorized(curve0: torch.Tensor, curve1: torch.Tensor) -> torch.Tensor:
    """Compute the discrete Frechet distance for one or many 2D curves.

    Parameters
    ----------
    curve0, curve1:
        Tensors with shape ``(m, 2)`` or ``(batch, m, 2)``. Batched inputs are
        vectorized across the leading ``batch`` dimension.

    Returns
    -------
    torch.Tensor
        Scalar tensor for unbatched input, or shape ``(batch,)`` for batched
        input. Lower values mean more similar curves.
    """

    curve0, curve1, squeeze_back = _normalize_curves(curve0, curve1)
    _, m, _ = curve0.shape

    pairwise_distances = torch.cdist(
        curve0,
        curve1,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    dp = torch.empty_like(pairwise_distances)

    dp[:, 0, 0] = pairwise_distances[:, 0, 0]
    if m > 1:
        dp[:, 0, 1:] = torch.cummax(pairwise_distances[:, 0, 1:], dim=-1).values
        dp[:, 1:, 0] = torch.cummax(pairwise_distances[:, 1:, 0], dim=1).values

    for i_idx, j_idx in _precompute_indices(m, device=curve0.device):
        previous = torch.minimum(
            torch.minimum(dp[:, i_idx - 1, j_idx], dp[:, i_idx, j_idx - 1]),
            dp[:, i_idx - 1, j_idx - 1],
        )
        dp[:, i_idx, j_idx] = torch.maximum(pairwise_distances[:, i_idx, j_idx], previous)

    distances = dp[:, -1, -1]
    return distances.squeeze(0) if squeeze_back else distances


def frechet_similarity_vectorized(curve0: torch.Tensor, curve1: torch.Tensor) -> torch.Tensor:
    """Return ``exp(-discrete_frechet_distance)`` for one or many curves.

    Higher values mean more similar curves. Values are in ``(0, 1]`` when input
    coordinates are finite.
    """

    return torch.exp(-discrete_frechet_distance_vectorized(curve0, curve1))


def frechet_distance_vectorized(curve0: torch.Tensor, curve1: torch.Tensor) -> torch.Tensor:
    """Backward-compatible Frechet similarity helper.

    Historically this function was named like a distance but returned
    ``exp(-distance)``. The behavior is preserved because
    :class:`MultitaskSimilarity` and the trainer select nearest tasks by taking
    the maximum score. Use :func:`discrete_frechet_distance_vectorized` if you
    need the raw distance.
    """

    return frechet_similarity_vectorized(curve0, curve1)


class MultitaskSimilarity:
    """Compute pairwise task similarity from ALE curves.

    Parameters
    ----------
    ale_curves:
        A :class:`MultiTaskALE` instance. ``compute()`` calls this object to get
        curves shaped ``(T, F, I, 1 + O)``.
    similarity_func:
        Function comparing two batches of 2D curves. It must accept tensors with
        shape ``(batch, I, 2)`` and return one score per batch item. Higher
        scores must mean more similar tasks.
    centered, cumulative, std, spline:
        Options forwarded to ``ale_curves(...)``.
    complete:
        If ``True``, compute every upper-triangular task pair. If ``False``,
        currently behaves the same; the argument is kept for API compatibility.
    output_reduction:
        How to combine per-output similarities when ALE has more than one model
        output. ``"mean"`` is scale-stable, ``"sum"`` weights multi-output
        models more strongly, and ``"max"`` keeps the closest output channel.
    spline_smooth:
        Smoothing factor forwarded when ``spline=True``.

    Attributes
    ----------
    similarity_tasks_features:
        Tensor with shape ``(T, T, F)``. Entry ``[i, j, f]`` is the aggregated
        similarity between task ``i`` and task ``j`` for feature ``f``.
    """

    def __init__(
        self,
        ale_curves: MultiTaskALE,
        similarity_func: SimilarityFunction = frechet_distance_vectorized,
        centered: bool = True,
        cumulative: bool = True,
        std: Optional[float] = 1.0,
        complete: bool = True,
        spline: bool = False,
        output_reduction: SimilarityReduction = "mean",
        spline_smooth: float = 1.0,
    ) -> None:
        self.device = ale_curves.device
        self.n_tasks = int(ale_curves.n_tasks)
        self.num_intervals = int(ale_curves.num_intervals)
        self.n_features_in = int(ale_curves.n_features_in)
        self.n_features_out = int(ale_curves.n_features_out)
        self.ale_curves = ale_curves

        self.centered = bool(centered)
        self.cumulative = bool(cumulative)
        self.std = std
        self.complete = bool(complete)
        self.spline = bool(spline)
        self.spline_smooth = float(spline_smooth)

        if output_reduction not in ("mean", "sum", "max"):
            raise ValueError("output_reduction must be 'mean', 'sum', or 'max'.")
        self.output_reduction = output_reduction
        self.similarity_func = similarity_func

        self._scores = torch.zeros(
            (self.n_tasks, self.n_tasks),
            device=self.device,
            dtype=torch.float32,
        )
        self.similarity_tasks_features = torch.zeros(
            (self.n_tasks, self.n_tasks, self.n_features_in),
            device=self.device,
            dtype=torch.float32,
        )

    @property
    def scores(self) -> torch.Tensor:
        """Pairwise task similarity scores with shape ``(T, T)``."""

        return self._scores

    @scores.setter
    def scores(self, value: torch.Tensor) -> None:
        self._scores = value.to(device=self.device, dtype=torch.float32)

    @property
    def matrix(self) -> torch.Tensor:
        """Alias for :attr:`scores`."""

        return self.scores

    @matrix.setter
    def matrix(self, value: torch.Tensor) -> None:
        self.scores = value

    @property
    def similarity_matrix(self) -> torch.Tensor:
        """Alias for :attr:`scores`."""

        return self.scores

    @similarity_matrix.setter
    def similarity_matrix(self, value: torch.Tensor) -> None:
        self.scores = value

    @property
    def distances(self) -> torch.Tensor:
        """Deprecated alias for :attr:`scores`."""

        warnings.warn(
            "MultitaskSimilarity.distances is deprecated because it stores "
            "similarity scores, not raw distances. Use .scores, .matrix, or "
            ".similarity_matrix instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scores

    @distances.setter
    def distances(self, value: torch.Tensor) -> None:
        warnings.warn(
            "MultitaskSimilarity.distances is deprecated because it stores "
            "similarity scores, not raw distances. Use .scores, .matrix, or "
            ".similarity_matrix instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.scores = value

    def _validate_curves(self, curves: torch.Tensor) -> torch.Tensor:
        curves = curves.to(self.device)
        if curves.dim() != 4:
            raise ValueError(f"ALE curves must have shape (T, F, I, 1 + O), got {tuple(curves.shape)}.")
        if curves.size(0) != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} tasks, got {curves.size(0)}.")
        if curves.size(1) != self.n_features_in:
            raise ValueError(f"Expected {self.n_features_in} features, got {curves.size(1)}.")
        if curves.size(2) != self.num_intervals:
            raise ValueError(f"Expected {self.num_intervals} intervals, got {curves.size(2)}.")
        if curves.size(3) != 1 + self.n_features_out:
            raise ValueError(f"Expected {1 + self.n_features_out} curve channels, got {curves.size(3)}.")
        return curves

    def _curves_for_similarity(self, task_curves: torch.Tensor) -> torch.Tensor:
        """Convert one task's ALE tensor to ``(F, O, I, 2)`` curves."""

        x = task_curves[:, :, 0]
        y = task_curves[:, :, 1:]
        x = x.unsqueeze(1).expand(-1, self.n_features_out, -1)
        y = y.permute(0, 2, 1)
        return torch.stack([x, y], dim=-1)

    def _reduce_outputs(self, scores: torch.Tensor) -> torch.Tensor:
        if self.output_reduction == "mean":
            return scores.mean(dim=1)
        if self.output_reduction == "sum":
            return scores.sum(dim=1)
        return scores.max(dim=1).values

    def _compute_by_feature(self, ale0: torch.Tensor, ale1: torch.Tensor) -> torch.Tensor:
        """Compute aggregated feature similarities for two task curve tensors.

        ``ale0`` and ``ale1`` must each have shape ``(F, I, 1 + O)``. The
        returned tensor has shape ``(F,)``.
        """

        curves0 = self._curves_for_similarity(ale0)
        curves1 = self._curves_for_similarity(ale1)
        flat0 = curves0.reshape(self.n_features_in * self.n_features_out, self.num_intervals, 2)
        flat1 = curves1.reshape(self.n_features_in * self.n_features_out, self.num_intervals, 2)

        scores = self.similarity_func(flat0, flat1).to(dtype=torch.float32, device=self.device)
        scores = scores.view(self.n_features_in, self.n_features_out)
        return self._reduce_outputs(scores)

    def compute(self) -> None:
        """Compute pairwise task similarities from current ALE curves."""

        curves = self.ale_curves(
            centered=self.centered,
            cumulative=self.cumulative,
            std=self.std,
            spline=self.spline,
            spline_smooth=self.spline_smooth,
        )
        curves = self._validate_curves(curves)

        self.scores.zero_()
        self.similarity_tasks_features.zero_()

        with torch.inference_mode():
            for task0 in range(self.n_tasks):
                for task1 in range(task0 + 1, self.n_tasks):
                    feature_scores = self._compute_by_feature(curves[task0], curves[task1])
                    self.similarity_tasks_features[task0, task1] = feature_scores
                    self.similarity_tasks_features[task1, task0] = feature_scores

                    task_score = feature_scores.sum()
                    self.scores[task0, task1] = task_score
                    self.scores[task1, task0] = task_score

    def tasks_groups(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return each task's most similar peer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(similarity, pairs)`` where ``similarity`` has shape ``(T,)`` and
            ``pairs`` has shape ``(T, 2)``. Each row in ``pairs`` is
            ``(task, nearest_task)``.
        """

        if self.n_tasks == 1:
            tasks = torch.zeros(1, dtype=torch.long, device=self.device)
            return torch.zeros(1, device=self.device), torch.column_stack((tasks, tasks))

        scores = self.scores.clone()
        scores.fill_diagonal_(float("-inf"))
        similarity, nearest = scores.max(dim=1)
        tasks = torch.arange(self.n_tasks, device=self.device)
        pairs = torch.column_stack((tasks, nearest))
        return similarity, pairs
