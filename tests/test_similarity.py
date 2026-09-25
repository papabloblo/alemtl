import math

import pytest
import torch

from alemtl.similarity.similarity import (
    MultitaskSimilarity,
    discrete_frechet_distance_vectorized,
    frechet_distance_vectorized,
    frechet_similarity_vectorized,
)


def _reference_discrete_frechet(curve0, curve1):
    """Scalar row-by-row reference, independent of the vectorized recurrence."""
    distances = [[0.0 for _ in curve1] for _ in curve0]
    for i, point0 in enumerate(curve0):
        for j, point1 in enumerate(curve1):
            distance = math.dist(point0, point1)
            if i == 0 and j == 0:
                distances[i][j] = distance
            elif i == 0:
                distances[i][j] = max(distance, distances[i][j - 1])
            elif j == 0:
                distances[i][j] = max(distance, distances[i - 1][j])
            else:
                distances[i][j] = max(
                    distance,
                    min(distances[i - 1][j], distances[i][j - 1], distances[i - 1][j - 1]),
                )
    return distances[-1][-1]


@pytest.mark.parametrize("swap_curves", [False, True])
def test_discrete_frechet_includes_starting_distance(swap_curves):
    curve0 = torch.tensor([[0.0, 0.0], [1.0, 10.0]])
    curve1 = torch.tensor([[0.0, 10.0], [1.0, 10.0]])
    if swap_curves:
        curve0, curve1 = curve1, curve0

    expected = torch.tensor(10.0)
    torch.testing.assert_close(discrete_frechet_distance_vectorized(curve0, curve1), expected)
    torch.testing.assert_close(frechet_similarity_vectorized(curve0, curve1), torch.exp(-expected))
    with pytest.warns(DeprecationWarning, match="frechet_distance_vectorized is deprecated"):
        torch.testing.assert_close(frechet_distance_vectorized(curve0, curve1), torch.exp(-expected))


@pytest.mark.parametrize("n_points", [1, 2, 3, 7])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_discrete_frechet_matches_reference_and_endpoint_bounds(n_points, batched, dtype):
    generator = torch.Generator().manual_seed(42)
    curves0 = torch.randn(5, n_points, 2, generator=generator, dtype=dtype)
    curves1 = torch.randn(5, n_points, 2, generator=generator, dtype=dtype)
    expected = torch.tensor(
        [_reference_discrete_frechet(a, b) for a, b in zip(curves0.tolist(), curves1.tolist())],
        dtype=dtype,
    )
    if not batched:
        curves0, curves1, expected = curves0[0], curves1[0], expected[0]

    actual = discrete_frechet_distance_vectorized(curves0, curves1)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(discrete_frechet_distance_vectorized(curves1, curves0), actual)
    endpoint_distances = torch.linalg.vector_norm(curves0[..., [0, -1], :] - curves1[..., [0, -1], :], dim=-1)
    tolerance = 10 * torch.finfo(dtype).eps
    assert torch.all(actual.unsqueeze(-1) + tolerance >= endpoint_distances)


class DummyALE:
    def __init__(self, curves: torch.Tensor) -> None:
        self.curves = curves
        self.device = curves.device
        self.n_tasks = curves.size(0)
        self.n_features_in = curves.size(1)
        self.num_intervals = curves.size(2)
        self.n_features_out = curves.size(3) - 1

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.curves


def test_discrete_frechet_distance_and_compat_similarity():
    curve = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    shifted = torch.tensor([[0.0, 1.0], [1.0, 2.0]])

    distance = discrete_frechet_distance_vectorized(curve, curve)
    similarity = frechet_similarity_vectorized(curve, curve)
    shifted_similarity = frechet_similarity_vectorized(curve, shifted)

    assert distance.item() == 0.0
    assert similarity.item() == 1.0
    assert shifted_similarity.item() < similarity.item()

    with pytest.warns(DeprecationWarning, match="frechet_distance_vectorized is deprecated"):
        compatibility_similarity = frechet_distance_vectorized(curve, curve)
    torch.testing.assert_close(compatibility_similarity, similarity)


def test_multitask_similarity_uses_explicit_similarity_default():
    x = torch.linspace(0, 1, steps=4)
    curves = torch.zeros(2, 1, 4, 2)
    curves[:, :, :, 0] = x
    curves[0, :, :, 1] = x
    curves[1, :, :, 1] = x + 0.1

    similarity = MultitaskSimilarity(DummyALE(curves))

    assert similarity.similarity_func is frechet_similarity_vectorized


def test_multitask_similarity_handles_multioutput_curves():
    x = torch.linspace(0, 1, steps=4)
    curves = torch.zeros(3, 2, 4, 3)
    curves[:, :, :, 0] = x

    curves[0, :, :, 1] = x
    curves[1, :, :, 1] = x + 0.1
    curves[2, :, :, 1] = x + 3.0

    curves[0, :, :, 2] = x.square()
    curves[1, :, :, 2] = x.square() + 0.1
    curves[2, :, :, 2] = x.square() + 3.0

    similarity = MultitaskSimilarity(DummyALE(curves), output_reduction="mean")
    similarity.compute()
    scores, groups = similarity.tasks_groups()

    assert similarity.similarity_tasks_features.shape == (3, 3, 2)
    assert similarity.scores[0, 1] > similarity.scores[0, 2]
    assert similarity.matrix is similarity.scores
    assert similarity.similarity_matrix is similarity.scores
    assert groups[0, 1].item() == 1
    assert scores.shape == (3,)

    with pytest.warns(DeprecationWarning, match="distances is deprecated"):
        torch.testing.assert_close(similarity.distances, similarity.scores)
