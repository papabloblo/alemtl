import torch
import pytest

from alemtl.similarity.similarity import (
    MultitaskSimilarity,
    discrete_frechet_distance_vectorized,
    frechet_distance_vectorized,
)


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
    similarity = frechet_distance_vectorized(curve, curve)
    shifted_similarity = frechet_distance_vectorized(curve, shifted)

    assert distance.item() == 0.0
    assert similarity.item() == 1.0
    assert shifted_similarity.item() < similarity.item()


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
