import torch
from torch import nn

from alemtl.models.multitask_model import MultiTaskModel
from alemtl.similarity.ale import Intervals, MultiTaskALE


def _model(shared_input_data: bool = False) -> MultiTaskModel:
    layout = {
        "encoder": {
            "shared": "hard",
            "module": lambda: nn.Sequential(nn.Linear(3, 5), nn.ReLU()),
        },
        "head": {
            "shared": "soft",
            "module": lambda: nn.Linear(5, 1),
        },
    }
    return MultiTaskModel(
        n_tasks=2,
        modules_layout=layout,
        similarity_layers={"in": "encoder", "out": "head"},
        shared_input_data=shared_input_data,
    )


def test_intervals_clamp_values_to_valid_bins():
    intervals = Intervals(
        torch.tensor([[0.0], [1.0], [2.0]]),
        num_intervals=2,
    )

    idx = intervals.compute_index_intervals(torch.tensor([[2.0], [3.0]]))

    assert idx.shape == (2, 1, 1)
    assert idx.max().item() == 1


def test_multitask_ale_returns_expected_curve_shape():
    torch.manual_seed(1)
    loader = [
        (torch.randn(2, 4, 3), torch.randn(2, 4, 1))
        for _ in range(3)
    ]
    ale = MultiTaskALE(
        model=_model(),
        dataloader=loader,
        n_tasks=2,
        num_intervals=4,
        n_guess=8,
    )

    ale.update(max_batches=2)
    curves = ale()

    assert curves.shape == (2, 3, 4, 2)


def test_shared_input_cardinality_is_counted_once_per_batch():
    X = torch.randn(2, 4, 3)
    X[1] = X[0]
    loader = [(X, torch.randn(2, 4, 1))]
    ale = MultiTaskALE(
        model=_model(shared_input_data=True),
        dataloader=loader,
        n_tasks=2,
        shared_input_data=True,
        num_intervals=3,
        n_guess=4,
    )

    ale.update(max_batches=1)

    assert ale.cardinality[0].sum().item() == 4 * 3
