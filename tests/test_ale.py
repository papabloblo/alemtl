import pytest
import torch
from torch import nn

from alemtl.models.multitask_model import MultiTaskModel
from alemtl.similarity.ale import Intervals, MultiTaskALE


@pytest.mark.parametrize("shared_input_data", [False, True])
def test_recompute_uses_current_model_and_rebuilds_latent_intervals(shared_input_data):
    model = MultiTaskModel(
        n_tasks=2,
        modules_layout={
            "encoder": {"shared": "hard", "module": lambda: nn.Linear(1, 1, bias=False)},
            "head": {"shared": "soft", "module": lambda: nn.Linear(1, 1, bias=False)},
        },
        similarity_layers={"in": "head", "out": "head"},
        shared_input_data=shared_input_data,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    X = torch.arange(8.0).view(1, 8, 1).expand(2, -1, -1)
    loader = [(X, torch.zeros_like(X))]
    kwargs = dict(model=model, dataloader=loader, n_tasks=2,
                  shared_input_data=shared_input_data, num_intervals=4, n_guess=8)
    ale = MultiTaskALE(**kwargs)
    ale.update()
    old_curves = ale(std=None).clone()
    counts = ale.cardinality.clone()

    with torch.no_grad():
        model.model_by_task(0)["model"].get_submodule("encoder").weight.fill_(2.0)
        for task in range(2):
            model.model_by_task(task)["model"].get_submodule("head").weight.fill_(3.0)

    ale.recompute()
    curves = ale(std=None)
    fresh = MultiTaskALE(**kwargs)
    fresh.update()

    torch.testing.assert_close(curves, fresh(std=None))
    torch.testing.assert_close(ale.cardinality, counts)
    assert not torch.allclose(curves[..., 0], old_curves[..., 0])
    # For a linear head, centered ALE is slope * centered interval endpoints.
    grid = curves[..., 0]
    torch.testing.assert_close(curves[..., 1], 3.0 * (grid - grid.mean(dim=-1, keepdim=True)))

    ale.recompute()
    torch.testing.assert_close(ale(std=None), curves)
    torch.testing.assert_close(ale.cardinality, counts)
    ale.update()  # Explicit incremental accumulation remains available.
    torch.testing.assert_close(ale.cardinality, 2 * counts)


def test_recompute_preserves_model_modes_and_validates_limit():
    model = _model()
    model.train()
    model.model_by_task(0)["model"].get_submodule("head").eval()
    modes = {module: module.training for module in model.modules()}
    loader = [(torch.randn(2, 4, 3), torch.zeros(2, 4, 1)) for _ in range(2)]
    ale = MultiTaskALE(model, loader, n_tasks=2, num_intervals=3, n_guess=4)
    observed_modes = []
    handle = model.model_similarity_input.register_forward_pre_hook(
        lambda module, inputs: observed_modes.append(module.training)
    )
    try:
        ale.recompute(max_batches=1)
    finally:
        handle.remove()
    assert observed_modes and not any(observed_modes)
    assert all(module.training == mode for module, mode in modes.items())
    assert ale.cardinality.sum().item() == 2 * 4 * 3
    counts = ale.cardinality.clone()
    with pytest.raises(ValueError, match="max_batches"):
        ale.recompute(max_batches=-1)
    torch.testing.assert_close(ale.cardinality, counts)

    ale.dataloader = []
    with pytest.raises(RuntimeError, match="no samples"):
        ale.recompute()
    assert all(module.training == mode for module, mode in modes.items())


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
