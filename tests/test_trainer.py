import pytest
from alemtl.training.loss import rmse_loss, mae_loss, mape_loss
import torch
from torch import nn

from alemtl.models.multitask_model import MultiTaskModel
from alemtl.training.loss import MultiTaskLoss
from alemtl.training.trainer import MultiTaskTrainer


def _model() -> MultiTaskModel:
    return MultiTaskModel(
        n_tasks=2,
        modules_layout={
            "head": {
                "shared": "soft",
                "module": lambda: nn.Linear(2, 1),
            },
        },
        similarity_layers={"in": None, "out": "head"},
    )


def _loader():
    X = torch.randn(2, 4, 2)
    y = torch.randn(2, 4, 1)
    return [(X, y), (X + 0.1, y)]


def _trainer(**kwargs) -> MultiTaskTrainer:
    model = _model()
    trainer_kwargs = {
        "print_each_epochs": 100,
        "logging_dir": "",
    }
    trainer_kwargs.update(kwargs)
    return MultiTaskTrainer(
        model=model,
        train_dataloader=_loader(),
        validation_dataloader=_loader(),
        test_dataloader=_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss=MultiTaskLoss(model=model, loss_fn=nn.MSELoss(reduction="none")),
        **trainer_kwargs,
    )


def test_trainer_runs_one_epoch_without_logging_dir(capsys):
    trainer = _trainer()

    trainer.train(epochs=1, max_batches=1)

    assert trainer.tracking.best_epoch == 0
    assert trainer.tracking.best_model_parameters is not None
    assert len(trainer.tracking.track["train"]["metrics"].batch_counts) == 1
    assert len(trainer.tracking.track["test"]["metrics"].batch_counts) == 1
    capsys.readouterr()


def test_trainer_metric_table_uses_aligned_borders(capsys):
    trainer = _trainer(print_each_epochs=1)

    trainer.train(epochs=1, max_batches=1)

    output = capsys.readouterr().out
    lines = output.splitlines()
    table_start = next(index for index, line in enumerate(lines) if "METRICS" in line) - 1
    table_end = next(
        index for index in range(table_start, len(lines))
        if lines[index].startswith("+-----------------+")
    )
    table_lines = [line for line in lines[table_start:table_end] if line]

    assert len({len(line) for line in table_lines}) == 1
    assert all(line.startswith(("+", "|")) for line in table_lines)


class DummySimilarity:
    def __init__(self):
        self.similarity_tasks_features = torch.ones(2, 2, 1)
        self.compute_calls = 0

    def compute(self):
        self.compute_calls += 1

    def tasks_groups(self):
        return torch.ones(2), torch.tensor([[0, 1], [1, 0]])


def test_similarity_step_updates_and_clears_task_groups():
    similarity = DummySimilarity()
    trainer = _trainer(
        multitask_similarity=similarity,
        similarity_each_epochs=2,
        keep_similarity_epochs=1,
    )

    trainer.similarity_step(epoch=1)
    assert similarity.compute_calls == 1
    assert trainer.loss.tasks_groups is not None

    trainer.similarity_step(epoch=2)
    assert trainer.loss.tasks_groups is None


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("loss_fn", [rmse_loss, nn.MSELoss(reduction="none")])
def test_evaluation_metrics_are_independent_of_batch_partition(batch_size, loss_fn, tmp_path):
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    X = torch.zeros(2, 5, 2)
    y = torch.tensor([[[1.], [1.], [1.], [1.], [9.]],
                      [[2.], [2.], [2.], [2.], [4.]]])
    loader = [(X[:, i:i + batch_size], y[:, i:i + batch_size])
              for i in range(0, 5, batch_size)]
    loss = MultiTaskLoss(model, loss_fn, errors_fn={"arbitrary_name": rmse_loss, "mae": mae_loss, "mape": mape_loss})
    trainer = MultiTaskTrainer(
        model, loader, loader, loader, torch.optim.SGD(model.parameters(), lr=0.0), loss,
        logging_dir=str(tmp_path), print_each_epochs=100,
    )
    trainer.train(epochs=1)
    expected_rmse = y.square().mean(dim=(1, 2)).sqrt()
    expected_loss = expected_rmse if loss_fn is rmse_loss else y.square().mean(dim=(1, 2))
    best = trainer.get_best_model()
    for phase in ("train_metrics", "val_metrics", "test_metrics"):
        torch.testing.assert_close(best[phase]["arbitrary_name"], expected_rmse.mean())
        torch.testing.assert_close(best[phase]["mae"], y.mean())
        torch.testing.assert_close(best[phase]["mape"], torch.tensor(1.))
        torch.testing.assert_close(best[phase]["LOSS"], expected_loss.mean())
    torch.testing.assert_close(trainer.tracking.last_val_loss(), expected_loss.mean())
    saved = torch.load(tmp_path / "metrics.pth", weights_only=True)
    torch.testing.assert_close(saved["test"]["loss_per_task"][0], expected_loss)
    torch.testing.assert_close(saved["test"]["errors"]["arbitrary_name"][0], expected_rmse)
