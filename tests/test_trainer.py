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
    return MultiTaskTrainer(
        model=model,
        train_dataloader=_loader(),
        validation_dataloader=_loader(),
        test_dataloader=_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss=MultiTaskLoss(model=model, loss_fn=nn.MSELoss(reduction="none")),
        print_each_epochs=100,
        logging_dir="",
        **kwargs,
    )


def test_trainer_runs_one_epoch_without_logging_dir(capsys):
    trainer = _trainer()

    trainer.train(epochs=1, max_batches=1)

    assert trainer.tracking.best_epoch == 0
    assert trainer.tracking.best_model_parameters is not None
    assert len(trainer.tracking.track["train"]["metrics"].batch_counts) == 1
    assert len(trainer.tracking.track["test"]["metrics"].batch_counts) == 1
    capsys.readouterr()


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
