import pytest
import torch
from torch import nn

from alemtl.models import CrossStitch, HardSharing, MMoE, MTAN, PLE, SingleTaskMLP, SoftSharing
from alemtl.training.loss import MultiTaskLoss
from alemtl.training.trainer import MultiTaskTrainer


N_TASKS = 3
INPUT_DIM = 4
OUTPUT_DIM = 2
BATCH = 5


def _baseline_cases():
    return [
        pytest.param(
            "hard_sharing",
            lambda: HardSharing(
                N_TASKS,
                input_dim=INPUT_DIM,
                trunk=(6,),
                head=(5,),
                output_dim=OUTPUT_DIM,
            ),
            id="HardSharing",
        ),
        pytest.param(
            "soft_sharing",
            lambda: SoftSharing(
                N_TASKS,
                input_dim=INPUT_DIM,
                hidden=(6, 5),
                output_dim=OUTPUT_DIM,
            ),
            id="SoftSharing",
        ),
        pytest.param(
            "single_task_mlp",
            lambda: SingleTaskMLP(
                N_TASKS,
                input_dim=INPUT_DIM,
                hidden=(6, 5),
                output_dim=OUTPUT_DIM,
            ),
            id="SingleTaskMLP",
        ),
        pytest.param(
            "mmoe",
            lambda: MMoE(
                N_TASKS,
                input_dim=INPUT_DIM,
                n_experts=2,
                expert_hidden=(6,),
                tower_hidden=(5,),
                output_dim=OUTPUT_DIM,
            ),
            id="MMoE",
        ),
        pytest.param(
            "cross_stitch",
            lambda: CrossStitch(
                N_TASKS,
                input_dim=INPUT_DIM,
                shared_dims=(6, 5),
                output_dim=OUTPUT_DIM,
            ),
            id="CrossStitch",
        ),
        pytest.param(
            "ple",
            lambda: PLE(
                N_TASKS,
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                n_layers=1,
                n_shared_experts=2,
                n_task_experts=1,
                expert_hidden=(6,),
                tower_hidden=(5,),
            ),
            id="PLE",
        ),
        pytest.param(
            "mtan",
            lambda: MTAN(
                N_TASKS,
                input_dim=INPUT_DIM,
                shared_dims=(6,),
                output_dim=OUTPUT_DIM,
                tower_hidden=(5,),
            ),
            id="MTAN",
        ),
    ]


@pytest.mark.parametrize(("name", "model_factory"), _baseline_cases())
def test_baseline_forward_shape_and_parameter_update(name, model_factory):
    torch.manual_seed(12)
    model = model_factory()
    X = torch.randn(N_TASKS, BATCH, INPUT_DIM)
    y = torch.randn(N_TASKS, BATCH, OUTPUT_DIM)

    y_pred = model(X)

    assert y_pred.shape == (N_TASKS, BATCH, OUTPUT_DIM)
    assert torch.isfinite(y_pred).all()

    before = {
        param_name: param.detach().clone()
        for param_name, param in model.named_parameters()
        if param.requires_grad
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.MSELoss()(model(X), y)
    loss.backward()
    optimizer.step()

    changed = [
        param_name
        for param_name, param in model.named_parameters()
        if param.requires_grad and not torch.allclose(param.detach(), before[param_name])
    ]
    assert changed, f"{name} did not update any trainable parameter"


@pytest.mark.parametrize(("name", "model_factory"), _baseline_cases())
def test_baseline_training_compatibility_with_multitask_trainer(name, model_factory, capsys):
    torch.manual_seed(24)
    model = model_factory()
    X = torch.randn(N_TASKS, BATCH, INPUT_DIM)
    y = torch.randn(N_TASKS, BATCH, OUTPUT_DIM)
    dataloader = [(X, y), (X + 0.1, y - 0.1)]

    trainer = MultiTaskTrainer(
        model=model,
        train_dataloader=dataloader,
        validation_dataloader=dataloader,
        test_dataloader=dataloader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss=MultiTaskLoss(model=model, loss_fn=nn.MSELoss(reduction="none")),
        print_each_epochs=100,
        logging_dir="",
        architecture=name,
    )

    trainer.train(epochs=1, max_batches=1)

    assert trainer.tracking.best_epoch == 0
    assert trainer.tracking.best_model_parameters is not None
    assert len(trainer.tracking.track["train"]["metrics"].batch_counts) == 1
    assert len(trainer.tracking.track["validation"]["metrics"].batch_counts) == 1
    assert len(trainer.tracking.track["test"]["metrics"].batch_counts) == 1
    capsys.readouterr()
