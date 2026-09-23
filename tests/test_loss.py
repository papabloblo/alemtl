import torch

from alemtl.training.loss import (
    MultiTaskLoss,
    accuracy,
    confusion_matrix,
    f1_score,
    mae_loss,
    mape_loss,
    rmse_loss,
)


class DummyModel:
    n_tasks = 2
    device = torch.device("cpu")

    def get_param_groups(self, groups):
        vectors = {
            0: torch.tensor([1.0, 2.0]),
            1: torch.tensor([2.0, 4.0]),
        }
        return [torch.stack([vectors[task] for task in group]) for group in groups]


def test_multitask_loss_reduces_all_non_task_dimensions():
    loss = MultiTaskLoss(
        model=DummyModel(),
        loss_fn=lambda real, pred: (pred - real).abs(),
        errors_fn={"mae": lambda real, pred: (pred - real).abs()},
    )
    real = torch.zeros(2, 3, 2)
    pred = torch.tensor(
        [
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],
            [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]],
        ]
    )

    per_task = loss.loss_per_task(real, pred)
    metrics = loss.errors_per_task(real, pred)

    torch.testing.assert_close(per_task, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(metrics["mae"], torch.tensor([2.0, 3.0]))


def test_multitask_loss_penalty_from_task_groups():
    loss = MultiTaskLoss(model=DummyModel(), loss_fn=lambda real, pred: (pred - real).abs())
    loss.update_tasks_groups(torch.tensor([[0, 1], [1, 0]]))

    penalty = loss.penalty()

    expected = torch.linalg.vector_norm(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(penalty, torch.tensor([expected, expected]))


def test_binary_metrics_per_output_column():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    torch.testing.assert_close(accuracy(logits, target), torch.tensor([1.0, 1.0]))
    counts = confusion_matrix(logits, target)
    torch.testing.assert_close(counts["true_positives"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(counts["true_negatives"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(counts["false_positives"], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(counts["false_negatives"], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(f1_score(logits, target), torch.tensor([1.0, 1.0]))


def test_regression_losses_reduce_batch_axis():
    real = torch.tensor([[[1.0], [2.0]], [[2.0], [4.0]]])
    pred = torch.tensor([[[2.0], [4.0]], [[1.0], [1.0]]])

    torch.testing.assert_close(mae_loss(real, pred), torch.tensor([[1.5], [2.0]]))
    torch.testing.assert_close(rmse_loss(real, pred), torch.tensor([[1.5811], [2.2361]]), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mape_loss(real, pred), torch.tensor([[1.0], [0.625]]))


def test_epoch_rmse_aggregates_all_outputs_before_taking_root():
    from alemtl.training.tracking import TrackMetrics

    loss = MultiTaskLoss(DummyModel(), rmse_loss, errors_fn={"score": rmse_loss})
    real = torch.zeros(2, 3, 2)
    pred = torch.tensor([[[0., 10.], [0., 10.], [3., 4.]],
                         [[1., 2.], [3., 4.], [5., 6.]]])
    metrics = TrackMetrics(["score"], loss_aggregation=loss.loss_aggregation,
                           errors_aggregation=loss.errors_aggregation)
    metrics.new_epoch()
    for start, end in ((0, 2), (2, 3)):
        y, p = real[:, start:end], pred[:, start:end]
        values, errors = loss.epoch_values(y, p, loss.loss_per_task(y, p), loss.errors_per_task(y, p))
        metrics.update(values, errors, torch.zeros(2), batch_size=end - start)
    expected = pred.square().mean(dim=(1, 2)).sqrt()
    torch.testing.assert_close(metrics.for_save()["errors"]["score"][0], expected)
    torch.testing.assert_close(metrics.info(0)["LOSS"], expected.mean())
