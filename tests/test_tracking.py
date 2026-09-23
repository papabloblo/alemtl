import pytest
import torch

from alemtl.training.tracking import TrackALE_Similarity, TrackMetrics, Tracker


def test_track_metrics_keeps_epoch_numbers_and_means():
    metrics = TrackMetrics(errors_names=["mae"], keep_epochs=2)

    for epoch in range(3):
        metrics.new_epoch(epoch)
        metrics.update(
            loss_per_task=torch.tensor([float(epoch + 1), float(epoch + 3)]),
            errors_per_task={"mae": torch.tensor([float(epoch)])},
            loss_penalty=torch.tensor([1.0]),
            l2_penalty=0.1,
        )

    assert metrics.epoch_numbers == [1, 2]
    torch.testing.assert_close(metrics.info(2)["LOSS"], torch.tensor(4.0))
    with pytest.raises(IndexError):
        metrics.info(0)

    saved = metrics.for_save()
    torch.testing.assert_close(saved["epochs"], torch.tensor([1, 2]))
    assert saved["loss_per_task"].shape == (2, 2)


def test_track_ale_similarity_prunes_old_epochs():
    tracker = TrackALE_Similarity(keep_epochs=2)

    tracker.update(0, torch.tensor([0.0]))
    tracker.update(1, torch.tensor([1.0]))
    tracker.update(2, torch.tensor([2.0]))

    assert sorted(tracker.track) == [1, 2]
    assert sorted(tracker.for_save()) == [1, 2]


def test_tracker_best_model_is_stable_after_metric_pruning():
    tracker = Tracker(errors_names=["mae"], keep_epochs=1)

    tracker.start_epoch()
    tracker.start_train()
    tracker.update_metrics("train", torch.tensor([2.0]), {"mae": torch.tensor([2.0])}, torch.tensor([0.0]), 0.1)
    tracker.end_train()
    tracker.start_validation()
    tracker.update_metrics("validation", torch.tensor([1.0]), {"mae": torch.tensor([1.0])}, torch.tensor([0.0]), 0.1)
    tracker.end_validation()
    tracker.update_best_model({"w": torch.tensor([1.0])}, epoch=0, best_loss_value=1.0)

    tracker.start_epoch()
    tracker.start_train()
    tracker.update_metrics("train", torch.tensor([4.0]), {"mae": torch.tensor([4.0])}, torch.tensor([0.0]), 0.1)
    tracker.end_train()

    best = tracker.best_model()

    assert best["epoch"] == 0
    torch.testing.assert_close(best["train_metrics"]["LOSS"], torch.tensor(2.0))
    torch.testing.assert_close(best["val_metrics"]["LOSS"], torch.tensor(1.0))


def test_tracker_rejects_non_metric_phase_and_saves(tmp_path):
    tracker = Tracker(errors_names=[], path=str(tmp_path), config_info={"seed": 1})
    tracker.start_epoch()
    tracker.start_train()
    tracker.update_metrics("train", torch.tensor([1.0]), {}, torch.tensor([0.0]), 0.1)
    tracker.end_train()
    tracker.start_validation()
    tracker.update_metrics("validation", torch.tensor([1.0]), {}, torch.tensor([0.0]), 0.1)
    tracker.end_validation()
    tracker.update_best_model({"w": torch.tensor([1.0])}, epoch=0, best_loss_value=1.0)

    with pytest.raises(KeyError):
        tracker.update_metrics("ale", torch.tensor([1.0]), {}, torch.tensor([0.0]))

    tracker.save()

    assert (tmp_path / "times.pth").exists()
    assert (tmp_path / "metrics.pth").exists()
    assert (tmp_path / "best_model.pth").exists()
    assert (tmp_path / "config.pth").exists()


def test_weighted_metrics_pruning_and_rmse_save():
    metrics = TrackMetrics(["rmse", "mae"], keep_epochs=1,
                           loss_aggregation="rmse", errors_aggregation={"rmse": "rmse"})
    for epoch in range(2):
        metrics.new_epoch(epoch)
        for value, size in ((1., 4), (9., 1)):
            metrics.update(torch.tensor([value ** 2]),
                           {"rmse": torch.tensor([value ** 2]), "mae": torch.tensor([value])},
                           torch.zeros(1), batch_size=size)
    assert metrics.sample_counts == [5]
    assert metrics.batch_counts == [2]
    assert metrics.epoch_numbers == [1]
    torch.testing.assert_close(metrics.info(1)["rmse"], torch.tensor(17.).sqrt())
    torch.testing.assert_close(metrics.info(1)["mae"], torch.tensor(2.6))
    torch.testing.assert_close(metrics.for_save()["loss_per_task"], torch.tensor([[17.]]).sqrt())
    with pytest.raises(ValueError, match="batch_size"):
        metrics.update(torch.ones(1), {"rmse": torch.ones(1), "mae": torch.ones(1)}, torch.zeros(1), batch_size=0)
    assert metrics.sample_counts == [5]
