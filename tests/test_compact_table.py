import numpy as np
import pytest
import torch

from examples.reproduce_compact_table import prepare_panel, parse_args, main


@pytest.mark.parametrize("dataset", ["multisine", "diabetes"])
def test_compact_data_is_fixed_disjoint_and_train_scaled(dataset):
    panel = prepare_panel(dataset, smoke=True)
    repeat = prepare_panel(dataset, smoke=True)
    for key, (x, y) in panel.splits.items():
        torch.testing.assert_close(x, repeat.splits[key][0])
        torch.testing.assert_close(y, repeat.splits[key][1])
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
    for ids in zip(*(panel.metadata["row_ids"][key] for key in ("train", "validation", "test"))):
        assert not set(ids[0]) & set(ids[1])
        assert not set(ids[0]) & set(ids[2])
        assert not set(ids[1]) & set(ids[2])
    x, y = panel.splits["train"]
    torch.testing.assert_close(x.mean(1), torch.zeros_like(x.mean(1)), atol=1e-6, rtol=0)
    torch.testing.assert_close(y.mean(1), torch.zeros_like(y.mean(1)), atol=1e-6, rtol=0)


def test_compact_rejects_no_regularized_epochs_and_duplicate_seeds():
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "5", "--update-every", "5"])
    with pytest.raises(SystemExit):
        parse_args(["--seeds", "11", "11"])


def test_compact_smoke_exports_recomputable_results(tmp_path):
    import json
    import pandas as pd

    main(["--smoke", "--out-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["arguments"]["smoke"] is True
    rows = json.loads((tmp_path / "runs.json").read_text())
    assert len(rows) == 8
    for row in rows:
        path = tmp_path / row["dataset"] / row["method"] / f"seed_{row['seed']}" / "predictions.npz"
        with np.load(path) as values:
            expected = np.sqrt(np.mean((values["prediction"] - values["target"]) ** 2, axis=(1, 2)))
        np.testing.assert_allclose(row["test_rmse"], expected.mean(), rtol=1e-6)
    table = pd.read_csv(tmp_path / "table.csv")
    assert (table.seeds == 1).all()
    assert table.rmse_std.isna().all()  # No invented variability from one seed.
    assert "SMOKE TEST ONLY" in (tmp_path / "table.tex").read_text()
    original = (tmp_path / "table.tex").read_text()
    (tmp_path / "table.tex").unlink()
    main(["--summarize-only", "--out-dir", str(tmp_path)])
    assert (tmp_path / "table.tex").read_text() == original
    with pytest.raises(SystemExit, match="not empty"):
        main(["--smoke", "--out-dir", str(tmp_path)])
