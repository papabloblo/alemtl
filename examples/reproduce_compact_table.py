"""Offline, bounded SoftwareX illustration; run with --help for options."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import load_diabetes

from alemtl.models import MultiTaskModel
from alemtl.similarity import MultiTaskALE, MultitaskSimilarity
from alemtl.training import MultiTaskLoss, MultiTaskTrainer, rmse_loss

METHODS = ("single", "hard", "soft", "ale_frechet")
LABELS = {"single": "Single-task", "hard": "Hard sharing", "soft": "Soft sharing", "ale_frechet": "ALE--Frechet"}
DATA_SEED = 1729


@dataclass
class Panel:
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]]
    y_mean: torch.Tensor
    y_scale: torch.Tensor
    metadata: dict


def prepare_panel(name: str, smoke: bool = False) -> Panel:
    """Fixed splits across methods/seeds; train-only scaling and equal task sizes."""
    rng = np.random.default_rng(DATA_SEED)
    if name == "multisine":
        n = 96 if smoke else 512
        groups = []
        for task in range(5):
            x = rng.uniform(0, 2 * np.pi, n)
            features = np.column_stack((x, np.sin(x), np.cos(x)))
            y = (0.5 + task / 4) * np.sin((1 + task / 4) * x + task * np.pi / 8)
            groups.append((features, y + rng.normal(0, 0.05, n), np.arange(n)))
        source = "Synthetic Multi-Sine interpolation; not the original forecasting benchmark."
        labels = [f"sine_{i}" for i in range(5)]
    elif name == "diabetes":
        data = load_diabetes(scaled=False)
        age, sex = data.data[:, 0], data.data[:, 1]
        groups, labels = [], []
        for value in sorted(np.unique(sex)):
            for older in (False, True):
                ids = np.flatnonzero((sex == value) & ((age >= 50) == older))
                groups.append((np.delete(data.data[ids], 1, axis=1), data.target[ids], ids))
                labels.append(f"sex_{value:g}_age_{'ge50' if older else 'lt50'}")
        source = "sklearn.datasets.load_diabetes(scaled=False); four predefined sex/age cohorts; sex excluded from predictors."
    else:
        raise ValueError(name)
    parts = {key: [] for key in ("train", "validation", "test")}
    ids_by_split = {key: [] for key in parts}
    for x, y, ids in groups:
        order = rng.permutation(len(y))
        cuts = (int(0.6 * len(y)), int(0.8 * len(y)))
        for key, indices in zip(parts, np.split(order, cuts)):
            parts[key].append((x[indices], y[indices, None]))
            ids_by_split[key].append(ids[indices])
    # Equal sizes avoid implicit oversampling of smaller evaluation cohorts.
    sizes = {key: min(len(y) for _, y in values) for key, values in parts.items()}
    arrays = {key: (np.stack([x[:sizes[key]] for x, _ in values]),
                    np.stack([y[:sizes[key]] for _, y in values])) for key, values in parts.items()}
    train_x, train_y = arrays["train"]
    x_mean, x_scale = train_x.mean(1, keepdims=True), train_x.std(1, keepdims=True)
    y_mean, y_scale = train_y.mean(1, keepdims=True), train_y.std(1, keepdims=True)
    x_scale = np.maximum(x_scale, 1e-8)
    y_scale = np.maximum(y_scale, 1e-8)
    splits = {key: (torch.tensor((x-x_mean)/x_scale, dtype=torch.float32),
                    torch.tensor((y-y_mean)/y_scale, dtype=torch.float32)) for key, (x, y) in arrays.items()}
    metadata = {"source": source, "data_seed": DATA_SEED, "tasks": labels,
                "rows_per_task": sizes,
                "row_ids": {key: [ids[:sizes[key]].tolist() for ids in values] for key, values in ids_by_split.items()}}
    return Panel(splits, torch.tensor(y_mean, dtype=torch.float32), torch.tensor(y_scale, dtype=torch.float32), metadata)


class Batches:
    def __init__(self, tensors, size, seed, shuffle=False):
        self.x, self.y = tensors
        self.size, self.shuffle = size, shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (self.x.size(1) + self.size - 1) // self.size

    def __iter__(self):
        order = torch.randperm(self.x.size(1), generator=self.generator) if self.shuffle else torch.arange(self.x.size(1))
        for start in range(0, len(order), self.size):
            ids = order[start:start+self.size]
            yield self.x[:, ids], self.y[:, ids]


def build_model(method, tasks, features, width):
    return MultiTaskModel(tasks, {
        "encoder": {"shared": "hard" if method == "hard" else "soft",
                    "module": lambda: nn.Sequential(nn.Linear(features, width), nn.Tanh())},
        "head": {"shared": "soft", "module": lambda: nn.Linear(width, 1)},
    }, similarity_layers={"in": "encoder", "out": "head"}, same_parameters=True)


def run_one(panel, method, seed, args, directory):
    torch.manual_seed(seed)
    tasks, _, features = panel.splits["train"][0].shape
    model = build_model(method, tasks, features, args.width)
    loaders = {key: Batches(tensors, args.batch_size, seed, key == "train") for key, tensors in panel.splits.items()}
    ale = similarity = None
    coefficient = 0.0
    if method == "ale_frechet":
        ale = MultiTaskALE(model, Batches(panel.splits["train"], args.batch_size, seed),
                           n_tasks=tasks, num_intervals=args.intervals,
                           n_guess=panel.splits["train"][0].size(1))
        similarity = MultitaskSimilarity(ale)
        coefficient = args.regularization / (tasks * features)
    pairs = [[i, j] for i in range(tasks) for j in range(i + 1, tasks)]
    if method == "soft":
        coefficient = args.regularization / len(pairs)
    loss = MultiTaskLoss(model, nn.MSELoss(reduction="none"),
                         errors_fn={"rmse": rmse_loss}, l2_penalty=coefficient)
    if method == "soft":
        loss.update_tasks_groups(pairs)
    trainer = MultiTaskTrainer(
        model, loaders["train"], loaders["validation"], loaders["test"],
        torch.optim.Adam(model.parameters(), lr=args.learning_rate), loss,
        ale=ale, multitask_similarity=similarity,
        ale_each_epochs=args.update_every if ale else None,
        similarity_each_epochs=args.update_every if ale else None,
        keep_similarity_epochs=args.keep_similarity, early_stopping_epochs=args.epochs + 1,
        print_each_epochs=args.epochs, print_limit_epochs=1, logging_dir="",
    )
    directory.mkdir(parents=True)
    started = time.perf_counter()
    with (directory / "training.log").open("w") as stream, redirect_stdout(stream):
        trainer.train(epochs=args.epochs)
    seconds = time.perf_counter() - started
    # The trainer restores the validation-selected checkpoint before testing.
    with torch.no_grad():
        x, y = panel.splits["test"]
        prediction = model(x) * panel.y_scale + panel.y_mean
        target = y * panel.y_scale + panel.y_mean
        per_task = (prediction-target).square().mean(dim=(1, 2)).sqrt()
    if not torch.isfinite(per_task).all():
        raise RuntimeError(f"Non-finite RMSE for {method}, seed {seed}")
    np.savez_compressed(directory / "predictions.npz", prediction=prediction.numpy(), target=target.numpy())
    torch.save(model.state_dict(), directory / "checkpoint.pt")
    return {"method": method, "seed": seed, "test_rmse": per_task.mean().item(),
            "per_task_rmse": per_task.tolist(), "train_seconds": seconds,
            "best_epoch": trainer.tracking.best_epoch,
            "parameters": sum(p.numel() for p in model.parameters())}


def summarize(rows):
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["dataset", "method"], sort=False).agg(
        seeds=("seed", "nunique"), rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", "std"), seconds_mean=("train_seconds", "mean"),
        parameters=("parameters", "first"),
    ).reset_index()
    return summary


def write_table(summary, out, smoke):
    lines = [r"\begin{tabular}{llrrr}", r"\hline", r"Dataset & Method & Test RMSE & Seconds/run & Seeds \\", r"\hline"]
    for row in summary.itertuples():
        score = f"{row.rmse_mean:.4f}" if row.seeds == 1 else f"{row.rmse_mean:.4f} $\\pm$ {row.rmse_std:.4f}"
        lines.append(f"{row.dataset.title()} & {LABELS[row.method]} & {score} & {row.seconds_mean:.2f} & {row.seeds} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    if smoke:
        lines.insert(0, "% SMOKE TEST ONLY: not publication results.")
    (out / "table.tex").write_text("\n".join(lines) + "\n")
    summary.to_csv(out / "table.csv", index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/compact_table"))
    parser.add_argument("--datasets", nargs="+", choices=["multisine", "diabetes"], default=["multisine", "diabetes"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--intervals", type=int, default=30)
    parser.add_argument("--update-every", type=int, default=10)
    parser.add_argument("--keep-similarity", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--regularization", type=float, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true", help="Audit saved predictions and rebuild tables without training.")
    parser.add_argument("--smoke", action="store_true", help="Two epochs, one seed, small synthetic data; not publication results.")
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs, args.seeds, args.update_every = 2, args.seeds[:1], 1
    for name in ("epochs", "width", "batch_size", "intervals", "update_every", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.epochs <= args.update_every and "ale_frechet" in args.methods:
        parser.error("epochs must exceed update-every so ALE regularization affects training")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.regularization) or args.regularization < 0:
        parser.error("learning rate must be finite and positive; regularization finite and nonnegative")
    for name in ("seeds", "methods", "datasets"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"duplicate {name} are not allowed")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out_dir.resolve()
    if args.summarize_only:
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise SystemExit("Cannot summarize an incomplete experiment.")
        rows = json.loads((out / "runs.json").read_text())
        for row in rows:
            archive = out / row["dataset"] / row["method"] / f"seed_{row['seed']}" / "predictions.npz"
            with np.load(archive) as values:
                rmse = np.sqrt(np.mean((values["prediction"] - values["target"]) ** 2, axis=(1, 2)))
            np.testing.assert_allclose(row["per_task_rmse"], rmse, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(row["test_rmse"], rmse.mean(), rtol=1e-5, atol=1e-6)
        write_table(summarize(rows), out, manifest["arguments"]["smoke"])
        print(f"Audited {len(rows)} prediction archives and rebuilt tables: {out}")
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}. Choose a new --out-dir.")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(root), *arguments], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    manifest = {"status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "device": "cpu", "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "scipy", "scikit-learn")},
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__).resolve(), *sorted((root / "src" / "alemtl").rglob("*.py"))]},
                "datasets": {}}
    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    save_manifest()
    rows = []
    start = time.perf_counter()
    try:
        for dataset in args.datasets:
            panel = prepare_panel(dataset, args.smoke)
            archive = out / f"{dataset}_data.npz"
            np.savez_compressed(archive, **{f"{key}_{axis}": value.numpy()
                                 for key, values in panel.splits.items() for axis, value in zip(("x", "y"), values)},
                                y_mean=panel.y_mean.numpy(), y_scale=panel.y_scale.numpy())
            manifest["datasets"][dataset] = {**panel.metadata, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            save_manifest()
            for method in args.methods:
                for seed in args.seeds:
                    result = run_one(panel, method, seed, args, out / dataset / method / f"seed_{seed}")
                    rows.append({"dataset": dataset, **result})
                    (out / "runs.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
                    print(f"{dataset} {method} seed={seed}: RMSE={result['test_rmse']:.4f}, {result['train_seconds']:.2f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        write_table(summarize(rows), out, args.smoke)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["total_seconds"] = time.perf_counter() - start
        save_manifest()
    print(f"Table and audit artifacts: {out}")


if __name__ == "__main__":
    main()
