#!/usr/bin/env python3
"""Reproduce the complete SoftwareX illustrative-example section with ALEMTL.

Depending on ``--section``, it performs one or both of
the following workflows:

``figures``
    Generate or load the input data, train the selected ALE--Frechet model,
    compute ALE profiles and task similarities during training, and rebuild the
    Multi-Sine and Electricity interpretability figures.

``benchmark``
    Rerun the validation-selected configurations for ALE--Frechet, Soft,
    Cross-Stitch, Hard, Single-task MLP, PLE, and MTAN; aggregate newly computed
    test RMSE values; calculate average ranks; and export the manuscript table.
    MMoE remains available through ``--benchmark-methods mmoe`` although it is
    not printed in the current SoftwareX table.

The paper defaults are encoded directly from the original experiment configs.
Synthetic Multi-Sine and Polynomial data can be generated internally. The
Exchange, METR-LA, and NN5 download/preprocessing workflow is included in
``scripts/prepare_real_datasets.py`` and can be invoked here with
``--prepare-real-datasets``. Electricity can be downloaded and prepared in
this script with ``--prepare-electricity``.

Examples
--------
Rebuild the Multi-Sine interpretability result::

    python reproduce_softwarex_illustrative_examples.py \
        --section figures --example multisine

Rebuild both interpretability figures::

    python reproduce_softwarex_illustrative_examples.py \
        --section figures --example all \
        --electricity-data-dir data/interim/electricity

Download/preprocess the real data and rerun the complete table::

    python reproduce_softwarex_illustrative_examples.py \
        --section benchmark --data-root data/interim \
        --prepare-real-datasets --prepare-electricity

Run a small functional check that is not a paper reproduction::

    python reproduce_softwarex_illustrative_examples.py \
        --section benchmark --benchmark-datasets multisine \
        --benchmark-methods ale_frechet soft hard --smoke --max-batches 1
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# When this file is executed from an ALEMTL source checkout, prefer that
# checkout's ``src`` tree. When copied elsewhere, ALEMTL must be installed in
# the active environment (for example, ``pip install alemtl``).
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_SRC = _REPOSITORY_ROOT / "src"
if _REPOSITORY_ROOT.is_dir():
    sys.path.insert(0, str(_REPOSITORY_ROOT))
if _REPOSITORY_SRC.is_dir():
    sys.path.insert(0, str(_REPOSITORY_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch
from torch import nn

from alemtl.data import MultitaskDataloader, MultitaskDataset
from alemtl.models import (
    CrossStitch,
    HardSharing,
    MMoE,
    MTAN,
    MultiTaskModel,
    PLE,
    SingleTaskMLP,
    SoftSharing,
)
from alemtl.similarity import MultiTaskALE, MultitaskSimilarity
from alemtl.training import (
    MultiTaskLoss,
    MultiTaskTrainer,
    mae_loss,
    mape_loss,
    rmse_loss,
)

ExampleName = Literal["multisine", "electricity"]
DatasetName = Literal["electricity", "exchange", "metrla", "multisine", "nn5", "polynomial"]
MethodName = Literal[
    "ale_frechet", "soft", "crossstitch", "hard",
    "single_task_mlp", "ple", "mtan", "mmoe"
]

MULTISINE_FEATURES = [
    "x",
    "x_sin",
    "x_cos",
    "y_lag_1",
    "y_lag_10",
    "y_lag_50",
    "x_lag_1",
    "x_sin_lag_1",
    "x_cos_lag_1",
    "y_rollmean_10",
    "y_rollstd_10",
    "y_rollmean_50",
    "y_rollstd_50",
]

ELECTRICITY_FEATURES = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "y_day_mean",
    "y_week_mean",
    "y_lag_1",
    "y_lag_24",
    "y_lag_168",
    "hour_lag_1",
    "hour_lag_24",
    "dow_lag_1",
    "dow_lag_24",
    "month_lag_1",
    "month_lag_24",
    "is_weekend_lag_1",
    "is_weekend_lag_24",
    "hour_sin_lag_1",
    "hour_sin_lag_24",
    "hour_cos_lag_1",
    "hour_cos_lag_24",
    "month_sin_lag_1",
    "month_sin_lag_24",
    "month_cos_lag_1",
    "month_cos_lag_24",
    "y_day_mean_lag_1",
    "y_day_mean_lag_24",
    "y_week_mean_lag_1",
    "y_week_mean_lag_24",
    "y_rollmean_24",
    "y_rollstd_24",
    "y_rollmean_168",
    "y_rollstd_168",
]


EXCHANGE_FEATURES = [
    "day", "month", "year", "is_month_start", "is_month_end", "ret",
    "vol_5", "vol_21", "y_lag_1", "y_lag_5", "y_lag_21",
    "day_lag_1", "day_lag_5", "month_lag_1", "month_lag_5",
    "year_lag_1", "year_lag_5", "is_month_start_lag_1",
    "is_month_start_lag_5", "is_month_end_lag_1",
    "is_month_end_lag_5", "ret_lag_1", "ret_lag_5", "vol_5_lag_1",
    "vol_5_lag_5", "vol_21_lag_1", "vol_21_lag_5", "y_rollmean_5",
    "y_rollstd_5", "y_rollmean_21", "y_rollstd_21",
]

METRLA_FEATURES = [
    "hour", "dow", "month", "y_lag_1", "y_lag_12", "y_lag_288",
    "hour_lag_1", "hour_lag_12", "dow_lag_1", "dow_lag_12",
    "month_lag_1", "month_lag_12", "y_rollmean_12", "y_rollstd_12",
    "y_rollmean_288", "y_rollstd_288",
]

NN5_FEATURES = [
    "day", "dow", "month", "year", "is_weekend", "is_month_start",
    "is_month_end", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "y_lag_1", "y_lag_7", "y_lag_14", "y_lag_28", "day_lag_1",
    "day_lag_7", "dow_lag_1", "dow_lag_7", "month_lag_1",
    "month_lag_7", "year_lag_1", "year_lag_7", "is_weekend_lag_1",
    "is_weekend_lag_7", "is_month_start_lag_1", "is_month_start_lag_7",
    "is_month_end_lag_1", "is_month_end_lag_7", "dow_sin_lag_1",
    "dow_sin_lag_7", "dow_cos_lag_1", "dow_cos_lag_7",
    "month_sin_lag_1", "month_sin_lag_7", "month_cos_lag_1",
    "month_cos_lag_7", "y_rollmean_7", "y_rollstd_7", "y_rollmean_28",
    "y_rollstd_28",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    feature_cols: tuple[str, ...]
    train_batch_size: int
    validation_batch_size: int
    test_batch_size: int
    ale_batch_size: int


@dataclass(frozen=True)
class BenchmarkSelection:
    learning_rate: float
    l2: float = 0.0
    update_every: int | None = None
    keep_epochs: int = 0
    seeds: tuple[int, ...] = tuple(range(10))


DATASET_SPECS: dict[DatasetName, DatasetSpec] = {
    "electricity": DatasetSpec(
        "electricity", tuple(ELECTRICITY_FEATURES), 128, 256, 256, 128
    ),
    "exchange": DatasetSpec(
        "exchange", tuple(EXCHANGE_FEATURES), 256, 256, 256, 256
    ),
    "metrla": DatasetSpec(
        "metrla", tuple(METRLA_FEATURES), 128, 256, 256, 256
    ),
    "multisine": DatasetSpec(
        "multisine", tuple(MULTISINE_FEATURES), 256, 512, 512, 512
    ),
    "nn5": DatasetSpec(
        "nn5", tuple(NN5_FEATURES), 128, 256, 256, 128
    ),
    "polynomial": DatasetSpec(
        "polynomial", tuple(MULTISINE_FEATURES), 256, 512, 512, 512
    ),
}

# Validation-selected configurations from the original experiment workflow.
# Only hyperparameters are encoded here: no published metric is read or used.
_FIVE_SEEDS = tuple(range(5))
_TEN_SEEDS = tuple(range(10))
SELECTED_BENCHMARK_CONFIGS: dict[DatasetName, dict[MethodName, BenchmarkSelection]] = {
    "electricity": {
        "ale_frechet": BenchmarkSelection(1e-3, 1e-2, 10, 1, _FIVE_SEEDS),
        "soft": BenchmarkSelection(1e-3, 1e-2, seeds=_FIVE_SEEDS),
        "crossstitch": BenchmarkSelection(1e-3, seeds=(0, 1, 2, 3)),
        "hard": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
        "single_task_mlp": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
        "ple": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
        "mtan": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS)
    },
    "exchange": {
        "ale_frechet": BenchmarkSelection(1e-3, 1e-3, 10, 1, _TEN_SEEDS),
        "soft": BenchmarkSelection(1e-2, 1e-2, seeds=_TEN_SEEDS),
        "crossstitch": BenchmarkSelection(1e-2, seeds=_TEN_SEEDS),
        "hard": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "single_task_mlp": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "ple": BenchmarkSelection(1e-2, seeds=_TEN_SEEDS),
        "mtan": BenchmarkSelection(1e-2, seeds=_TEN_SEEDS),
    },
    "metrla": {
        "ale_frechet": BenchmarkSelection(1e-3, 1e-2, 10, 10, _FIVE_SEEDS),
        "soft": BenchmarkSelection(1e-3, 1e-2, seeds=_FIVE_SEEDS),
        "crossstitch": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
        "hard": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
        "single_task_mlp": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
        "ple": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
        "mtan": BenchmarkSelection(1e-3, seeds=_FIVE_SEEDS),
    },
    "multisine": {
        "ale_frechet": BenchmarkSelection(1e-2, 1e-2, 10, 1, _TEN_SEEDS),
        "soft": BenchmarkSelection(1e-2, 1e-2, seeds=_TEN_SEEDS),
        "crossstitch": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "hard": BenchmarkSelection(1e-2, seeds=_TEN_SEEDS),
        "single_task_mlp": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "ple": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "mtan": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
    },
    "nn5": {
        "ale_frechet": BenchmarkSelection(1e-3, 1e-2, 10, 3, _TEN_SEEDS),
        "soft": BenchmarkSelection(1e-3, 1e-2, seeds=_TEN_SEEDS),
        "crossstitch": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "hard": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "single_task_mlp": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "ple": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "mtan": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
    },
    "polynomial": {
        "ale_frechet": BenchmarkSelection(1e-3, 1e-3, 10, 3, _TEN_SEEDS),
        "soft": BenchmarkSelection(1e-3, 1e-2, seeds=_TEN_SEEDS),
        "crossstitch": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "hard": BenchmarkSelection(1e-2, seeds=_TEN_SEEDS),
        "single_task_mlp": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "ple": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
        "mtan": BenchmarkSelection(1e-3, seeds=_TEN_SEEDS),
    },
}

MANUSCRIPT_METHODS: tuple[MethodName, ...] = (
    "ale_frechet", "soft", "crossstitch", "hard",
    "single_task_mlp", "ple", "mtan",
)
METHOD_DISPLAY = {
    "ale_frechet": "ALE--Frechet",
    "soft": "Soft",
    "crossstitch": "Cross-Stitch",
    "hard": "Hard",
    "single_task_mlp": "Single-task MLP",
    "ple": "PLE",
    "mtan": "MTAN",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """All values needed to translate one selected paper run."""

    name: ExampleName
    feature_cols: tuple[str, ...]
    seed: int
    learning_rate: float
    l2: float
    update_every: int
    keep_epochs: int
    train_batch_size: int
    validation_batch_size: int
    test_batch_size: int
    ale_batch_size: int
    epochs: int = 1000
    early_stopping_epochs: int = 25
    n_intervals: int = 30
    n_guess: int = 1000
    input_hidden: int = 128
    soft_hidden: int = 128
    final_hidden: int = 64


PAPER_CONFIGS: dict[ExampleName, ExperimentConfig] = {
    "multisine": ExperimentConfig(
        name="multisine",
        feature_cols=tuple(MULTISINE_FEATURES),
        seed=9,
        learning_rate=1e-2,
        l2=1e-2,
        update_every=5,
        keep_epochs=3,
        train_batch_size=256,
        validation_batch_size=512,
        test_batch_size=512,
        ale_batch_size=512,
    ),
    "electricity": ExperimentConfig(
        name="electricity",
        feature_cols=tuple(ELECTRICITY_FEATURES),
        seed=0,
        learning_rate=1e-3,
        l2=1e-2,
        update_every=10,
        keep_epochs=10,
        train_batch_size=128,
        validation_batch_size=256,
        test_batch_size=256,
        ale_batch_size=128,
    ),
}


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    task_labels: list[str]
    ale_epoch: int
    similarity_epoch: int
    ale_curves: torch.Tensor
    feature_similarity: torch.Tensor
    task_similarity: np.ndarray
    best_model: dict
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the paper ALE--Frechet examples with ALEMTL and generate "
            "the illustrative-example figures from newly computed tensors."
        )
    )
    parser.add_argument(
        "--section",
        choices=("figures", "benchmark", "all"),
        default="figures",
        help=(
            "Part of the illustrative-example section to reproduce. "
            "'figures' reruns the two interpretability experiments; "
            "'benchmark' reruns the selected configurations for the RMSE table."
        ),
    )
    parser.add_argument(
        "--example",
        choices=("multisine", "electricity", "all"),
        default="multisine",
        help="Interpretability example to execute when --section includes figures.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/interim"),
        help="Root used for generated Multi-Sine data and default Electricity data.",
    )
    parser.add_argument(
        "--electricity-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing electricity_train.csv, electricity_val.csv, "
            "and electricity_test.csv. Defaults to DATA_ROOT/electricity."
        ),
    )
    parser.add_argument(
        "--prepare-real-datasets",
        action="store_true",
        help=(
            "Download and preprocess Exchange, METR-LA, and NN5 when they are "
            "selected for the benchmark. Raw files are cached separately."
        ),
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=None,
        help=(
            "Root containing raw dataset downloads. Defaults to the parent of "
            "--data-root, so data/interim uses data/raw."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/softwarex_illustrative_examples"),
        help="Root for run artifacts, CSV files, tensors, and paper-ready figures.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--prepare-electricity",
        action="store_true",
        help=(
            "Download and preprocess UCI Electricity Load Diagrams before training. "
            "By default, all meters are retained for the paper training run."
        ),
    )
    parser.add_argument(
        "--electricity-zip",
        type=Path,
        default=None,
        help="Optional local UCI ZIP; avoids downloading it again.",
    )
    parser.add_argument(
        "--electricity-training-max-tasks",
        type=int,
        default=None,
        help=(
            "Optional cap on Electricity tasks used for preprocessing and training. "
            "Leave unset for the paper run, which trains on all 370 meters."
        ),
    )
    parser.add_argument(
        "--electricity-figure-max-tasks",
        type=int,
        default=50,
        help=(
            "Number of trained Electricity tasks displayed in the manuscript figure. "
            "The original interpretability figure displays the first 50."
        ),
    )
    parser.add_argument(
        "--electricity-task-selection",
        choices=("first", "random"),
        default="first",
        help=(
            "How to choose meters only when a training task cap is supplied. "
            "The full paper run leaves the cap unset."
        ),
    )
    parser.add_argument(
        "--electricity-data-seed",
        type=int,
        default=42,
        help="Seed used only when --electricity-task-selection=random.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Overwrite generated/preprocessed dataset CSV files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the paper default of 1000 epochs.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Override the paper early-stopping patience of 25 epochs.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional trainer batch limit per train/validation epoch for debugging.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a small end-to-end check (few epochs/intervals and reduced data). "
            "The resulting values are not paper results."
        ),
    )
    parser.add_argument(
        "--multisine-feature-indices",
        nargs="*",
        type=int,
        default=(125, 123, 69),
        help="Latent feature indices plotted in the manuscript Multi-Sine figure.",
    )
    parser.add_argument(
        "--electricity-top-pairs",
        type=int,
        default=20,
        help="Number of strongest Electricity pairs shown in the graph/CSV.",
    )
    parser.add_argument(
        "--benchmark-datasets",
        nargs="+",
        choices=tuple(DATASET_SPECS),
        default=tuple(DATASET_SPECS),
        help="Datasets to include when --section includes benchmark.",
    )
    parser.add_argument(
        "--benchmark-methods",
        nargs="+",
        choices=tuple(METHOD_DISPLAY),
        default=MANUSCRIPT_METHODS,
        help=(
            "Methods to rerun. Defaults to the seven methods printed in the "
            "SoftwareX table; add mmoe explicitly to reproduce the companion benchmark."
        ),
    )
    parser.add_argument(
        "--benchmark-seeds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional seed override for every benchmark run. Leave unset to use "
            "the original selected seed sets (five or ten seeds depending on dataset)."
        ),
    )
    parser.add_argument(
        "--benchmark-max-tasks",
        type=int,
        default=None,
        help="Optional task cap for benchmark debugging; leave unset for paper runs.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count; zero is safest for reproducibility.",
    )
    return parser.parse_args()


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {preference!r} requested, but CUDA is unavailable.")
    return device


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # Older supported PyTorch releases.
            torch.use_deterministic_algorithms(True)


def _json_value(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items() if key != "state_dict"}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Dataset preparation translated from the original repository
# ---------------------------------------------------------------------------


def temporal_split_per_task(
    frame: pd.DataFrame,
    *,
    task_col: str,
    order_col: str,
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order-preserving split used by both original data generators."""

    if not math.isclose(train + validation + test, 1.0, abs_tol=1e-8):
        raise ValueError("train + validation + test must equal 1.")

    parts: list[pd.DataFrame] = []
    metadata: list[dict] = []
    for task, group in frame.groupby(task_col, sort=True):
        group = group.sort_values(order_col).reset_index(drop=True)
        n_total = len(group)
        n_train = int(n_total * train)
        n_validation = int(n_total * validation)
        n_test = n_total - n_train - n_validation
        parts.extend(
            [
                group.iloc[:n_train].assign(split="train"),
                group.iloc[n_train : n_train + n_validation].assign(split="val"),
                group.iloc[n_train + n_validation :].assign(split="test"),
            ]
        )
        metadata.append(
            {
                "task": task,
                "n_total": n_total,
                "n_train": n_train,
                "n_val": n_validation,
                "n_test": n_test,
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(metadata)


def make_supervised_sequence(
    frame: pd.DataFrame,
    *,
    task_col: str,
    order_col: str,
    y_col: str,
    covariates: Sequence[str],
    horizon: int,
    y_lags: Sequence[int],
    cov_lags: Sequence[int],
    roll_windows: Sequence[int],
) -> pd.DataFrame:
    """Build targets, lags, and strictly-past rolling statistics per task."""

    parts: list[pd.DataFrame] = []
    for _, group in frame.sort_values([task_col, order_col]).groupby(task_col, sort=True):
        group = group.sort_values(order_col).copy()
        group["y_target"] = group[y_col].shift(-horizon)
        for lag in y_lags:
            group[f"y_lag_{lag}"] = group[y_col].shift(lag)
        for covariate in covariates:
            if covariate not in group or not pd.api.types.is_numeric_dtype(group[covariate]):
                continue
            for lag in cov_lags:
                group[f"{covariate}_lag_{lag}"] = group[covariate].shift(lag)
        past = group[y_col].shift(1)
        for window in roll_windows:
            group[f"y_rollmean_{window}"] = past.rolling(window, min_periods=window).mean()
            group[f"y_rollstd_{window}"] = past.rolling(window, min_periods=window).std()

        required = ["y_target"] + [
            column
            for column in group.columns
            if column.startswith(("y_lag_", "y_roll"))
            or any(column.endswith(f"_lag_{lag}") for lag in cov_lags)
        ]
        group = group.dropna(subset=required)
        if not group.empty:
            parts.append(group)

    if not parts:
        raise RuntimeError("Supervised transformation produced no rows.")
    return pd.concat(parts, ignore_index=True)


def generate_multisine_data(
    output_dir: Path,
    *,
    force: bool = False,
    n_tasks: int = 5,
    n_samples: int = 2000,
    noise_std: float = 0.05,
    generation_seed: int = 42,
) -> dict[str, Path]:
    """Generate the original Multi-Sine panel and split-before-lag CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "multisine_train.csv",
        "validation": output_dir / "multisine_val.csv",
        "test": output_dir / "multisine_test.csv",
        "full": output_dir / "multisine_full.csv",
        "meta": output_dir / "multisine_meta.csv",
    }
    if not force and all(paths[key].exists() for key in ("train", "validation", "test")):
        return paths

    rng = np.random.default_rng(generation_seed)
    x = np.linspace(0, 2 * np.pi, n_samples)
    amplitudes = np.linspace(0.5, 1.5, n_tasks)
    frequencies = np.linspace(1.0, 2.0, n_tasks)
    phases = np.linspace(0, np.pi / 2, n_tasks)

    raw_parts: list[pd.DataFrame] = []
    for task_index in range(n_tasks):
        y = amplitudes[task_index] * np.sin(
            frequencies[task_index] * x + phases[task_index]
        )
        y = y + rng.normal(0, noise_std, size=n_samples)
        raw_parts.append(
            pd.DataFrame(
                {
                    "x": x.astype(float),
                    "x_sin": np.sin(x),
                    "x_cos": np.cos(x),
                    "y": y.astype(float),
                    "task": f"task_{task_index + 1}",
                    "amplitude": amplitudes[task_index],
                    "frequency": frequencies[task_index],
                    "phase": phases[task_index],
                    "seed": generation_seed,
                }
            )
        )
    raw = pd.concat(raw_parts, ignore_index=True)

    # The original generator splits the raw sequence first and only then builds
    # lagged features, preventing targets/lags from crossing split boundaries.
    raw_split, raw_meta = temporal_split_per_task(
        raw, task_col="task", order_col="x"
    )
    supervised_parts: list[pd.DataFrame] = []
    meta_parts: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        split_raw = raw_split.loc[raw_split["split"] == split_name].drop(columns="split")
        split_supervised = make_supervised_sequence(
            split_raw,
            task_col="task",
            order_col="x",
            y_col="y",
            covariates=["x", "x_sin", "x_cos"],
            horizon=1,
            y_lags=[1, 10, 50],
            cov_lags=[1],
            roll_windows=[10, 50],
        ).assign(split=split_name)
        supervised_parts.append(split_supervised)
        counts = split_supervised.groupby("task").size().rename(f"n_{split_name}")
        meta_parts.append(counts.to_frame())

    full = pd.concat(supervised_parts, ignore_index=True)
    full.to_csv(paths["full"], index=False)
    full.loc[full["split"] == "train"].to_csv(paths["train"], index=False)
    full.loc[full["split"] == "val"].to_csv(paths["validation"], index=False)
    full.loc[full["split"] == "test"].to_csv(paths["test"], index=False)

    meta = raw_meta.set_index("task")
    for part in meta_parts:
        meta = meta.join(part, how="left", rsuffix="_supervised")
    meta.reset_index().to_csv(paths["meta"], index=False)
    return paths



def generate_polynomial_data(
    output_dir: Path,
    *,
    force: bool = False,
    n_tasks: int = 5,
    n_samples: int = 2000,
    degree: int = 3,
    noise_std: float = 0.05,
    generation_seed: int = 123,
) -> dict[str, Path]:
    """Generate the original Polynomial panel and split-before-lag CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "polynomial_train.csv",
        "validation": output_dir / "polynomial_val.csv",
        "test": output_dir / "polynomial_test.csv",
        "full": output_dir / "polynomial_full.csv",
        "meta": output_dir / "polynomial_meta.csv",
    }
    if not force and all(paths[key].exists() for key in ("train", "validation", "test")):
        return paths

    rng = np.random.default_rng(generation_seed)
    x = np.linspace(-2, 2, n_samples)
    raw_parts: list[pd.DataFrame] = []
    for task_index in range(n_tasks):
        coefficients = rng.uniform(-1.5, 1.5, size=degree + 1)
        y = sum(coefficients[k] * (x ** k) for k in range(degree + 1))
        y = y + rng.normal(0, noise_std, size=n_samples)
        raw_parts.append(
            pd.DataFrame(
                {
                    "x": x.astype(float),
                    "x_sin": np.sin(x),
                    "x_cos": np.cos(x),
                    "y": y.astype(float),
                    "task": f"task_{task_index + 1}",
                    "degree": degree,
                    "seed": generation_seed,
                    **{f"w{k}": coefficients[k] for k in range(degree + 1)},
                }
            )
        )
    raw = pd.concat(raw_parts, ignore_index=True)
    raw_split, raw_meta = temporal_split_per_task(raw, task_col="task", order_col="x")

    supervised_parts: list[pd.DataFrame] = []
    meta_parts: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        split_raw = raw_split.loc[raw_split["split"] == split_name].drop(columns="split")
        split_supervised = make_supervised_sequence(
            split_raw,
            task_col="task",
            order_col="x",
            y_col="y",
            covariates=["x", "x_sin", "x_cos"],
            horizon=1,
            y_lags=[1, 10, 50],
            cov_lags=[1],
            roll_windows=[10, 50],
        ).assign(split=split_name)
        supervised_parts.append(split_supervised)
        counts = split_supervised.groupby("task").size().rename(f"n_{split_name}")
        meta_parts.append(counts.to_frame())

    full = pd.concat(supervised_parts, ignore_index=True)
    full.to_csv(paths["full"], index=False)
    full.loc[full["split"] == "train"].to_csv(paths["train"], index=False)
    full.loc[full["split"] == "val"].to_csv(paths["validation"], index=False)
    full.loc[full["split"] == "test"].to_csv(paths["test"], index=False)

    meta = raw_meta.set_index("task")
    for part in meta_parts:
        meta = meta.join(part, how="left", rsuffix="_supervised")
    meta.reset_index().to_csv(paths["meta"], index=False)
    return paths

def add_calendar_encodings(frame: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    frame = frame.copy()
    frame["hour"] = frame[time_col].dt.hour
    frame["dow"] = frame[time_col].dt.dayofweek
    frame["month"] = frame[time_col].dt.month
    frame["is_weekend"] = frame["dow"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)
    return frame


def _download_electricity_zip(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    print(f"Downloading UCI Electricity dataset to {destination} ...")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, destination.open("wb") as target:
            shutil.copyfileobj(response, target)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _choose_meter_columns(
    meter_columns: Sequence[str],
    *,
    max_tasks: int | None,
    selection: Literal["first", "random"],
    seed: int,
) -> list[str]:
    columns = list(meter_columns)
    if max_tasks is None or max_tasks >= len(columns):
        return columns
    if max_tasks <= 0:
        raise ValueError("The Electricity task cap must be positive.")
    if selection == "first":
        return columns[:max_tasks]
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(columns), size=max_tasks, replace=False)
    return [columns[int(index)] for index in indices]


def prepare_electricity_data(
    output_dir: Path,
    *,
    zip_path: Path | None,
    max_tasks: int | None,
    task_selection: Literal["first", "random"],
    selection_seed: int,
    force: bool,
    chunksize: int = 200_000,
) -> dict[str, Path]:
    """Translate the original chunk-safe UCI preprocessing into this script."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_task").mkdir(exist_ok=True)
    paths = {
        "train": output_dir / "electricity_train.csv",
        "validation": output_dir / "electricity_val.csv",
        "test": output_dir / "electricity_test.csv",
        "full": output_dir / "electricity_full.csv",
        "meta": output_dir / "electricity_meta.csv",
    }
    if not force and all(paths[key].exists() for key in ("train", "validation", "test")):
        return paths

    archive = zip_path or output_dir / "electricityloaddiagrams20112014.zip"
    archive = _download_electricity_zip(archive.expanduser().resolve())

    for path in paths.values():
        path.unlink(missing_ok=True)
    for path in (output_dir / "by_task").glob("electricity_*.csv"):
        path.unlink()

    with zipfile.ZipFile(archive) as compressed:
        text_members = [name for name in compressed.namelist() if name.lower().endswith(".txt")]
        if not text_members:
            raise RuntimeError("UCI Electricity ZIP contains no .txt member.")
        member = text_members[0]
        with compressed.open(member) as source:
            header = pd.read_csv(
                source,
                sep=";",
                decimal=",",
                nrows=0,
                engine="python",
                on_bad_lines="skip",
            )
    columns = list(header.columns)
    if len(columns) < 2:
        raise RuntimeError("Could not identify timestamp and meter columns in UCI data.")
    time_column = columns[0]
    meters = _choose_meter_columns(
        columns[1:],
        max_tasks=max_tasks,
        selection=task_selection,
        seed=selection_seed,
    )

    written = {"full": False, "train": False, "validation": False, "test": False}
    final_columns: list[str] | None = None
    metadata: list[dict] = []

    def append_frame(frame: pd.DataFrame, key: str) -> None:
        nonlocal final_columns
        if final_columns is None:
            final_columns = list(frame.columns)
        frame = frame.reindex(columns=final_columns)
        frame.to_csv(paths[key], mode="a", index=False, header=not written[key])
        written[key] = True

    for meter_index, meter in enumerate(meters, start=1):
        print(f"[{meter_index}/{len(meters)}] Preparing Electricity meter {meter}")
        hourly_sums: list[pd.DataFrame] = []
        hourly_counts: list[pd.DataFrame] = []
        with zipfile.ZipFile(archive) as compressed:
            with compressed.open(member) as source:
                reader = pd.read_csv(
                    source,
                    sep=";",
                    decimal=",",
                    usecols=[time_column, meter],
                    parse_dates=[time_column],
                    chunksize=chunksize,
                    low_memory=True,
                    engine="python",
                    on_bad_lines="skip",
                )
                for chunk in reader:
                    chunk = chunk.rename(columns={time_column: "time", meter: "y"})
                    chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
                    chunk["y"] = pd.to_numeric(chunk["y"], errors="coerce")
                    chunk = chunk.dropna(subset=["time", "y"])
                    if chunk.empty:
                        continue
                    indexed = chunk.set_index("time").sort_index()
                    sums = indexed["y"].resample("1h").sum(min_count=1).to_frame("y_sum")
                    counts = indexed["y"].resample("1h").count().to_frame("y_count")
                    merged = sums.join(counts, how="inner")
                    merged = merged.loc[merged["y_count"] > 0]
                    if not merged.empty:
                        hourly_sums.append(merged[["y_sum"]].reset_index())
                        hourly_counts.append(merged[["y_count"]].reset_index())

        if not hourly_sums:
            print(f"  No valid data for {meter}; skipping.")
            continue
        sum_frame = (
            pd.concat(hourly_sums, ignore_index=True)
            .groupby("time", as_index=False)["y_sum"]
            .sum()
        )
        count_frame = (
            pd.concat(hourly_counts, ignore_index=True)
            .groupby("time", as_index=False)["y_count"]
            .sum()
        )
        meter_frame = sum_frame.merge(count_frame, on="time", how="inner")
        meter_frame["y"] = (meter_frame["y_sum"] / meter_frame["y_count"]).astype("float32")
        meter_frame = (
            meter_frame.drop(columns=["y_sum", "y_count"])
            .dropna(subset=["time", "y"])
            .groupby("time", as_index=False)["y"]
            .mean()
            .sort_values("time")
            .reset_index(drop=True)
        )
        meter_frame["task"] = meter
        meter_frame = add_calendar_encodings(meter_frame, "time")
        meter_frame = meter_frame.sort_values(["task", "time"]).reset_index(drop=True)
        meter_frame["y_day_mean"] = meter_frame.groupby("task")["y"].transform(
            lambda values: values.shift(1).rolling(24, min_periods=24).mean()
        ).astype("float32")
        meter_frame["y_week_mean"] = meter_frame.groupby("task")["y"].transform(
            lambda values: values.shift(1).rolling(168, min_periods=168).mean()
        ).astype("float32")
        covariates = [
            column
            for column in meter_frame.columns
            if column not in {"time", "task", "y", "split"}
        ]
        supervised = make_supervised_sequence(
            meter_frame,
            task_col="task",
            order_col="time",
            y_col="y",
            covariates=covariates,
            horizon=1,
            y_lags=[1, 24, 168],
            cov_lags=[1, 24],
            roll_windows=[24, 168],
        )
        split, meta = temporal_split_per_task(
            supervised, task_col="task", order_col="time"
        )
        append_frame(split, "full")
        append_frame(split.loc[split["split"] == "train"], "train")
        append_frame(split.loc[split["split"] == "val"], "validation")
        append_frame(split.loc[split["split"] == "test"], "test")
        safe_meter = str(meter).replace("/", "_").replace(" ", "_")
        split.to_csv(output_dir / "by_task" / f"electricity_{safe_meter}.csv", index=False)
        metadata.append(meta.iloc[0].to_dict())

    if not metadata:
        raise RuntimeError("Electricity preprocessing produced no task datasets.")
    pd.DataFrame(metadata).to_csv(paths["meta"], index=False)
    return paths


def existing_dataset_paths(directory: Path, dataset: DatasetName) -> dict[str, Path]:
    paths = {
        "train": directory / f"{dataset}_train.csv",
        "validation": directory / f"{dataset}_val.csv",
        "test": directory / f"{dataset}_test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed dataset files:\n  - "
            + "\n  - ".join(missing)
            + ("\nRun with --prepare-electricity or pass --electricity-data-dir." if dataset == "electricity" else "")
        )
    return paths


# ---------------------------------------------------------------------------
# ALEMTL model/training translation
# ---------------------------------------------------------------------------


def load_split(path: Path, feature_cols: Sequence[str], *, max_tasks: int | None = None) -> pd.DataFrame:
    use_columns = ["task", "y_target", *feature_cols]
    frame = pd.read_csv(path, usecols=lambda column: column in set(use_columns))
    missing = sorted(set(use_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    if max_tasks is not None:
        # Preserve first occurrence order from the processed CSV.  This avoids
        # silently changing the publication task subset.
        tasks = frame["task"].drop_duplicates().astype(str).tolist()[:max_tasks]
        frame = frame.loc[frame["task"].astype(str).isin(tasks)].copy()
    numeric = ["y_target", *feature_cols]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=use_columns)
    if frame.empty:
        raise RuntimeError(f"No usable rows remained after loading {path}.")
    return frame[use_columns].reset_index(drop=True)


def build_dataloaders(
    paths: dict[str, Path],
    config: ExperimentConfig,
    *,
    num_workers: int,
    max_tasks: int | None = None,
) -> tuple[dict[str, MultitaskDataloader], list[str]]:
    frames = {
        split: load_split(path, config.feature_cols, max_tasks=max_tasks)
        for split, path in paths.items()
        if split in {"train", "validation", "test"}
    }
    task_sets = [set(frame["task"].astype(str)) for frame in frames.values()]
    common_tasks = set.intersection(*task_sets)
    if not common_tasks:
        raise RuntimeError("Train, validation, and test splits have no common tasks.")
    for split in frames:
        frames[split] = frames[split].loc[
            frames[split]["task"].astype(str).isin(common_tasks)
        ].reset_index(drop=True)

    datasets = {
        split: MultitaskDataset(frame, task_id="task", target_names=["y_target"])
        for split, frame in frames.items()
    }
    train_labels = [str(label) for label in datasets["train"].label_encoder_.classes_.tolist()]
    for split in ("validation", "test"):
        labels = [str(label) for label in datasets[split].label_encoder_.classes_.tolist()]
        if labels != train_labels:
            raise RuntimeError(
                f"Task ordering differs between train and {split}: {train_labels} vs {labels}."
            )

    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    loaders = {
        "train": MultitaskDataloader(
            datasets["train"], batch_size=config.train_batch_size, shuffle=True, **kwargs
        ),
        "validation": MultitaskDataloader(
            datasets["validation"],
            batch_size=config.validation_batch_size,
            shuffle=False,
            **kwargs,
        ),
        "test": MultitaskDataloader(
            datasets["test"], batch_size=config.test_batch_size, shuffle=False, **kwargs
        ),
        # The original trainer initializes and updates ALE from the training set.
        "ale": MultitaskDataloader(
            datasets["train"], batch_size=config.ale_batch_size, shuffle=False, **kwargs
        ),
    }
    return loaders, train_labels


def build_model(config: ExperimentConfig, n_tasks: int, device: torch.device) -> MultiTaskModel:
    input_size = len(config.feature_cols)
    layout = [
        (
            "hard_lstm",  # Historical name retained to match the original YAML.
            {
                "shared": "hard",
                "module": lambda: nn.Sequential(
                    nn.Linear(input_size, config.input_hidden), nn.ReLU()
                ),
            },
        ),
        (
            "soft1",
            {
                "shared": "soft",
                "module": lambda: nn.Sequential(
                    nn.Linear(config.input_hidden, config.soft_hidden), nn.ReLU()
                ),
            },
        ),
        (
            "soft2",
            {
                "shared": "soft",
                "module": lambda: nn.Sequential(
                    nn.Linear(config.soft_hidden, config.final_hidden), nn.ReLU()
                ),
            },
        ),
        (
            "head",
            {
                "shared": None,
                "module": lambda: nn.Linear(config.final_hidden, 1),
            },
        ),
    ]
    return MultiTaskModel(
        n_tasks=n_tasks,
        modules_layout=layout,
        similarity_layers={"in": "soft1", "out": "head"},
        device=device,
        same_parameters=False,
        shared_input_data=False,
        weight_init="kaiming_uniform",
        bias_range=0.1,
    )


def _latest_tensor(tracker, name: str) -> tuple[int, torch.Tensor]:
    values = tracker.track[name]["metrics"].track
    if not values:
        raise RuntimeError(
            f"No {name} tensor was computed. Ensure epochs >= update_every ({name})."
        )
    epoch = max(values)
    return int(epoch), values[epoch].detach().cpu().clone()


def train_experiment(
    config: ExperimentConfig,
    paths: dict[str, Path],
    *,
    device: torch.device,
    output_dir: Path,
    num_workers: int,
    max_tasks: int | None,
    max_batches: int | None,
) -> ExperimentResult:
    set_seed(config.seed)
    loaders, task_labels = build_dataloaders(
        paths, config, num_workers=num_workers, max_tasks=max_tasks
    )
    model = build_model(config, n_tasks=len(task_labels), device=device)

    ale = MultiTaskALE(
        model=model,
        dataloader=loaders["ale"],
        n_tasks=len(task_labels),
        shared_input_data=False,
        n_features_out=1,
        num_intervals=config.n_intervals,
        device=device,
        epsilon=1e-4,
        n_guess=config.n_guess,
    )
    similarity = MultitaskSimilarity(ale_curves=ale)
    losses = MultiTaskLoss(
        model=model,
        loss_fn=rmse_loss,
        errors_fn={"rmse": rmse_loss, "mae": mae_loss, "mape": mape_loss},
        l2_penalty=config.l2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    run_dir = output_dir / config.name / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = MultiTaskTrainer(
        model=model,
        train_dataloader=loaders["train"],
        validation_dataloader=loaders["validation"],
        test_dataloader=loaders["test"],
        optimizer=optimizer,
        scheduler=scheduler,
        loss=losses,
        ale=ale,
        multitask_similarity=similarity,
        maximize_loss=False,
        early_stopping_epochs=config.early_stopping_epochs,
        print_each_epochs=1,
        ale_each_epochs=config.update_every,
        similarity_each_epochs=config.update_every,
        keep_similarity_epochs=config.keep_epochs,
        track_device="cpu",
        track_epochs=None,
        save_results_each=None,
        logging_dir=str(run_dir),
        learning_type="regression",
        dataset_name=config.name,
        n_intervals_ale=config.n_intervals,
        learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        ale_batch_size=config.ale_batch_size,
        l2penalty=config.l2,
        architecture=(
            f"{config.input_hidden}-{config.soft_hidden}-"
            f"{config.final_hidden} ALE-Frechet"
        ),
        seed=config.seed,
        print_limit_epochs=5,
        config_info=asdict(config),
        amp=False,
    )
    trainer.train(epochs=config.epochs, max_batches=max_batches)

    ale_epoch, ale_curves = _latest_tensor(trainer.tracking, "ale")
    similarity_epoch, feature_similarity = _latest_tensor(trainer.tracking, "similarity")
    # The figure builder in the original repository uses the mean across latent
    # features, whereas training uses ALEMTL's summed score for regularization.
    task_similarity = feature_similarity.float().mean(dim=-1).numpy()

    result = ExperimentResult(
        config=config,
        task_labels=task_labels,
        ale_epoch=ale_epoch,
        similarity_epoch=similarity_epoch,
        ale_curves=ale_curves,
        feature_similarity=feature_similarity,
        task_similarity=task_similarity,
        best_model=trainer.get_best_model(),
        output_dir=output_dir / config.name,
    )
    save_raw_results(result)
    return result


# ---------------------------------------------------------------------------
# Paper output generation translated from the original plotting scripts
# ---------------------------------------------------------------------------


def centered_ale_array(ale_curves: torch.Tensor) -> np.ndarray:
    if ale_curves.ndim != 4 or ale_curves.shape[-1] < 2:
        raise ValueError(f"Unexpected ALE shape {tuple(ale_curves.shape)}.")
    values = ale_curves.detach().cpu().float().numpy().copy()
    y = values[..., 1]
    finite = np.isfinite(y)
    counts = finite.sum(axis=2, keepdims=True)
    sums = np.where(finite, y, 0.0).sum(axis=2, keepdims=True)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    values[..., 1] = np.where(finite, y - means, y)
    return values


def choose_ale_features(
    ale_curves: torch.Tensor,
    explicit: Sequence[int] | None,
    top_k: int = 3,
) -> list[int]:
    n_features = int(ale_curves.shape[1])
    if explicit:
        selected = list(dict.fromkeys(int(index) for index in explicit))
        valid = [index for index in selected if 0 <= index < n_features]
        if len(valid) == len(selected):
            return valid
        print(
            f"Requested ALE features {selected} are not all valid for {n_features} features; "
            "falling back to variance-ranked features."
        )
    y = centered_ale_array(ale_curves)[..., 1]
    scores = np.nanvar(y, axis=(0, 2))
    return [int(index) for index in np.argsort(scores)[::-1][: min(top_k, n_features)]]


def _ale_limits(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lower, upper = float(finite.min()), float(finite.max())
    if math.isclose(lower, upper):
        delta = max(abs(lower) * 0.1, 0.1)
        return lower - delta, upper + delta
    delta = (upper - lower) * padding
    return lower - delta, upper + delta


def _matrix_limits(matrix: np.ndarray) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def _circle_positions(n_nodes: int, radius: float = 1.0) -> np.ndarray:
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_nodes, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _scaled_strength(values: Iterable[float]) -> dict[float, float]:
    unique = list(values)
    if not unique:
        return {}
    low, high = min(unique), max(unique)
    if math.isclose(low, high):
        return {value: 1.0 for value in unique}
    return {value: (value - low) / (high - low) for value in unique}


def draw_nearest_graph(ax: plt.Axes, matrix: np.ndarray, labels: Sequence[str]) -> None:
    n_tasks = matrix.shape[0]
    positions = _circle_positions(n_tasks)
    scores = matrix.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbors = np.argmax(scores, axis=1)
    edge_values = [float(matrix[index, neighbor]) for index, neighbor in enumerate(neighbors)]
    scaling = _scaled_strength(edge_values)

    for index, neighbor in enumerate(neighbors):
        value = float(matrix[index, neighbor])
        strength = scaling.get(value, 0.5)
        start = positions[index]
        end = positions[neighbor]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            connectionstyle="arc3,rad=0.12",
            linewidth=0.7 + 2.3 * strength,
            alpha=0.25 + 0.7 * strength,
            color="black",
            shrinkA=13,
            shrinkB=13,
        )
        ax.add_patch(arrow)

    ax.scatter(positions[:, 0], positions[:, 1], s=270, zorder=3)
    for index, (x_coord, y_coord) in enumerate(positions):
        label = labels[index] if index < len(labels) else f"T{index + 1}"
        ax.text(x_coord, y_coord, label, ha="center", va="center", fontsize=7, zorder=4)
    ax.set_title("Nearest-neighbour sharing graph")
    ax.set_aspect("equal")
    ax.axis("off")


def strongest_pairs(
    matrix: np.ndarray, labels: Sequence[str], n_pairs: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for task_i in range(matrix.shape[0]):
        for task_j in range(task_i + 1, matrix.shape[1]):
            value = float(matrix[task_i, task_j])
            if np.isfinite(value):
                rows.append(
                    {
                        "task_i": labels[task_i],
                        "task_j": labels[task_j],
                        "task_i_index": task_i,
                        "task_j_index": task_j,
                        "similarity": value,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["task_i", "task_j", "task_i_index", "task_j_index", "similarity"])
    return (
        pd.DataFrame(rows)
        .sort_values("similarity", ascending=False, kind="mergesort")
        .head(n_pairs)
        .reset_index(drop=True)
    )


def draw_pair_graph(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    if pairs.empty:
        ax.text(0.5, 0.5, "No finite pairs", ha="center", va="center")
        ax.axis("off")
        return
    labels = list(dict.fromkeys(pairs["task_i"].tolist() + pairs["task_j"].tolist()))
    positions = _circle_positions(len(labels))
    position_map = {label: positions[index] for index, label in enumerate(labels)}
    scaling = _scaled_strength(pairs["similarity"].astype(float).tolist())
    for row in pairs.itertuples(index=False):
        value = float(row.similarity)
        strength = scaling.get(value, 0.5)
        start = position_map[str(row.task_i)]
        end = position_map[str(row.task_j)]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linewidth=0.5 + 3.0 * strength,
            alpha=0.2 + 0.75 * strength,
            color="black",
            zorder=1,
        )
    ax.scatter(positions[:, 0], positions[:, 1], s=130, zorder=2)
    for label, (x_coord, y_coord) in position_map.items():
        ax.text(x_coord, y_coord, label, fontsize=5.5, ha="center", va="center", zorder=3)
    ax.set_title("Strongest task pairs")
    ax.set_aspect("equal")
    ax.axis("off")


def save_similarity_csv(matrix: np.ndarray, labels: Sequence[str], path: Path) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("task").to_csv(path)


def save_ale_csv(ale_curves: torch.Tensor, labels: Sequence[str], path: Path) -> None:
    array = ale_curves.detach().cpu().float().numpy()
    rows: list[dict] = []
    for task_index in range(array.shape[0]):
        for feature_index in range(array.shape[1]):
            for interval in range(array.shape[2]):
                rows.append(
                    {
                        "task_index": task_index,
                        "task": labels[task_index],
                        "feature_index": feature_index,
                        "interval": interval,
                        "x": float(array[task_index, feature_index, interval, 0]),
                        "ale": float(array[task_index, feature_index, interval, 1]),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_raw_results(result: ExperimentResult) -> None:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ale_epoch": result.ale_epoch,
            "similarity_epoch": result.similarity_epoch,
            "ale_curves": result.ale_curves,
            "feature_similarity": result.feature_similarity,
            "task_labels": result.task_labels,
            "config": asdict(result.config),
        },
        result.output_dir / f"{result.config.name}_ale_similarity_tensors.pt",
    )
    save_similarity_csv(
        result.task_similarity,
        result.task_labels,
        result.output_dir / f"{result.config.name}_similarity_matrix.csv",
    )
    summary = {
        "config": asdict(result.config),
        "task_labels": result.task_labels,
        "ale_epoch_zero_based": result.ale_epoch,
        "similarity_epoch_zero_based": result.similarity_epoch,
        "ale_shape": list(result.ale_curves.shape),
        "feature_similarity_shape": list(result.feature_similarity.shape),
        "best_model": _json_value(result.best_model),
    }
    with (result.output_dir / f"{result.config.name}_training_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)


def build_multisine_outputs(
    result: ExperimentResult,
    *,
    feature_indices: Sequence[int] | None,
    dpi: int,
) -> None:
    output_dir = result.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"Task {index + 1}" for index in range(len(result.task_labels))]
    selected = choose_ale_features(result.ale_curves, feature_indices, top_k=3)
    centered = centered_ale_array(result.ale_curves)

    figure = plt.figure(figsize=(11.5, 6.6))
    grid = GridSpec(2, 3, figure=figure, height_ratios=[1.0, 1.15], hspace=0.36, wspace=0.34)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(labels))))
    for panel, feature_index in enumerate(selected):
        axis = figure.add_subplot(grid[0, panel])
        for task_index, label in enumerate(labels):
            axis.plot(
                centered[task_index, feature_index, :, 0],
                centered[task_index, feature_index, :, 1],
                linewidth=1.4,
                label=label,
                color=colors[task_index],
            )
        axis.set_title(f"Latent feature {feature_index}")
        axis.set_xlabel("ALE grid")
        axis.set_ylabel("Centered ALE")
        axis.set_ylim(*_ale_limits(centered[:, feature_index, :, 1]))
        axis.grid(alpha=0.25, linewidth=0.5)
        if panel == 0:
            axis.legend(frameon=False, ncol=2)

    heatmap_axis = figure.add_subplot(grid[1, :2])
    vmin, vmax = _matrix_limits(result.task_similarity)
    image = heatmap_axis.imshow(result.task_similarity, aspect="auto", vmin=vmin, vmax=vmax)
    heatmap_axis.set_title("Mean ALE--Fréchet task similarity")
    heatmap_axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    heatmap_axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            heatmap_axis.text(
                column,
                row,
                f"{result.task_similarity[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)

    graph_axis = figure.add_subplot(grid[1, 2])
    draw_nearest_graph(graph_axis, result.task_similarity, labels)
    figure.suptitle("Multi-Sine ALE--Fréchet interpretability", fontsize=12, y=0.995)

    base = output_dir / "multisine_interpretability"
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    save_ale_csv(result.ale_curves, result.task_labels, output_dir / "multisine_ale_curves.csv")

    matrix = result.task_similarity.copy()
    np.fill_diagonal(matrix, -np.inf)
    nearest = np.argmax(matrix, axis=1)
    nearest_rows = [
        {
            "task": result.task_labels[index],
            "nearest_task": result.task_labels[int(neighbor)],
            "similarity": float(result.task_similarity[index, neighbor]),
        }
        for index, neighbor in enumerate(nearest)
    ]
    pd.DataFrame(nearest_rows).to_csv(output_dir / "multisine_nearest_tasks.csv", index=False)

    task_2_3 = float(result.task_similarity[1, 2]) if len(labels) >= 3 else float("nan")
    task_5_3 = float(result.task_similarity[4, 2]) if len(labels) >= 5 else float("nan")
    with (output_dir / "multisine_manuscript_values.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"Selected latent features: {selected}\n")
        stream.write(f"Task 2--Task 3 similarity: {task_2_3:.8f}\n")
        stream.write(f"Task 5--Task 3 similarity: {task_5_3:.8f}\n")
        stream.write(f"ALE tensor shape: {tuple(result.ale_curves.shape)}\n")
        stream.write(f"Feature-similarity tensor shape: {tuple(result.feature_similarity.shape)}\n")

    print(f"Multi-Sine selected latent features: {selected}")
    print(f"Multi-Sine Task 2--Task 3 similarity: {task_2_3:.4f}")
    print(f"Multi-Sine Task 5--Task 3 similarity: {task_5_3:.4f}")


def build_electricity_outputs(
    result: ExperimentResult,
    *,
    n_pairs: int,
    max_figure_tasks: int | None,
    dpi: int,
) -> None:
    output_dir = result.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_task_count = len(result.task_labels)
    if max_figure_tasks is not None and max_figure_tasks <= 0:
        raise ValueError("electricity-figure-max-tasks must be positive.")
    figure_task_count = (
        full_task_count
        if max_figure_tasks is None
        else min(max_figure_tasks, full_task_count)
    )
    labels = result.task_labels[:figure_task_count]
    matrix = result.task_similarity[:figure_task_count, :figure_task_count]
    pairs = strongest_pairs(matrix, labels, n_pairs)
    pairs.to_csv(output_dir / "electricity_top_pairs.csv", index=False)

    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    off_diagonal = off_diagonal[np.isfinite(off_diagonal)]

    figure = plt.figure(figsize=(12.0, 4.0))
    grid = GridSpec(1, 3, figure=figure, width_ratios=[1.15, 0.8, 1.05], wspace=0.35)

    heatmap_axis = figure.add_subplot(grid[0, 0])
    vmin, vmax = _matrix_limits(matrix)
    image = heatmap_axis.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    heatmap_axis.set_title(f"Task-similarity matrix ({figure_task_count} tasks)")
    heatmap_axis.set_xlabel("Task")
    heatmap_axis.set_ylabel("Task")
    tick_step = max(1, len(labels) // 10)
    ticks = list(range(0, len(labels), tick_step))
    heatmap_axis.set_xticks(ticks, [labels[index] for index in ticks], rotation=90)
    heatmap_axis.set_yticks(ticks, [labels[index] for index in ticks])
    figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)

    histogram_axis = figure.add_subplot(grid[0, 1])
    histogram_axis.hist(off_diagonal, bins=30)
    histogram_axis.set_title("Off-diagonal similarities")
    histogram_axis.set_xlabel("Similarity")
    histogram_axis.set_ylabel("Frequency")
    histogram_axis.grid(alpha=0.2, linewidth=0.5)

    graph_axis = figure.add_subplot(grid[0, 2])
    draw_pair_graph(graph_axis, pairs)
    figure.suptitle("Electricity ALE--Fréchet similarity structure", fontsize=12, y=1.02)

    base = output_dir / "electricity_similarity_structure"
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    with (output_dir / "electricity_manuscript_values.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"Number of trained tasks: {full_task_count}\n")
        stream.write(f"Number of tasks displayed: {figure_task_count}\n")
        stream.write(f"ALE tensor shape: {tuple(result.ale_curves.shape)}\n")
        stream.write(f"Feature-similarity tensor shape: {tuple(result.feature_similarity.shape)}\n")
        stream.write(f"Off-diagonal minimum: {float(off_diagonal.min()):.8f}\n")
        stream.write(f"Off-diagonal mean: {float(off_diagonal.mean()):.8f}\n")
        stream.write(f"Off-diagonal maximum: {float(off_diagonal.max()):.8f}\n")
        stream.write("Strongest pairs:\n")
        for row in pairs.itertuples(index=False):
            stream.write(f"  {row.task_i} -- {row.task_j}: {row.similarity:.8f}\n")

    if not pairs.empty:
        first = pairs.iloc[0]
        print(
            "Electricity strongest pair: "
            f"{first['task_i']}--{first['task_j']} ({float(first['similarity']):.4f})"
        )


def smoke_config(config: ExperimentConfig) -> ExperimentConfig:
    return replace(
        config,
        epochs=1,
        early_stopping_epochs=1,
        update_every=1,
        keep_epochs=1,
        n_intervals=2,
        n_guess=2,
        input_hidden=2,
        soft_hidden=2,
        final_hidden=2,
        train_batch_size=min(config.train_batch_size, 32),
        validation_batch_size=min(config.validation_batch_size, 64),
        test_batch_size=min(config.test_batch_size, 64),
        ale_batch_size=min(config.ale_batch_size, 64),
    )


def run_one(name: ExampleName, args: argparse.Namespace, device: torch.device) -> ExperimentResult:
    config = PAPER_CONFIGS[name]
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.early_stopping is not None:
        config = replace(config, early_stopping_epochs=args.early_stopping)
    if args.smoke:
        config = smoke_config(config)

    if name == "multisine":
        data_dir = args.data_root / "multisine"
        paths = generate_multisine_data(
            data_dir,
            force=args.force_data,
            n_tasks=2 if args.smoke else 5,
            n_samples=540 if args.smoke else 2000,
        )
        max_tasks = None
    else:
        data_dir = args.electricity_data_dir or (args.data_root / "electricity")
        if args.prepare_electricity:
            paths = prepare_electricity_data(
                data_dir,
                zip_path=args.electricity_zip,
                max_tasks=(
                    min(args.electricity_training_max_tasks or 5, 5)
                    if args.smoke
                    else args.electricity_training_max_tasks
                ),
                task_selection=args.electricity_task_selection,
                selection_seed=args.electricity_data_seed,
                force=args.force_data,
            )
        else:
            paths = existing_dataset_paths(data_dir, "electricity")
        max_tasks = (
            min(args.electricity_training_max_tasks or 5, 5)
            if args.smoke
            else args.electricity_training_max_tasks
        )

    result = train_experiment(
        config,
        paths,
        device=device,
        output_dir=args.out_dir,
        num_workers=args.num_workers,
        max_tasks=max_tasks,
        max_batches=args.max_batches,
    )
    if name == "multisine":
        build_multisine_outputs(
            result,
            feature_indices=args.multisine_feature_indices,
            dpi=args.dpi,
        )
    else:
        build_electricity_outputs(
            result,
            n_pairs=args.electricity_top_pairs,
            max_figure_tasks=(
                min(args.electricity_figure_max_tasks, len(result.task_labels))
                if args.smoke
                else args.electricity_figure_max_tasks
            ),
            dpi=args.dpi,
        )
    return result



# ---------------------------------------------------------------------------
# Six-dataset predictive table translated from train_compare_methods.py
# ---------------------------------------------------------------------------


class DenseTaskPairSimilarity:
    """Fixed all-pairs graph used by the original SoftSharing baseline."""

    def __init__(self, n_tasks: int, device: torch.device) -> None:
        self.n_tasks = int(n_tasks)
        self.device = device
        if self.n_tasks <= 1:
            self.groups = [[0, 0]]
        else:
            self.groups = [
                [left, right]
                for left in range(self.n_tasks)
                for right in range(left + 1, self.n_tasks)
            ]
        self.weights = torch.full(
            (len(self.groups),),
            1.0 / max(1, len(self.groups)),
            device=device,
            dtype=torch.float32,
        )
        self.similarity_tasks_features = torch.ones(
            (self.n_tasks, self.n_tasks, 1), device=device, dtype=torch.float32
        )

    def compute(self) -> None:
        return None

    def tasks_groups(self) -> tuple[torch.Tensor, list[list[int]]]:
        return self.weights, self.groups


def benchmark_experiment_config(
    spec: DatasetSpec,
    selection: BenchmarkSelection,
    seed: int,
    *,
    smoke: bool,
    epochs_override: int | None,
    early_stopping_override: int | None,
) -> ExperimentConfig:
    config = ExperimentConfig(
        name=spec.name,  # type: ignore[arg-type]
        feature_cols=spec.feature_cols,
        seed=seed,
        learning_rate=selection.learning_rate,
        l2=selection.l2,
        update_every=selection.update_every or 0,
        keep_epochs=selection.keep_epochs,
        train_batch_size=spec.train_batch_size,
        validation_batch_size=spec.validation_batch_size,
        test_batch_size=spec.test_batch_size,
        ale_batch_size=spec.ale_batch_size,
        epochs=1000 if epochs_override is None else epochs_override,
        early_stopping_epochs=(
            25 if early_stopping_override is None else early_stopping_override
        ),
    )
    return smoke_config(config) if smoke else config


def prepare_benchmark_dataset(
    dataset: DatasetName,
    args: argparse.Namespace,
) -> dict[str, Path]:
    directory = args.data_root / dataset
    smoke_tasks = 2 if args.smoke else 5
    smoke_samples = 540 if args.smoke else 2000
    if dataset == "multisine":
        return generate_multisine_data(
            directory,
            force=args.force_data,
            n_tasks=smoke_tasks,
            n_samples=smoke_samples,
        )
    if dataset == "polynomial":
        return generate_polynomial_data(
            directory,
            force=args.force_data,
            n_tasks=smoke_tasks,
            n_samples=smoke_samples,
        )
    if dataset == "electricity" and args.prepare_electricity:
        return prepare_electricity_data(
            args.electricity_data_dir or directory,
            zip_path=args.electricity_zip,
            max_tasks=(
                min(args.benchmark_max_tasks or 5, 5)
                if args.smoke
                else args.benchmark_max_tasks
            ),
            task_selection=args.electricity_task_selection,
            selection_seed=args.electricity_data_seed,
            force=args.force_data,
        )
    if dataset == "electricity" and args.electricity_data_dir is not None:
        directory = args.electricity_data_dir
    if dataset in {"exchange", "metrla", "nn5"} and args.prepare_real_datasets:
        try:
            from scripts.prepare_real_datasets import prepare_dataset
        except ImportError as error:
            raise RuntimeError(
                "scripts/prepare_real_datasets.py is required when "
                "--prepare-real-datasets is used. Run this command from the "
                "ALEMTL source repository or source distribution."
            ) from error
        raw_data_root = args.raw_data_root or args.data_root.parent
        prepare_dataset(
            dataset,
            data_root=raw_data_root,
            output_dir=directory,
            force=args.force_data,
            max_tasks=(
                min(args.benchmark_max_tasks or 2, 2)
                if args.smoke
                else args.benchmark_max_tasks
            ),
        )
    try:
        return existing_dataset_paths(directory, dataset)
    except FileNotFoundError as error:
        if dataset in {"exchange", "metrla", "nn5"}:
            raise FileNotFoundError(
                f"{error}\nRun scripts/prepare_real_datasets.py --dataset {dataset} "
                "or rerun this command with --prepare-real-datasets. No archived "
                "model output or published paper result is read."
            ) from error
        raise


def build_benchmark_model(
    method: MethodName,
    config: ExperimentConfig,
    n_tasks: int,
    device: torch.device,
):
    if method == "ale_frechet":
        return build_model(config, n_tasks=n_tasks, device=device)

    input_dim = len(config.feature_cols)
    if config.input_hidden <= 2:  # smoke architecture
        hidden = [2, 2]
        head = [2]
        expert = [2]
        tower = [2]
        n_experts = 2
        n_shared_experts = 2
        n_task_experts = 1
    else:
        hidden = [128, 128]
        head = [64]
        expert = [128]
        tower = [64]
        n_experts = 8
        n_shared_experts = 4
        n_task_experts = 2

    common = dict(n_tasks=n_tasks, input_dim=input_dim, output_dim=1, device=device)
    if method == "soft":
        return SoftSharing(hidden=hidden, **common)
    if method == "single_task_mlp":
        return SingleTaskMLP(hidden=hidden, **common)
    if method == "hard":
        return HardSharing(trunk=hidden, head=head, **common)
    if method == "crossstitch":
        return CrossStitch(shared_dims=hidden, init_identity=True, **common)
    if method == "mmoe":
        return MMoE(
            n_experts=n_experts,
            expert_hidden=expert,
            tower_hidden=tower,
            **common,
        )
    if method == "ple":
        return PLE(
            n_layers=2,
            n_shared_experts=n_shared_experts,
            n_task_experts=n_task_experts,
            expert_hidden=expert,
            tower_hidden=tower,
            **common,
        )
    if method == "mtan":
        return MTAN(shared_dims=hidden, tower_hidden=tower, **common)
    raise ValueError(f"Unsupported method: {method}")


def train_benchmark_run(
    dataset: DatasetName,
    method: MethodName,
    selection: BenchmarkSelection,
    seed: int,
    paths: dict[str, Path],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    spec = DATASET_SPECS[dataset]
    config = benchmark_experiment_config(
        spec,
        selection,
        seed,
        smoke=args.smoke,
        epochs_override=args.epochs,
        early_stopping_override=args.early_stopping,
    )
    set_seed(seed)
    task_cap = (
        min(args.benchmark_max_tasks or 2, 2)
        if args.smoke
        else args.benchmark_max_tasks
    )
    loaders, task_labels = build_dataloaders(
        paths,
        config,
        num_workers=args.num_workers,
        max_tasks=task_cap,
    )
    model = build_benchmark_model(method, config, len(task_labels), device)

    ale = None
    similarity = None
    ale_every = None
    similarity_every = None
    keep_epochs = 0
    l2 = selection.l2 if method in {"ale_frechet", "soft"} else 0.0

    if method == "ale_frechet":
        ale = MultiTaskALE(
            model=model,
            dataloader=loaders["ale"],
            n_tasks=len(task_labels),
            shared_input_data=False,
            n_features_out=1,
            num_intervals=config.n_intervals,
            device=device,
            epsilon=1e-4,
            n_guess=config.n_guess,
        )
        similarity = MultitaskSimilarity(ale_curves=ale)
        ale_every = config.update_every
        similarity_every = config.update_every
        keep_epochs = config.keep_epochs
    elif method == "soft":
        similarity = DenseTaskPairSimilarity(len(task_labels), device)

    loss = MultiTaskLoss(
        model=model,
        loss_fn=rmse_loss,
        errors_fn={"rmse": rmse_loss, "mae": mae_loss, "mape": mape_loss},
        l2_penalty=l2,
    )
    if method == "soft":
        _, groups = similarity.tasks_groups()  # type: ignore[union-attr]
        loss.update_tasks_groups(groups)

    optimizer = torch.optim.Adam(model.parameters(), lr=selection.learning_rate)
    run_dir = args.out_dir / "benchmark" / dataset / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = MultiTaskTrainer(
        model=model,
        train_dataloader=loaders["train"],
        validation_dataloader=loaders["validation"],
        test_dataloader=loaders["test"],
        optimizer=optimizer,
        loss=loss,
        ale=ale,
        multitask_similarity=similarity,
        scheduler=None,
        maximize_loss=False,
        early_stopping_epochs=config.early_stopping_epochs,
        print_each_epochs=max(1, config.epochs),
        ale_each_epochs=ale_every,
        similarity_each_epochs=similarity_every,
        keep_similarity_epochs=keep_epochs,
        track_device="cpu",
        track_epochs=None,
        save_results_each=None,
        logging_dir=str(run_dir),
        learning_type="regression",
        dataset_name=dataset,
        n_intervals_ale=config.n_intervals,
        learning_rate=selection.learning_rate,
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        ale_batch_size=config.ale_batch_size,
        l2penalty=l2,
        architecture=METHOD_DISPLAY[method],
        seed=seed,
        print_limit_epochs=1,
        config_info={
            "dataset": dataset,
            "method": method,
            "selection": asdict(selection),
            "experiment": asdict(config),
        },
        amp=False,
    )
    trainer.train(epochs=config.epochs, max_batches=args.max_batches)
    best = trainer.get_best_model()
    row: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "method_display": METHOD_DISPLAY[method],
        "seed": seed,
        "n_tasks": len(task_labels),
        "best_epoch": int(best["epoch"]),
        "learning_rate": selection.learning_rate,
        "l2": l2,
        "update_every": selection.update_every,
        "keep_epochs": selection.keep_epochs,
    }
    for split_key, source in (
        ("train", best.get("train_metrics", {})),
        ("validation", best.get("val_metrics", {})),
        ("test", best.get("test_metrics", {})),
    ):
        for metric in ("rmse", "mae", "mape", "LOSS"):
            value = source.get(metric, float("nan")) if isinstance(source, dict) else float("nan")
            row[f"{split_key}_{metric.lower()}"] = float(value)

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_value(row), stream, indent=2)
    del trainer, optimizer, loss, similarity, ale, model, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def format_latex_rmse_table(summary: pd.DataFrame, methods: Sequence[MethodName]) -> str:
    dataset_order = [name for name in DATASET_SPECS if name in set(summary["dataset"])]
    lines = [
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        "Dataset & " + " & ".join(METHOD_DISPLAY[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    for dataset in dataset_order:
        subset = summary.loc[summary["dataset"] == dataset].set_index("method")
        ordered_means = pd.Series(
            {method: float(subset.loc[method, "test_rmse_mean"]) for method in methods}
        )
        rank = ordered_means.rank(method="min")
        cells = []
        for method in methods:
            mean = float(subset.loc[method, "test_rmse_mean"])
            std = float(subset.loc[method, "test_rmse_std"])
            text = f"{mean:.4f} $\\pm$ {std:.4f}"
            if rank[method] == 1:
                text = r"\textbf{" + text + "}"
            elif rank[method] == 2:
                text = r"\underline{" + text + "}"
            cells.append(text)
        label = dataset.replace("multisine", "Multi-Sine").replace("metrla", "METR-LA").replace("nn5", "NN5").title()
        label = label.replace("Metr-La", "METR-LA").replace("Nn5", "NN5")
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    rank_matrix = summary.pivot(index="dataset", columns="method", values="test_rmse_mean")
    average_ranks = rank_matrix[list(methods)].rank(axis=1, method="average").mean(axis=0)
    lines.extend(
        [
            r"\midrule",
            "Average rank & "
            + " & ".join(f"{float(average_ranks[m]):.4f}" for m in methods)
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    datasets: list[DatasetName] = list(dict.fromkeys(args.benchmark_datasets))
    methods: list[MethodName] = list(dict.fromkeys(args.benchmark_methods))
    rows: list[dict[str, object]] = []
    if args.benchmark_seeds == []:
        raise ValueError(
            "--benchmark-seeds was provided without seed values. "
            "Omit the option to use the paper seed sets, or provide one or more integers."
        )
    benchmark_root = args.out_dir / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        paths = prepare_benchmark_dataset(dataset, args)
        for method in methods:
            selection = SELECTED_BENCHMARK_CONFIGS[dataset][method]
            if args.benchmark_seeds is not None:
                seeds = tuple(args.benchmark_seeds)
            elif args.smoke:
                seeds = (selection.seeds[0],)
            else:
                seeds = selection.seeds
            for seed in seeds:
                print(f"\nBenchmark: dataset={dataset}, method={method}, seed={seed}")
                row = train_benchmark_run(
                    dataset,
                    method,
                    selection,
                    seed,
                    paths,
                    args=args,
                    device=device,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(benchmark_root / "benchmark_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["dataset", "method", "method_display"], as_index=False)
        .agg(
            seed_n=("seed", "nunique"),
            validation_rmse_mean=("validation_rmse", "mean"),
            validation_rmse_std=("validation_rmse", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
        )
        .sort_values(["dataset", "method"])
    )
    summary["validation_rmse_std"] = summary["validation_rmse_std"].fillna(0.0)
    summary["test_rmse_std"] = summary["test_rmse_std"].fillna(0.0)
    summary.to_csv(benchmark_root / "benchmark_mean_std.csv", index=False)

    mean_wide = summary.pivot(index="dataset", columns="method", values="test_rmse_mean")
    std_wide = summary.pivot(index="dataset", columns="method", values="test_rmse_std")
    formatted = pd.DataFrame(index=mean_wide.index)
    for method in methods:
        formatted[METHOD_DISPLAY[method]] = [
            f"{mean_wide.loc[dataset, method]:.4f} ± {std_wide.loc[dataset, method]:.4f}"
            for dataset in mean_wide.index
        ]
    formatted.to_csv(benchmark_root / "illustrative_rmse_table.csv")

    ranks = mean_wide[methods].rank(axis=1, method="average")
    average_ranks = ranks.mean(axis=0).rename("average_rank").reset_index()
    average_ranks.columns = ["method", "average_rank"]
    average_ranks["method_display"] = average_ranks["method"].map(METHOD_DISPLAY)
    average_ranks.sort_values("average_rank").to_csv(
        benchmark_root / "illustrative_average_ranks.csv", index=False
    )
    (benchmark_root / "illustrative_rmse_table.tex").write_text(
        format_latex_rmse_table(summary, methods), encoding="utf-8"
    )
    selected = {
        dataset: {
            method: _json_value(asdict(SELECTED_BENCHMARK_CONFIGS[dataset][method]))
            for method in methods
        }
        for dataset in datasets
    }
    with (benchmark_root / "selected_configurations.json").open("w", encoding="utf-8") as stream:
        json.dump(selected, stream, indent=2)
    return {
        "datasets": datasets,
        "methods": methods,
        "n_runs": len(rows),
        "runs_csv": str(benchmark_root / "benchmark_runs.csv"),
        "summary_csv": str(benchmark_root / "benchmark_mean_std.csv"),
        "table_csv": str(benchmark_root / "illustrative_rmse_table.csv"),
        "table_tex": str(benchmark_root / "illustrative_rmse_table.tex"),
        "ranks_csv": str(benchmark_root / "illustrative_average_ranks.csv"),
    }

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.out_dir = args.out_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.electricity_data_dir is not None:
        args.electricity_data_dir = args.electricity_data_dir.expanduser().resolve()
    if args.raw_data_root is not None:
        args.raw_data_root = args.raw_data_root.expanduser().resolve()
    if args.electricity_zip is not None:
        args.electricity_zip = args.electricity_zip.expanduser().resolve()

    print(f"ALEMTL illustrative-section reproduction on {device}")
    print("No archived ALE tensor, similarity matrix, or published metric is read.")
    manifest: dict[str, object] = {
        "section": args.section,
        "device": str(device),
        "smoke": bool(args.smoke),
    }

    if args.section in {"figures", "all"}:
        names: list[ExampleName] = (
            ["multisine", "electricity"] if args.example == "all" else [args.example]
        )
        results = [run_one(name, args, device) for name in names]
        manifest["figures"] = {
            result.config.name: {
                "config": asdict(result.config),
                "task_labels": result.task_labels,
                "ale_epoch_zero_based": result.ale_epoch,
                "similarity_epoch_zero_based": result.similarity_epoch,
                "ale_shape": list(result.ale_curves.shape),
                "feature_similarity_shape": list(result.feature_similarity.shape),
                "output_dir": str(result.output_dir),
            }
            for result in results
        }

    if args.section in {"benchmark", "all"}:
        manifest["benchmark"] = run_benchmark(args, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "reproduction_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_value(manifest), stream, indent=2)
    print(f"Outputs written to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
