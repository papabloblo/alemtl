#!/usr/bin/env python3
"""Download and preprocess the real datasets used by the SoftwareX example.

This module is adapted from ``scripts/generate_real_datasets.py`` in the
ALE--Frechet MTL reproducibility repository:
https://github.com/papabloblo/ale-frechet-mtdl-reproducibility

The original code is distributed under the MIT License. See
``THIRD_PARTY_NOTICES.md``. The adaptation keeps the original feature
engineering and leakage-safe temporal splitting while removing dependencies on
the original experiment configuration system.

The generated hierarchy is compatible with
``examples/reproduce_softwarex_illustrative_examples.py``::

    data/
    ├── raw/
    │   ├── exchange/
    │   ├── metrla/
    │   └── nn5/
    └── interim/
        ├── exchange/exchange_{train,val,test}.csv
        ├── metrla/metrla_{train,val,test}.csv
        └── nn5/nn5_{train,val,test}.csv

Raw files are cached and every processed dataset includes a JSON manifest with
its source URL, SHA-256 digest, preprocessing parameters, row counts, and task
counts. The raw datasets retain their original licences and terms of use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

EXCHANGE_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
METRLA_RECORD_API = "https://zenodo.org/api/records/5146275"
NN5_URL = (
    "https://zenodo.org/records/4656117/files/"
    "nn5_daily_dataset_without_missing_values.zip?download=1"
)
DEFAULT_CURRENCIES = ("USD", "GBP", "JPY", "CHF", "AUD", "CAD", "CNY")
USER_AGENT = "ALEMTL/0.1.0 (SoftwareX reproducibility dataset preparation)"


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    source_url: str
    raw_file: str
    raw_sha256: str
    generated_at_utc: str
    output_directory: str
    n_tasks: int
    n_rows: int
    split_rows: dict[str, int]
    parameters: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, force: bool = False) -> Path:
    """Download *url* atomically, reusing a cached file unless ``force``."""

    destination = destination.expanduser().resolve()
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
            shutil.copyfileobj(response, temporary)
    temp_path.replace(destination)
    return destination


def download_json(url: str) -> dict[str, Any]:
    """Download and decode a JSON document."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}.")
    return payload


def temporal_split_per_task(
    frame: pd.DataFrame,
    *,
    task_col: str = "task",
    time_col: str = "time",
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split every task sequentially, preserving chronological order."""

    if not np.isclose(train + validation + test, 1.0):
        raise ValueError("train + validation + test must equal 1.")
    parts: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for task, group in frame.groupby(task_col, sort=True):
        group = group.sort_values(time_col).reset_index(drop=True)
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
    if not parts:
        raise RuntimeError("Temporal splitting received no task rows.")
    return pd.concat(parts, ignore_index=True), pd.DataFrame(metadata)


def make_supervised_panel(
    frame: pd.DataFrame,
    *,
    task_col: str = "task",
    time_col: str = "time",
    y_col: str = "y",
    covariates: Sequence[str] | None = None,
    horizon: int = 1,
    y_lags: Sequence[int] = (1,),
    cov_lags: Sequence[int] = (),
    roll_windows: Sequence[int] = (),
) -> pd.DataFrame:
    """Create targets, lags, and strictly-past rolling features per task."""

    frame = frame.copy().sort_values([task_col, time_col])
    if covariates is None:
        excluded = {task_col, time_col, y_col, "split"}
        covariates = [column for column in frame.columns if column not in excluded]

    parts: list[pd.DataFrame] = []
    for task, group in frame.groupby(task_col, sort=False):
        group = group.sort_values(time_col).copy()
        if task_col not in group.columns:
            group[task_col] = task
        group["y_target"] = group[y_col].shift(-horizon)
        for lag in y_lags:
            group[f"y_lag_{lag}"] = group[y_col].shift(lag)
        for covariate in covariates:
            if covariate not in group or not pd.api.types.is_numeric_dtype(group[covariate]):
                continue
            for lag in cov_lags:
                group[f"{covariate}_lag_{lag}"] = group[covariate].shift(lag)
        past_y = group[y_col].shift(1)
        for window in roll_windows:
            group[f"y_rollmean_{window}"] = past_y.rolling(
                window, min_periods=window
            ).mean()
            group[f"y_rollstd_{window}"] = past_y.rolling(
                window, min_periods=window
            ).std()
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
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def standardize_per_task(
    combined: pd.DataFrame,
    *,
    task_col: str = "task",
    split_col: str = "split",
    exclude_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score numeric predictors per task using training statistics only."""

    frame = combined.copy()
    if exclude_cols is None:
        exclude_cols = (
            task_col,
            split_col,
            "time",
            "y",
            "y_target",
            "hour",
            "dow",
            "month",
        )
    numeric = [
        column
        for column in frame.select_dtypes(include=["number"]).columns
        if column not in set(exclude_cols)
    ]
    if not numeric:
        return frame, pd.DataFrame(columns=["task", "feature", "mean", "std"])
    frame[numeric] = frame[numeric].astype("float64")
    stats: list[pd.DataFrame] = []
    for task, group in frame.groupby(task_col):
        training = group.loc[group[split_col] == "train"]
        if training.empty:
            raise RuntimeError(f"Task {task!r} has no training rows.")
        means = training[numeric].mean()
        standard_deviations = training[numeric].std(ddof=0).replace(0, 1.0)
        frame.loc[group.index, numeric] = (group[numeric] - means) / standard_deviations
        task_stats = means.to_frame("mean").join(
            standard_deviations.to_frame("std")
        )
        task_stats["task"] = task
        stats.append(task_stats.reset_index(names="feature"))
    return frame, pd.concat(stats, ignore_index=True)


def add_hourly_calendar_encodings(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["hour"] = frame["time"].dt.hour
    frame["dow"] = frame["time"].dt.dayofweek
    frame["month"] = frame["time"].dt.month
    frame["is_weekend"] = frame["dow"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)
    return frame


def add_daily_calendar_encodings(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["day"] = frame["time"].dt.day
    frame["dow"] = frame["time"].dt.dayofweek
    frame["month"] = frame["time"].dt.month
    frame["year"] = frame["time"].dt.year
    frame["is_weekend"] = frame["dow"].isin([5, 6]).astype(int)
    frame["is_month_start"] = frame["time"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["time"].dt.is_month_end.astype(int)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dow"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dow"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)
    return frame


def safe_task_name(value: object) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def select_tasks(values: Iterable[str], max_tasks: int | None, seed: int) -> list[str]:
    tasks = list(dict.fromkeys(str(value) for value in values))
    if max_tasks is None or max_tasks >= len(tasks):
        return tasks
    if max_tasks < 1:
        raise ValueError("max_tasks must be positive.")
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(tasks), size=max_tasks, replace=False)
    return [tasks[index] for index in indices]


def write_outputs(
    combined: pd.DataFrame,
    metadata: pd.DataFrame,
    scale_stats: pd.DataFrame,
    output_dir: Path,
    dataset: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task = output_dir / "by_task"
    by_task.mkdir(exist_ok=True)
    paths = {
        "full": output_dir / f"{dataset}_full.csv",
        "train": output_dir / f"{dataset}_train.csv",
        "validation": output_dir / f"{dataset}_val.csv",
        "test": output_dir / f"{dataset}_test.csv",
        "meta": output_dir / f"{dataset}_meta.csv",
        "scaler": output_dir / f"{dataset}_scaler_stats.csv",
    }
    combined.to_csv(paths["full"], index=False)
    combined.loc[combined["split"] == "train"].to_csv(paths["train"], index=False)
    combined.loc[combined["split"] == "val"].to_csv(paths["validation"], index=False)
    combined.loc[combined["split"] == "test"].to_csv(paths["test"], index=False)
    metadata.to_csv(paths["meta"], index=False)
    scale_stats.to_csv(paths["scaler"], index=False)
    for task, group in combined.groupby("task"):
        group.to_csv(by_task / f"{dataset}_{safe_task_name(task)}.csv", index=False)
    return paths


def write_manifest(
    *,
    dataset: str,
    source_url: str,
    raw_file: Path,
    output_dir: Path,
    combined: pd.DataFrame,
    parameters: dict[str, Any],
) -> Path:
    split_rows = {
        split: int((combined["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    manifest = DatasetManifest(
        dataset=dataset,
        source_url=source_url,
        raw_file=str(raw_file.resolve()),
        raw_sha256=sha256_file(raw_file),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        output_directory=str(output_dir.resolve()),
        n_tasks=int(combined["task"].nunique()),
        n_rows=int(len(combined)),
        split_rows=split_rows,
        parameters=parameters,
    )
    path = output_dir / f"{dataset}_preprocessing_manifest.json"
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return path


def prepare_exchange_from_zip(
    zip_path: Path,
    output_dir: Path,
    *,
    currencies: Sequence[str] = DEFAULT_CURRENCIES,
) -> dict[str, Path]:
    """Prepare the ECB exchange-rate panel from a local source ZIP."""

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise RuntimeError("The ECB archive contains no CSV file.")
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(stream)
    frame = frame.rename(columns={frame.columns[0]: "time"})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time")
    selected = [currency.upper() for currency in currencies if currency.upper() in frame]
    if not selected:
        raise RuntimeError("None of the requested currencies exists in the ECB file.")
    frame = frame[["time", *selected]]
    long = frame.melt(id_vars="time", var_name="task", value_name="y").dropna(subset=["y"])
    long["day"] = long["time"].dt.day
    long["month"] = long["time"].dt.month
    long["year"] = long["time"].dt.year
    long["is_month_start"] = long["time"].dt.is_month_start.astype(int)
    long["is_month_end"] = long["time"].dt.is_month_end.astype(int)
    long = long.sort_values(["task", "time"]).reset_index(drop=True)
    long["log_y"] = np.log(long["y"].astype(float).clip(lower=1e-12))
    long["ret"] = long.groupby("task")["log_y"].transform(lambda series: series.diff())
    long["vol_5"] = long.groupby("task")["ret"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=5).std()
    )
    long["vol_21"] = long.groupby("task")["ret"].transform(
        lambda series: series.shift(1).rolling(21, min_periods=21).std()
    )
    long = long.drop(columns="log_y")
    covariates = [column for column in long if column not in {"time", "task", "y", "split"}]
    supervised = make_supervised_panel(
        long,
        covariates=covariates,
        horizon=1,
        y_lags=(1, 5, 21),
        cov_lags=(1, 5),
        roll_windows=(5, 21),
    )
    combined, metadata = temporal_split_per_task(supervised)
    scaled, stats = standardize_per_task(combined)
    return write_outputs(scaled, metadata, stats, output_dir, "exchange")


def resolve_metrla_download_url() -> str:
    record = download_json(METRLA_RECORD_API)
    for entry in record.get("files", []):
        name = str(entry.get("key") or entry.get("filename") or "")
        if name.lower() != "metr-la.csv":
            continue
        links = entry.get("links", {})
        for key in ("content", "download", "self"):
            if links.get(key):
                return str(links[key])
    raise RuntimeError("METR-LA.csv was not found in Zenodo record 5146275.")


def prepare_metrla_from_csv(
    csv_path: Path,
    output_dir: Path,
    *,
    max_tasks: int | None = None,
    task_seed: int = 42,
    adjacency_path: Path | None = None,
) -> dict[str, Path]:
    """Prepare METR-LA from the wide five-minute sensor CSV."""

    wide = pd.read_csv(csv_path, low_memory=False)
    if wide.empty:
        raise RuntimeError(f"Empty METR-LA CSV: {csv_path}")
    if str(wide.columns[0]).lower() != "time":
        wide = wide.rename(columns={wide.columns[0]: "time"})
    wide["time"] = pd.to_datetime(wide["time"], errors="coerce")
    wide = wide.dropna(subset=["time"]).sort_values("time")
    value_columns = [str(column) for column in wide.columns if column != "time"]
    value_columns = select_tasks(value_columns, max_tasks, task_seed)
    long = wide.melt(
        id_vars="time", value_vars=value_columns, var_name="task", value_name="y"
    ).dropna(subset=["y"])
    long = add_hourly_calendar_encodings(long)
    long = long.sort_values(["task", "time"])
    long["delta_speed"] = long.groupby("task")["y"].transform(lambda series: series.diff())

    if adjacency_path is not None and adjacency_path.exists():
        try:
            with adjacency_path.open("rb") as stream:
                adjacency_object = pickle.load(stream)
            adjacency = (
                adjacency_object[2]
                if isinstance(adjacency_object, tuple) and len(adjacency_object) >= 3
                else adjacency_object
            )
            degree = np.asarray(adjacency).sum(axis=1).ravel()
            unique_tasks = sorted(long["task"].astype(str).unique())
            if len(unique_tasks) == len(degree):
                long["degree"] = long["task"].map(
                    {task: float(degree[index]) for index, task in enumerate(unique_tasks)}
                )
        except (OSError, pickle.PickleError, ValueError, TypeError):
            pass

    covariates = [column for column in long if column not in {"time", "task", "y", "split"}]
    supervised = make_supervised_panel(
        long,
        covariates=covariates,
        horizon=12,
        y_lags=(1, 12, 288),
        cov_lags=(1, 12),
        roll_windows=(12, 288),
    )
    combined, metadata = temporal_split_per_task(supervised)
    scaled, stats = standardize_per_task(combined)
    return write_outputs(scaled, metadata, stats, output_dir, "metrla")


def parse_tsf_file(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse a Monash ``.ts``/``.tsf`` forecasting file into a long panel."""

    metadata: dict[str, str] = {}
    attribute_names: list[str] = []
    rows: list[dict[str, Any]] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if lower.startswith("@data"):
                in_data = True
                continue
            if not in_data:
                if lower.startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attribute_names.append(parts[1])
                elif lower.startswith("@"):
                    parts = line.split(maxsplit=1)
                    metadata[parts[0][1:].lower()] = parts[1] if len(parts) > 1 else ""
                continue
            parts = line.split(":")
            if len(parts) < len(attribute_names) + 1:
                continue
            attributes = parts[: len(attribute_names)]
            values_text = ":".join(parts[len(attribute_names) :])
            values = [
                np.nan if value.strip() in {"?", "NaN", "nan", ""} else float(value)
                for value in values_text.split(",")
            ]
            row = dict(zip(attribute_names, attributes))
            row["series_value"] = values
            rows.append(row)

    frequency = metadata.get("frequency", "").strip().lower()
    pandas_frequency = {"daily": "D", "weekly": "W", "monthly": "MS", "hourly": "h"}.get(
        frequency, "D"
    )
    records: list[pd.DataFrame] = []
    for index, row in enumerate(rows):
        values = pd.Series(row["series_value"], dtype="float32")
        task = (
            row.get("series_name")
            or row.get("series_id")
            or row.get("item_id")
            or f"series_{index:03d}"
        )
        start_raw = row.get("start_timestamp") or row.get("start")
        start = pd.to_datetime(start_raw, errors="coerce") if start_raw else pd.Timestamp("2000-01-01")
        if pd.isna(start):
            start = pd.Timestamp("2000-01-01")
        time = pd.date_range(start=start, periods=len(values), freq=pandas_frequency)
        records.append(pd.DataFrame({"time": time, "task": str(task), "y": values}))
    if not records:
        raise RuntimeError(f"No series found in TSF file: {path}")
    return pd.concat(records, ignore_index=True), metadata


def extract_nn5_tsf(zip_path: Path, raw_directory: Path, *, force: bool = False) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name for name in archive.namelist() if name.lower().endswith((".ts", ".tsf"))
        ]
        if not members:
            raise RuntimeError("The NN5 archive contains no .ts or .tsf file.")
        output = raw_directory / Path(members[0]).name
        if force or not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[0]) as source, output.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    return output


def prepare_nn5_from_tsf(
    tsf_path: Path,
    output_dir: Path,
    *,
    max_tasks: int | None = None,
    task_seed: int = 42,
) -> dict[str, Path]:
    """Prepare the Monash NN5 Daily ATM-withdrawal dataset."""

    long, _ = parse_tsf_file(tsf_path)
    long = long.dropna(subset=["y"]).sort_values(["task", "time"]).reset_index(drop=True)
    selected = select_tasks(long["task"].astype(str), max_tasks, task_seed)
    long = long.loc[long["task"].astype(str).isin(selected)].copy()
    long = add_daily_calendar_encodings(long)
    covariates = [column for column in long if column not in {"time", "task", "y", "split"}]
    supervised = make_supervised_panel(
        long,
        covariates=covariates,
        horizon=1,
        y_lags=(1, 7, 14, 28),
        cov_lags=(1, 7),
        roll_windows=(7, 28),
    )
    combined, metadata = temporal_split_per_task(supervised)
    scaled, stats = standardize_per_task(combined)
    return write_outputs(scaled, metadata, stats, output_dir, "nn5")


def prepare_dataset(
    dataset: str,
    *,
    data_root: Path = Path("data"),
    output_dir: Path | None = None,
    force: bool = False,
    currencies: Sequence[str] = DEFAULT_CURRENCIES,
    max_tasks: int | None = None,
    task_seed: int = 42,
) -> dict[str, Path]:
    """Download and prepare one supported real dataset."""

    dataset = dataset.lower()
    if dataset not in {"exchange", "metrla", "nn5"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    data_root = data_root.expanduser().resolve()
    raw_directory = data_root / "raw" / dataset
    processed_directory = output_dir or (data_root / "interim" / dataset)
    processed_directory = processed_directory.expanduser().resolve()
    raw_directory.mkdir(parents=True, exist_ok=True)
    if force and processed_directory.exists():
        shutil.rmtree(processed_directory)

    if dataset == "exchange":
        raw_file = download_file(
            EXCHANGE_URL, raw_directory / "eurofxref-hist.zip", force=force
        )
        paths = prepare_exchange_from_zip(
            raw_file, processed_directory, currencies=currencies
        )
        parameters = {"currencies": list(currencies)}
        source_url = EXCHANGE_URL
    elif dataset == "metrla":
        raw_file = raw_directory / "METR-LA.csv"
        if force or not raw_file.exists():
            download_file(resolve_metrla_download_url(), raw_file, force=True)
        paths = prepare_metrla_from_csv(
            raw_file,
            processed_directory,
            max_tasks=max_tasks,
            task_seed=task_seed,
            adjacency_path=raw_directory / "adj_mx.pkl",
        )
        parameters = {"max_tasks": max_tasks, "task_seed": task_seed, "horizon": 12}
        source_url = METRLA_RECORD_API
    else:
        raw_zip = download_file(
            NN5_URL,
            raw_directory / "nn5_daily_dataset_without_missing_values.zip",
            force=force,
        )
        raw_file = extract_nn5_tsf(raw_zip, raw_directory, force=force)
        paths = prepare_nn5_from_tsf(
            raw_file,
            processed_directory,
            max_tasks=max_tasks,
            task_seed=task_seed,
        )
        parameters = {"max_tasks": max_tasks, "task_seed": task_seed, "horizon": 1}
        source_url = NN5_URL

    combined = pd.read_csv(paths["full"], usecols=["task", "split"])
    write_manifest(
        dataset=dataset,
        source_url=source_url,
        raw_file=raw_file,
        output_dir=processed_directory,
        combined=combined,
        parameters=parameters,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and preprocess the Exchange, METR-LA, and NN5 datasets "
            "used by the SoftwareX illustrative benchmark."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("exchange", "metrla", "nn5", "all"),
        help="Dataset to prepare; 'all' prepares all three sequentially.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing raw/ and interim/ directories (default: data).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Custom processed output directory; valid only for one dataset.",
    )
    parser.add_argument(
        "--currencies",
        default=",".join(DEFAULT_CURRENCIES),
        help="Comma-separated ECB currencies used for Exchange.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Diagnostic task cap for METR-LA/NN5; omit for paper reproduction.",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=42,
        help="Seed used only when --max-tasks selects a diagnostic subset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload cached raw files and overwrite processed outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ("exchange", "metrla", "nn5") if args.dataset == "all" else (args.dataset,)
    if args.outdir is not None and len(datasets) != 1:
        raise SystemExit("--outdir can only be used with one dataset.")
    currencies = tuple(
        value.strip().upper() for value in args.currencies.split(",") if value.strip()
    )
    for dataset in datasets:
        print(f"Preparing {dataset}...")
        paths = prepare_dataset(
            dataset,
            data_root=args.data_root,
            output_dir=args.outdir,
            force=args.force,
            currencies=currencies,
            max_tasks=args.max_tasks,
            task_seed=args.task_seed,
        )
        print(f"Prepared {dataset}: {paths['full']}")


if __name__ == "__main__":
    main()
