from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms


class MultitaskDataset(Dataset):
    """
    In-memory dataframe dataset and base class for balanced multi-task datasets.

    Each source row belongs to one task. A dataset item contains one sample from
    every task, cycling shorter tasks with modulo indexing. Subclasses only need
    to implement storage-specific sample loading; task metadata, balancing, and
    stacking remain centralized here.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            task_id: str,
            target_names: Sequence[str] | str,
            device: str = "cpu",
            dtype: torch.dtype = torch.float32,
            y_dtype: torch.dtype | None = None,
            presplit_X: bool = True,
            idx_column: Optional[str] = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("No tasks found")

        self._initialize_common(
            task_id=task_id,
            target_names=target_names,
            device=device,
            dtype=dtype,
            y_dtype=y_dtype,
            presplit_X=presplit_X,
            idx_column=idx_column,
        )
        self._set_schema(data.columns, require_features=presplit_X)
        self._validate_dataframe(data, validate_features=presplit_X)

        normalized_task_ids = self._fit_task_metadata(
            data[self.task_id].to_numpy(),
        )
        self._build_dataframe_storage(data, normalized_task_ids)

    def _initialize_common(
            self,
            task_id: str,
            target_names: Sequence[str] | str,
            device: str,
            dtype: torch.dtype,
            y_dtype: torch.dtype | None,
            presplit_X: bool,
            idx_column: Optional[str],
    ) -> None:
        if not isinstance(task_id, str):
            raise TypeError("task_id must be a str column name")
        if idx_column is not None and not isinstance(idx_column, str):
            raise TypeError("idx_column must be a str column name or None")

        self.task_id = task_id
        self.target_names = self._normalize_target_names(target_names)
        self.device = torch.device(device)
        self.dtype = dtype
        self.y_dtype = dtype if y_dtype is None else y_dtype
        self.presplit_X = presplit_X
        self.idx_column = idx_column

        self.feature_names_: List[str] = []
        self.X: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []
        self.idx_: List[np.ndarray] = []

    @staticmethod
    def _normalize_target_names(target_names: Sequence[str] | str) -> List[str]:
        if isinstance(target_names, str):
            targets = [target_names]
        else:
            targets = list(target_names)
        if not targets:
            raise ValueError("target_names must be a non-empty sequence.")
        if any(not isinstance(target, str) for target in targets):
            raise TypeError("target_names must contain only str column names")
        return targets

    def _set_schema(
            self,
            columns: Sequence[str],
            *,
            require_features: bool,
    ) -> None:
        columns = list(columns)
        if self.task_id not in columns:
            raise ValueError(f"task_id='{self.task_id}' not found in columns.")

        missing_targets = [
            target for target in self.target_names
            if target not in columns
        ]
        if missing_targets:
            raise ValueError(f"Targets not found in columns: {missing_targets}")
        if self.idx_column is not None and self.idx_column not in columns:
            raise ValueError(f"idx_column '{self.idx_column}' not found in columns.")

        excluded = set(self.target_names) | {self.task_id}
        if self.idx_column is not None:
            excluded.add(self.idx_column)
        self.feature_names_ = [
            column for column in columns
            if column not in excluded
        ]
        if require_features and not self.feature_names_:
            raise ValueError(
                "No feature columns found after removing targets and task_id."
            )

    def _validate_dataframe(
            self,
            data: pd.DataFrame,
            *,
            validate_features: bool,
    ) -> None:
        if data[self.task_id].isna().any():
            raise ValueError(
                f"NaN values found in task_id column '{self.task_id}'."
            )

        feature_columns = self.feature_names_ if validate_features else []
        bad_features = [
            column for column in feature_columns
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if bad_features:
            raise TypeError(
                f"Non-numeric feature columns: {bad_features}. "
                "Encode/cast before using the dataset."
            )

        bad_targets = [
            column for column in self.target_names
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if bad_targets:
            raise TypeError(
                f"Non-numeric target columns: {bad_targets}. "
                "Encode/cast before using the dataset."
            )

        consumed_columns = feature_columns + self.target_names
        if data[consumed_columns].isna().any().any():
            raise ValueError(
                "NaNs found in features/targets. "
                "Impute or drop before creating the dataset."
            )

    def _fit_task_metadata(
            self,
            task_values: Sequence[Any],
            task_counts: Optional[Dict[Any, int]] = None,
    ) -> Optional[np.ndarray]:
        values = np.asarray(task_values)
        if values.size == 0:
            raise ValueError("No tasks found")

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(values)
        original_ids = self.label_encoder_.classes_.tolist()
        self.tasks_ = np.arange(len(original_ids), dtype=np.int64)
        self.n_tasks = len(original_ids)

        normalized_task_ids: Optional[np.ndarray]
        if task_counts is None:
            normalized_task_ids = self.label_encoder_.transform(values)
            self.task_counts_ = np.bincount(
                normalized_task_ids,
                minlength=self.n_tasks,
            ).astype(np.int64)
        else:
            normalized_task_ids = None
            self.task_counts_ = np.asarray(
                [task_counts[task] for task in original_ids],
                dtype=np.int64,
            )

        normalized_ids = self.tasks_.tolist()
        self.task_id_map_: Dict[Any, int] = dict(
            zip(original_ids, normalized_ids)
        )
        self.task_id_inverse_map_: Dict[int, Any] = dict(
            zip(normalized_ids, original_ids)
        )
        self._task_starts = np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum(self.task_counts_[:-1], dtype=np.int64),
        ))
        return normalized_task_ids

    def _build_dataframe_storage(
            self,
            data: pd.DataFrame,
            normalized_task_ids: np.ndarray,
    ) -> None:
        for task in self.tasks_:
            task_rows = np.flatnonzero(normalized_task_ids == task)
            task_data = data.iloc[task_rows]
            if self.presplit_X:
                self.X.append(torch.as_tensor(
                    task_data[self.feature_names_].to_numpy(copy=True),
                    dtype=self.dtype,
                    device=self.device,
                ))

            self.Y.append(torch.as_tensor(
                task_data[self.target_names].to_numpy(copy=True),
                dtype=self.y_dtype,
                device=self.device,
            ))

            if self.idx_column is not None:
                self.idx_.append(task_data[self.idx_column].to_numpy())

    def _balanced_local_indices(self, idx: int) -> np.ndarray:
        return np.remainder(idx, self.task_counts_)

    def _load_task_sample(
            self,
            task: int,
            local_idx: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        features = self.X[task][local_idx] if self.X else None
        return features, self.Y[task][local_idx]

    def _stack_features(
            self,
            features: Sequence[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if all(feature is None for feature in features):
            return None
        if any(feature is None for feature in features):
            raise RuntimeError(
                "Every task must return features, or every task must return None."
            )
        try:
            return torch.stack(list(features))  # type: ignore[arg-type]
        except RuntimeError as exc:
            raise ValueError(
                "All task features must have the same shape to stack."
            ) from exc

    def _load_balanced_batch(
            self,
            local_indices: np.ndarray,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        samples = [
            self._load_task_sample(task, int(local_idx))
            for task, local_idx in enumerate(local_indices)
        ]
        features, targets = zip(*samples)
        return self._stack_features(features), torch.stack(list(targets))

    def __getitem__(
            self,
            idx: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        return self._load_balanced_batch(self._balanced_local_indices(idx))

    def __len__(self) -> int:
        return int(self.task_counts_.max())

    def get_original_task_id(
            self,
            normalized_ids: Sequence[int],
    ) -> List[object]:
        return self.label_encoder_.inverse_transform(
            np.asarray(normalized_ids)
        ).tolist()

    def get_task_counts_by_original(self) -> Dict[Any, int]:
        return {
            self.task_id_inverse_map_[task]: int(self.task_counts_[task])
            for task in self.tasks_
        }

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_tasks={self.n_tasks}, "
            f"n_features={self.n_features}, n_targets={self.n_targets}, "
            f"len={len(self)})"
        )


class MultitaskDatasetCsv(MultitaskDataset):
    """
    Disk-backed CSV variant of :class:`MultitaskDataset`.

    The CSV is scanned in bounded chunks to create task-grouped NumPy memory
    maps. Random samples are then read from disk on demand without retaining the
    source dataframe or per-task tensors in memory.
    """

    def __init__(
            self,
            data_path: str,
            task_id: str,
            target_names: Sequence[str] | str | None = None,
            *,
            target_name: Sequence[str] | str | None = None,
            chunksize: int = 100_000,
            cache_dir: Optional[str] = None,
            csv_kwargs: Optional[Dict[str, Any]] = None,
            device: str = "cpu",
            dtype: torch.dtype = torch.float32,
            y_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(data_path, (str, os.PathLike)):
            raise TypeError("data_path must be a path-like object")
        self.data_path = os.fspath(data_path)
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"CSV file not found: {self.data_path}")
        if not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer")
        if target_names is not None and target_name is not None:
            raise ValueError("Pass either target_names or target_name, not both.")

        targets = target_names if target_names is not None else target_name
        self._initialize_common(
            task_id=task_id,
            target_names=[] if targets is None else targets,
            device=device,
            dtype=dtype,
            y_dtype=y_dtype,
            presplit_X=False,
            idx_column=None,
        )

        self.chunksize = chunksize
        self.csv_kwargs = dict(csv_kwargs or {})
        for managed_arg in ("chunksize", "iterator", "nrows"):
            if managed_arg in self.csv_kwargs:
                raise ValueError(
                    f"csv_kwargs must not contain '{managed_arg}'; "
                    "it is managed by the dataset."
                )

        self._x_np_dtype = self._numpy_dtype(self.dtype, "dtype")
        self._y_np_dtype = self._numpy_dtype(self.y_dtype, "y_dtype")
        self._owner_pid = os.getpid()
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = tempfile.mkdtemp(
            prefix="alemtl-csv-",
            dir=cache_dir,
        )
        self._x_path = os.path.join(self.cache_dir, "features.dat")
        self._y_path = os.path.join(self.cache_dir, "targets.dat")
        self._X_memmap: Optional[np.memmap] = None
        self._Y_memmap: Optional[np.memmap] = None

        try:
            self._build_cache()
            self._open_memmaps()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _numpy_dtype(dtype: torch.dtype, argument_name: str) -> np.dtype:
        try:
            return torch.empty((), dtype=dtype).numpy().dtype
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                f"{argument_name}={dtype} cannot be stored in a NumPy memory map."
            ) from exc

    def _read_chunks(self):
        return pd.read_csv(
            self.data_path,
            chunksize=self.chunksize,
            **self.csv_kwargs,
        )

    def _build_cache(self) -> None:
        try:
            columns = pd.read_csv(
                self.data_path,
                nrows=0,
                **self.csv_kwargs,
            ).columns
        except pd.errors.EmptyDataError as exc:
            raise ValueError("No tasks found") from exc

        self._set_schema(columns, require_features=True)
        task_counts: Dict[Any, int] = {}
        task_values: List[Any] = []
        seen_tasks = set()
        row_count = 0

        for chunk in self._read_chunks():
            self._validate_dataframe(chunk, validate_features=True)
            for task, count in chunk[self.task_id].value_counts(
                    sort=False,
            ).items():
                if task not in seen_tasks:
                    seen_tasks.add(task)
                    task_values.append(task)
                task_counts[task] = task_counts.get(task, 0) + int(count)
            row_count += len(chunk)

        if row_count == 0:
            raise ValueError("No tasks found")

        self._fit_task_metadata(task_values, task_counts)
        self._write_cache(row_count)

    def _write_cache(self, row_count: int) -> None:
        x_memmap = np.memmap(
            self._x_path,
            mode="w+",
            dtype=self._x_np_dtype,
            shape=(row_count, self.n_features),
        )
        y_memmap = np.memmap(
            self._y_path,
            mode="w+",
            dtype=self._y_np_dtype,
            shape=(row_count, self.n_targets),
        )
        cursors = self._task_starts.copy()

        try:
            for chunk in self._read_chunks():
                normalized_ids = self.label_encoder_.transform(
                    chunk[self.task_id].to_numpy()
                )
                for task in np.unique(normalized_ids):
                    mask = normalized_ids == task
                    count = int(mask.sum())
                    start = int(cursors[task])
                    stop = start + count
                    x_memmap[start:stop] = chunk.loc[
                        mask,
                        self.feature_names_,
                    ].to_numpy(dtype=self._x_np_dtype)
                    y_memmap[start:stop] = chunk.loc[
                        mask,
                        self.target_names,
                    ].to_numpy(dtype=self._y_np_dtype)
                    cursors[task] = stop

            if not np.array_equal(
                    cursors,
                    self._task_starts + self.task_counts_,
            ):
                raise RuntimeError(
                    "The CSV changed while the dataset cache was being created."
                )
            x_memmap.flush()
            y_memmap.flush()
        finally:
            del x_memmap
            del y_memmap

    def _open_memmaps(self) -> None:
        row_count = int(self.task_counts_.sum())
        self._X_memmap = np.memmap(
            self._x_path,
            mode="r",
            dtype=self._x_np_dtype,
            shape=(row_count, self.n_features),
        )
        self._Y_memmap = np.memmap(
            self._y_path,
            mode="r",
            dtype=self._y_np_dtype,
            shape=(row_count, self.n_targets),
        )

    def _load_balanced_batch(
            self,
            local_indices: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._X_memmap is None or self._Y_memmap is None:
            self._open_memmaps()

        row_indices = self._task_starts + local_indices
        features = np.array(self._X_memmap[row_indices], copy=True)
        targets = np.array(self._Y_memmap[row_indices], copy=True)
        return (
            torch.as_tensor(
                features,
                dtype=self.dtype,
                device=self.device,
            ),
            torch.as_tensor(
                targets,
                dtype=self.y_dtype,
                device=self.device,
            ),
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_X_memmap"] = None
        state["_Y_memmap"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_memmaps()

    def close(self) -> None:
        self._X_memmap = None
        self._Y_memmap = None
        cache_dir = getattr(self, "cache_dir", None)
        if (
                cache_dir
                and os.getpid() == getattr(self, "_owner_pid", None)
                and os.path.isdir(cache_dir)
        ):
            shutil.rmtree(cache_dir)

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


class MultitaskDatasetImg(MultitaskDataset):
    """Image-backed variant that loads feature tensors on demand."""

    def __init__(
            self,
            img_dir: str,
            img_data: pd.DataFrame,
            col_img_file: str = "img_file",
            col_task_id: str = "task_id",
            transform=None,
            *args,
            **kwargs,
    ) -> None:
        target_names = [
            column for column in img_data.columns
            if column not in {col_img_file, col_task_id}
        ]
        super().__init__(
            data=img_data,
            task_id=col_task_id,
            target_names=target_names,
            presplit_X=False,
            idx_column=col_img_file,
            *args,
            **kwargs,
        )
        self.img_dir = os.fspath(img_dir)
        self.transform = transform

    def _load_image(self, img_file: str) -> torch.Tensor:
        img_path = os.path.join(self.img_dir, img_file)
        with Image.open(img_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        return image

    def _load_task_sample(
            self,
            task: int,
            local_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(self.idx_[task][local_idx])
        return image, self.Y[task][local_idx]
