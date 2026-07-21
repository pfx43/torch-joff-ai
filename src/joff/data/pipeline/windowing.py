"""Dynamic window datasets for time-series experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowSample:
    """One resolved dynamic window sample in numpy form."""

    history: np.ndarray
    future: np.ndarray
    u_history: np.ndarray
    y_history: np.ndarray
    u_future: np.ndarray
    y_future: np.ndarray
    index: int


@dataclass(frozen=True)
class SequenceSample:
    """One continuous RNN sequence sample in numpy form."""

    x: np.ndarray
    target: np.ndarray
    index: int
    target_index: int


@dataclass(frozen=True)
class MPCWindowSample:
    """One resolved MPC horizon sample in numpy form."""

    past: np.ndarray
    target_future: np.ndarray
    control_future: np.ndarray
    reference_future: np.ndarray
    state_past: np.ndarray
    output_past: np.ndarray
    control_past: np.ndarray
    reference_past: np.ndarray
    index: int


class DynamicWindowDataset(Dataset):
    """Build flattened dynamic samples from aligned ``u`` and ``y`` arrays."""

    def __init__(
        self,
        u_data: np.ndarray,
        y_data: np.ndarray,
        *,
        lookback: int = 3,
        future_steps: int = 1,
        indices: np.ndarray | None = None,
        segment_ids: np.ndarray | None = None,
        return_mode: str = "dict",
    ) -> None:
        self.u_data = _as_2d(u_data)
        self.y_data = _as_2d(y_data)
        if self.u_data.shape[0] != self.y_data.shape[0]:
            raise ValueError(
                f"u_data and y_data must share row count. Current rows: "
                f"{self.u_data.shape[0]} and {self.y_data.shape[0]}."
            )
        if lookback <= 0 or future_steps <= 0:
            raise ValueError(
                f"lookback and future_steps must be positive. Current inputs: "
                f"lookback={lookback}, future_steps={future_steps}."
            )
        self.lookback = lookback
        self.future_steps = future_steps
        self.return_mode = return_mode
        self.segment_ids = np.zeros(self.u_data.shape[0], dtype=int) if segment_ids is None else np.asarray(segment_ids)
        if self.segment_ids.shape[0] != self.u_data.shape[0]:
            raise ValueError("segment_ids must have the same row count as u_data/y_data.")
        self.sample_starts = self._build_sample_starts(indices)

    def __len__(self) -> int:
        """Return the number of valid dynamic samples."""

        return int(self.sample_starts.shape[0])

    def __getitem__(self, idx: int):
        """Return one sample as a dict of tensors or a tuple for legacy code."""

        sample = self.sample_numpy(idx)
        if self.return_mode == "tuple":
            return (
                torch.as_tensor(sample.history, dtype=torch.float32),
                torch.as_tensor(sample.future, dtype=torch.float32),
            )
        if self.return_mode != "dict":
            raise ValueError("return_mode must be one of: dict, tuple.")
        return {
            "history": torch.as_tensor(sample.history, dtype=torch.float32),
            "future": torch.as_tensor(sample.future, dtype=torch.float32),
            "u_history": torch.as_tensor(sample.u_history, dtype=torch.float32),
            "y_history": torch.as_tensor(sample.y_history, dtype=torch.float32),
            "u_future": torch.as_tensor(sample.u_future, dtype=torch.float32),
            "y_future": torch.as_tensor(sample.y_future, dtype=torch.float32),
            "index": torch.as_tensor(sample.index, dtype=torch.long),
        }

    def sample_numpy(self, idx: int) -> WindowSample:
        """Return one sample as numpy arrays."""

        start = int(self.sample_starts[idx])
        hist = slice(start, start + self.lookback)
        fut = slice(start + self.lookback, start + self.lookback + self.future_steps)
        u_history = self.u_data[hist]
        y_history = self.y_data[hist]
        u_future = self.u_data[fut]
        y_future = self.y_data[fut]
        history = np.concatenate([u_history, y_history], axis=1).reshape(-1)
        future = np.concatenate([u_future, y_future], axis=1).reshape(-1)
        return WindowSample(
            history=history,
            future=future,
            u_history=u_history,
            y_history=y_history,
            u_future=u_future,
            y_future=y_future,
            index=start,
        )

    def target_values(self) -> np.ndarray:
        """Return flattened future target values for all samples."""

        values = [self.sample_numpy(i).y_future.reshape(-1) for i in range(len(self))]
        if not values:
            return np.empty((0, self.y_data.shape[1] * self.future_steps), dtype=float)
        return np.vstack(values)

    def subset(self, sample_indices: np.ndarray) -> "DynamicWindowSubset":
        """Return a lightweight subset by sample indices."""

        return DynamicWindowSubset(self, np.asarray(sample_indices, dtype=int))

    def _build_sample_starts(self, indices: np.ndarray | None) -> np.ndarray:
        if indices is None:
            candidates = np.arange(0, self.u_data.shape[0] - self.lookback - self.future_steps + 1)
        else:
            candidates = np.asarray(indices, dtype=int)
        valid: list[int] = []
        for start in candidates:
            end = start + self.lookback + self.future_steps
            if start < 0 or end > self.u_data.shape[0]:
                continue
            if np.unique(self.segment_ids[start:end]).shape[0] == 1:
                valid.append(int(start))
        return np.asarray(valid, dtype=int)


class SequenceDataset(Dataset):
    """Build continuous sequence samples for RNN/GRU/LSTM models."""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        *,
        input_length: int = 3,
        target_length: int | None = None,
        task: str = "n_to_1",
        target_offset: int = 0,
        stride: int = 1,
        indices: np.ndarray | None = None,
        segment_ids: np.ndarray | None = None,
        return_mode: str = "dict",
        squeeze_single_target: bool = True,
    ) -> None:
        self.x_data = _as_2d(x_data)
        self.y_data = _as_2d(y_data)
        if self.x_data.shape[0] != self.y_data.shape[0]:
            raise ValueError(
                f"x_data and y_data must share row count. Current rows: "
                f"{self.x_data.shape[0]} and {self.y_data.shape[0]}."
            )
        if input_length <= 0:
            raise ValueError(f"input_length must be positive. Current input: {input_length}.")
        if target_offset < 0:
            raise ValueError(f"target_offset must be non-negative. Current input: {target_offset}.")
        if stride <= 0:
            raise ValueError(f"stride must be positive. Current input: {stride}.")
        self.task = _resolve_sequence_task(task)
        self.input_length = int(input_length)
        self.target_length = int(
            target_length if target_length is not None else self.input_length if self.task == "n_to_n" else 1
        )
        if self.target_length <= 0:
            raise ValueError(f"target_length must be positive. Current input: {self.target_length}.")
        self.target_offset = int(target_offset)
        self.stride = int(stride)
        self.return_mode = return_mode
        self.squeeze_single_target = bool(squeeze_single_target)
        self.segment_ids = np.zeros(self.x_data.shape[0], dtype=int) if segment_ids is None else np.asarray(segment_ids)
        if self.segment_ids.shape[0] != self.x_data.shape[0]:
            raise ValueError("segment_ids must have the same row count as x_data/y_data.")
        self.sample_starts = self._build_sample_starts(indices)

    def __len__(self) -> int:
        """Return the number of valid sequence samples."""

        return int(self.sample_starts.shape[0])

    def __getitem__(self, idx: int):
        """Return one sequence sample as a dict of tensors or a tuple."""

        sample = self.sample_numpy(idx)
        target = sample.target
        if self.squeeze_single_target and target.shape[0] == 1:
            target = target[0]
        x_tensor = torch.as_tensor(sample.x, dtype=torch.float32)
        target_tensor = torch.as_tensor(target, dtype=torch.float32)
        if self.return_mode == "tuple":
            return x_tensor, target_tensor
        if self.return_mode != "dict":
            raise ValueError("return_mode must be one of: dict, tuple.")
        return {
            "x": x_tensor,
            "target": target_tensor,
            "index": torch.as_tensor(sample.index, dtype=torch.long),
            "target_index": torch.as_tensor(sample.target_index, dtype=torch.long),
        }

    def sample_numpy(self, idx: int) -> SequenceSample:
        """Return one sample as numpy arrays."""

        start = int(self.sample_starts[idx])
        target_start = self._target_start(start)
        x_rows = slice(start, start + self.input_length)
        y_rows = slice(target_start, target_start + self.target_length)
        return SequenceSample(
            x=self.x_data[x_rows],
            target=self.y_data[y_rows],
            index=start,
            target_index=target_start,
        )

    def target_values(self) -> np.ndarray:
        """Return flattened target values for all sequence samples."""

        values = [self.sample_numpy(i).target.reshape(-1) for i in range(len(self))]
        if not values:
            return np.empty((0, self.y_data.shape[1] * self.target_length), dtype=float)
        return np.vstack(values)

    def summary(self) -> dict[str, int | str]:
        """Return a serializable sequence construction summary."""

        return {
            "task": self.task,
            "input_length": self.input_length,
            "target_length": self.target_length,
            "target_offset": self.target_offset,
            "stride": self.stride,
            "samples": len(self),
            "input_dim": int(self.x_data.shape[1]),
            "target_dim": int(self.y_data.shape[1]),
        }

    def _target_start(self, start: int) -> int:
        if self.task == "n_to_n":
            return start + self.target_offset
        return start + self.input_length + self.target_offset

    def _build_sample_starts(self, indices: np.ndarray | None) -> np.ndarray:
        if indices is None:
            candidates = np.arange(0, self.x_data.shape[0], self.stride)
        else:
            candidates = np.asarray(indices, dtype=int)
        valid: list[int] = []
        for start in candidates:
            target_start = self._target_start(int(start))
            x_start = int(start)
            x_end = x_start + self.input_length
            y_end = target_start + self.target_length
            span_start = min(x_start, target_start)
            span_end = max(x_end, y_end)
            if span_start < 0 or span_end > self.x_data.shape[0]:
                continue
            if np.unique(self.segment_ids[span_start:span_end]).shape[0] == 1:
                valid.append(int(start))
        return np.asarray(valid, dtype=int)


class MPCWindowDataset(Dataset):
    """Build MPC samples with past context and future horizons."""

    def __init__(
        self,
        *,
        state: np.ndarray | None = None,
        output: np.ndarray | None = None,
        control: np.ndarray | None = None,
        reference: np.ndarray | None = None,
        target: np.ndarray | None = None,
        past_horizon: int = 10,
        prediction_horizon: int = 20,
        control_horizon: int = 5,
        indices: np.ndarray | None = None,
        episode_ids: np.ndarray | None = None,
        return_mode: str = "dict",
    ) -> None:
        self.state = _optional_2d(state)
        self.output = _optional_2d(output)
        self.control = _optional_2d(control)
        self.reference = _optional_2d(reference)
        self.target = _optional_2d(target)
        self.row_count = _shared_row_count(
            state=self.state,
            output=self.output,
            control=self.control,
            reference=self.reference,
            target=self.target,
        )
        if self.row_count == 0:
            raise ValueError("MPCWindowDataset requires at least one non-empty role array.")
        if not any(array is not None and array.shape[1] > 0 for array in self.input_arrays):
            raise ValueError("MPCWindowDataset requires at least one input role array.")
        if self.target is None:
            self.target = _concat_optional([self.output, self.state])
        if self.target is None or self.target.shape[1] == 0:
            raise ValueError("MPCWindowDataset requires target data or output/state roles.")
        if past_horizon <= 0 or prediction_horizon <= 0 or control_horizon <= 0:
            raise ValueError(
                "past_horizon, prediction_horizon, and control_horizon must be positive. "
                f"Current inputs: past_horizon={past_horizon}, "
                f"prediction_horizon={prediction_horizon}, control_horizon={control_horizon}."
            )
        self.past_horizon = int(past_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.control_horizon = int(control_horizon)
        self.return_mode = return_mode
        self.episode_ids = np.zeros(self.row_count, dtype=int) if episode_ids is None else np.asarray(episode_ids)
        if self.episode_ids.shape[0] != self.row_count:
            raise ValueError("episode_ids must have the same row count as MPC role arrays.")
        self.sample_starts = self._build_sample_starts(indices)

    @property
    def input_arrays(self) -> tuple[np.ndarray | None, ...]:
        """Return role arrays used in the past-context input."""

        return (self.state, self.output, self.control, self.reference)

    def __len__(self) -> int:
        """Return the number of valid MPC samples."""

        return int(self.sample_starts.shape[0])

    def __getitem__(self, idx: int):
        """Return one MPC sample as a dict of tensors or a tuple."""

        sample = self.sample_numpy(idx)
        if self.return_mode == "tuple":
            return (
                torch.as_tensor(sample.past, dtype=torch.float32),
                torch.as_tensor(sample.target_future, dtype=torch.float32),
            )
        if self.return_mode != "dict":
            raise ValueError("return_mode must be one of: dict, tuple.")
        return {
            "past": torch.as_tensor(sample.past, dtype=torch.float32),
            "target_future": torch.as_tensor(sample.target_future, dtype=torch.float32),
            "control_future": torch.as_tensor(sample.control_future, dtype=torch.float32),
            "reference_future": torch.as_tensor(sample.reference_future, dtype=torch.float32),
            "state_past": torch.as_tensor(sample.state_past, dtype=torch.float32),
            "output_past": torch.as_tensor(sample.output_past, dtype=torch.float32),
            "control_past": torch.as_tensor(sample.control_past, dtype=torch.float32),
            "reference_past": torch.as_tensor(sample.reference_past, dtype=torch.float32),
            "index": torch.as_tensor(sample.index, dtype=torch.long),
        }

    def sample_numpy(self, idx: int) -> MPCWindowSample:
        """Return one sample as numpy arrays."""

        start = int(self.sample_starts[idx])
        past = slice(start, start + self.past_horizon)
        prediction = slice(
            start + self.past_horizon,
            start + self.past_horizon + self.prediction_horizon,
        )
        control = slice(
            start + self.past_horizon,
            start + self.past_horizon + self.control_horizon,
        )
        state_past = _slice_or_empty(self.state, past, self.past_horizon)
        output_past = _slice_or_empty(self.output, past, self.past_horizon)
        control_past = _slice_or_empty(self.control, past, self.past_horizon)
        reference_past = _slice_or_empty(self.reference, past, self.past_horizon)
        past_matrix = _concat_optional(
            [state_past, output_past, control_past, reference_past],
            rows=self.past_horizon,
        )
        target_future = self.target[prediction]
        control_future = _slice_or_empty(self.control, control, self.control_horizon)
        reference_future = _slice_or_empty(
            self.reference,
            prediction,
            self.prediction_horizon,
        )
        return MPCWindowSample(
            past=past_matrix,
            target_future=target_future,
            control_future=control_future,
            reference_future=reference_future,
            state_past=state_past,
            output_past=output_past,
            control_past=control_past,
            reference_past=reference_past,
            index=start,
        )

    def _build_sample_starts(self, indices: np.ndarray | None) -> np.ndarray:
        span = self.past_horizon + max(self.prediction_horizon, self.control_horizon)
        if indices is None:
            candidates = np.arange(0, self.row_count - span + 1)
        else:
            candidates = np.asarray(indices, dtype=int)
        valid: list[int] = []
        for start in candidates:
            end = start + span
            if start < 0 or end > self.row_count:
                continue
            if np.unique(self.episode_ids[start:end]).shape[0] == 1:
                valid.append(int(start))
        return np.asarray(valid, dtype=int)


class DynamicWindowSubset(Dataset):
    """Subset wrapper preserving sample index metadata."""

    def __init__(self, dataset: DynamicWindowDataset, sample_indices: np.ndarray) -> None:
        self.dataset = dataset
        self.sample_indices = np.asarray(sample_indices, dtype=int)

    def __len__(self) -> int:
        """Return subset size."""

        return int(self.sample_indices.shape[0])

    def __getitem__(self, idx: int):
        """Return one subset sample."""

        return self.dataset[int(self.sample_indices[idx])]


def _as_2d(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D data. Current shape: {array.shape}.")
    return array


def _optional_2d(data: np.ndarray | None) -> np.ndarray | None:
    if data is None:
        return None
    return _as_2d(data)


def _resolve_sequence_task(task: str) -> str:
    normalized = str(task).strip().lower().replace("-", "_")
    aliases = {
        "many_to_one": "n_to_1",
        "sequence_to_one": "n_to_1",
        "last": "n_to_1",
        "one": "n_to_1",
        "many_to_many": "n_to_n",
        "sequence_to_sequence": "n_to_n",
        "all": "n_to_n",
        "future": "n_to_m",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"n_to_1", "n_to_n", "n_to_m"}:
        raise ValueError(
            f"Unknown sequence task {task!r}. Legal options are: n_to_1, n_to_n, n_to_m."
        )
    return normalized


def _shared_row_count(**arrays: np.ndarray | None) -> int:
    row_count: int | None = None
    for name, array in arrays.items():
        if array is None:
            continue
        if row_count is None:
            row_count = int(array.shape[0])
        elif array.shape[0] != row_count:
            raise ValueError(
                f"MPC role array {name!r} has {array.shape[0]} rows, "
                f"but expected {row_count}."
            )
    return 0 if row_count is None else row_count


def _concat_optional(arrays: list[np.ndarray | None], *, rows: int | None = None) -> np.ndarray | None:
    present = [array for array in arrays if array is not None and array.shape[1] > 0]
    if present:
        return np.concatenate(present, axis=1)
    if rows is None:
        return None
    return np.empty((rows, 0), dtype=float)


def _slice_or_empty(array: np.ndarray | None, rows: slice, row_count: int) -> np.ndarray:
    if array is None:
        return np.empty((row_count, 0), dtype=float)
    return array[rows]
