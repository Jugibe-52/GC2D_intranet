"""Compressed NPZ persistence for parallel BM4 recurrence campaigns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from studies.bm4_parallel_recurrence import ParallelBM4RecurrenceResult


PARALLEL_BM4_RECURRENCE_NPZ_SCHEMA_VERSION = 1
_ARCHIVE_KEYS = frozenset(
	(
		"metadata_json",
		"times",
		"initial_positions",
		"positions",
		"runtime_seconds",
		"total_newton_iterations",
		"mean_newton_iterations",
		"maximum_newton_iterations",
		"maximum_residual_to_tolerance",
		"wall_runtime_seconds",
	)
)


def _json_default(value: object) -> object:
	"""Serialize NumPy scalars, arrays, and paths used by study metadata."""
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Path):
		return str(value)
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


@dataclass(frozen=True, slots=True)
class StoredParallelBM4Recurrence:
	"""A validated recurrence result and its immutable archive metadata."""

	path: Path
	result: ParallelBM4RecurrenceResult
	metadata: Mapping[str, Any]

	def __post_init__(self) -> None:
		"""Normalize the archive path and prevent top-level metadata mutation."""
		from studies.bm4_parallel_recurrence import ParallelBM4RecurrenceResult

		if not isinstance(self.result, ParallelBM4RecurrenceResult):
			raise TypeError("`result` must be ParallelBM4RecurrenceResult.")
		object.__setattr__(self, "path", Path(self.path))
		object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def write_parallel_bm4_recurrence_npz(
	result: ParallelBM4RecurrenceResult,
	path: str | Path,
	*,
	metadata: Mapping[str, Any] | None = None,
	overwrite: bool = False,
) -> Path:
	"""Atomically persist a complete parallel BM4 recurrence result."""
	from studies.bm4_parallel_recurrence import ParallelBM4RecurrenceResult

	if not isinstance(result, ParallelBM4RecurrenceResult):
		raise TypeError("`result` must be ParallelBM4RecurrenceResult.")
	target = Path(path)
	if target.suffix.lower() != ".npz":
		raise ValueError("The parallel BM4 recurrence path must use the .npz suffix.")
	if target.exists() and not overwrite:
		raise FileExistsError(f"Parallel BM4 recurrence archive already exists: {target}")

	metadata_document = {
		"schema_version": PARALLEL_BM4_RECURRENCE_NPZ_SCHEMA_VERSION,
		"config": asdict(result.config),
		"experiment": dict(metadata or {}),
	}
	metadata_json = json.dumps(
		metadata_document,
		default=_json_default,
		sort_keys=True,
		separators=(",", ":"),
	)

	target.parent.mkdir(parents=True, exist_ok=True)
	descriptor, temporary_name = tempfile.mkstemp(
		prefix=f".{target.name}.",
		suffix=".tmp",
		dir=target.parent,
	)
	os.close(descriptor)
	temporary_path = Path(temporary_name)
	try:
		with temporary_path.open("wb") as stream:
			np.savez_compressed(
				stream,
				metadata_json=np.asarray(metadata_json),
				times=result.times,
				initial_positions=result.initial_positions,
				positions=result.positions,
				runtime_seconds=result.runtime_seconds,
				total_newton_iterations=result.total_newton_iterations,
				mean_newton_iterations=result.mean_newton_iterations,
				maximum_newton_iterations=result.maximum_newton_iterations,
				maximum_residual_to_tolerance=result.maximum_residual_to_tolerance,
				wall_runtime_seconds=np.asarray(result.wall_runtime_seconds),
			)
		os.replace(temporary_path, target)
	finally:
		temporary_path.unlink(missing_ok=True)
	return target


def load_parallel_bm4_recurrence_npz(
	path: str | Path,
) -> StoredParallelBM4Recurrence:
	"""Load and validate a complete parallel BM4 recurrence archive."""
	from studies.bm4_parallel_recurrence import (
		ParallelBM4RecurrenceConfig,
		ParallelBM4RecurrenceResult,
	)

	source = Path(path)
	if source.suffix.lower() != ".npz":
		raise ValueError("The parallel BM4 recurrence path must use the .npz suffix.")
	if not source.is_file():
		raise FileNotFoundError(f"Parallel BM4 recurrence archive not found: {source}")

	with np.load(source, allow_pickle=False) as archive:
		missing = _ARCHIVE_KEYS.difference(archive.files)
		if missing:
			raise ValueError(
				"Parallel BM4 recurrence archive is missing fields: "
				+ ", ".join(sorted(missing))
			)
		metadata_value = np.asarray(archive["metadata_json"])
		if metadata_value.shape != ():
			raise ValueError("Parallel BM4 metadata must be one JSON scalar.")
		metadata_document = json.loads(str(metadata_value.item()))
		arrays = {
			name: np.array(archive[name], copy=True)
			for name in _ARCHIVE_KEYS
			if name not in {"metadata_json", "wall_runtime_seconds"}
		}
		wall_runtime_value = np.asarray(archive["wall_runtime_seconds"], dtype=float)

	if not isinstance(metadata_document, dict):
		raise ValueError("Parallel BM4 metadata must decode to a JSON object.")
	if metadata_document.get("schema_version") != PARALLEL_BM4_RECURRENCE_NPZ_SCHEMA_VERSION:
		raise ValueError("Unsupported parallel BM4 recurrence archive schema version.")
	config_values = metadata_document.get("config")
	if not isinstance(config_values, dict):
		raise ValueError("Parallel BM4 metadata does not contain a valid configuration.")
	config_values = dict(config_values)
	config_values["t_span"] = tuple(config_values["t_span"])
	if wall_runtime_value.shape != ():
		raise ValueError("Parallel BM4 wall runtime must be scalar.")

	config = ParallelBM4RecurrenceConfig(**config_values)
	result = ParallelBM4RecurrenceResult(
		config=config,
		times=arrays["times"],
		initial_positions=arrays["initial_positions"],
		positions=arrays["positions"],
		runtime_seconds=arrays["runtime_seconds"],
		total_newton_iterations=arrays["total_newton_iterations"],
		mean_newton_iterations=arrays["mean_newton_iterations"],
		maximum_newton_iterations=arrays["maximum_newton_iterations"],
		maximum_residual_to_tolerance=arrays["maximum_residual_to_tolerance"],
		wall_runtime_seconds=float(wall_runtime_value),
	)
	return StoredParallelBM4Recurrence(
		path=source,
		result=result,
		metadata=metadata_document,
	)


__all__ = [
	"PARALLEL_BM4_RECURRENCE_NPZ_SCHEMA_VERSION",
	"StoredParallelBM4Recurrence",
	"load_parallel_bm4_recurrence_npz",
	"write_parallel_bm4_recurrence_npz",
]
