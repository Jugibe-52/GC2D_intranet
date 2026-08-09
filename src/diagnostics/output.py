"""Shared synchronized CSV, NPZ and JSON diagnostic persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DiagnosticBlockPaths:
	"""Filesystem paths written for one synchronized diagnostic block."""

	summary: Path
	arrays: Path
	metadata: Path


def _json_default(value: object) -> object:
	"""Serialize common NumPy and path metadata without losing scalar values."""
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Path):
		return str(value)
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def write_diagnostic_block(
	*,
	output_directory: Path,
	block_name: str,
	block_index: int,
	rows: Sequence[Mapping[str, object]],
	arrays: Mapping[str, np.ndarray],
	metadata: Mapping[str, Any],
) -> DiagnosticBlockPaths:
	"""Write synchronized scalar, array and metadata files for one block."""
	if not rows:
		raise ValueError("A diagnostic output block requires at least one row.")
	output_directory.mkdir(parents=True, exist_ok=True)
	stem = f"{block_name}_{{kind}}_{block_index:05d}"
	paths = DiagnosticBlockPaths(
		summary=output_directory / f"{stem.format(kind='summary')}.csv",
		arrays=output_directory / f"{stem.format(kind='jacobians')}.npz",
		metadata=output_directory / f"{stem.format(kind='metadata')}.json",
	)
	with paths.summary.open("w", encoding="utf-8", newline="") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	np.savez_compressed(paths.arrays, **arrays)
	payload = {
		**dict(metadata),
		"schema_version": 1,
		"created_at": datetime.now().astimezone().isoformat(),
		"block_index": block_index,
		"sample_count": len(rows),
	}
	with paths.metadata.open("w", encoding="utf-8") as stream:
		json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
		stream.write("\n")
	return paths


__all__ = ["DiagnosticBlockPaths", "write_diagnostic_block"]
