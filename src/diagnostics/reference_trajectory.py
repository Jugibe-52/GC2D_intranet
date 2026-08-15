"""Versioned persistence for high-precision numerical reference trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .paths import find_project_root


_REFERENCE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_VERSION = re.compile(r"^v[1-9][0-9]*$")


@dataclass(frozen=True, slots=True)
class ReferenceTrajectoryPaths:
	"""Files forming one self-documented versioned reference artifact."""

	directory: Path
	trajectory: Path
	metadata: Path
	readme: Path


@dataclass(frozen=True, slots=True)
class StoredReferenceTrajectory:
	"""Validated arrays and metadata loaded from a reference artifact."""

	times: np.ndarray
	states: np.ndarray
	initial_state: np.ndarray
	audit_states: np.ndarray
	audit_periodic_distances: np.ndarray
	metadata: Mapping[str, Any]
	paths: ReferenceTrajectoryPaths

	def __post_init__(self) -> None:
		"""Own immutable arrays after validating the packed time history."""
		times = np.array(self.times, dtype=float, copy=True)
		states = np.array(self.states, dtype=float, copy=True)
		initial_state = np.array(self.initial_state, dtype=float, copy=True)
		audit_states = np.array(self.audit_states, dtype=float, copy=True)
		audit_periodic_distances = np.array(
			self.audit_periodic_distances,
			dtype=float,
			copy=True,
		)
		if (
			times.ndim != 1
			or times.size < 2
			or not np.all(np.isfinite(times))
			or np.any(np.diff(times) <= 0.0)
		):
			raise ValueError("Reference times must be finite and strictly increasing.")
		if (
			states.ndim != 2
			or states.shape[1] != times.size
			or states.shape[0] == 0
			or not np.all(np.isfinite(states))
		):
			raise ValueError("Reference states must be a finite packed time history.")
		if initial_state.shape != (states.shape[0],) or not np.all(
			np.isfinite(initial_state)
		):
			raise ValueError("The stored reference initial state has an invalid shape.")
		if not np.array_equal(states[:, 0], initial_state):
			raise ValueError("The first reference sample must equal the initial state.")
		if audit_states.shape != states.shape or not np.all(np.isfinite(audit_states)):
			raise ValueError("Audit states must match the reference state history.")
		if (
			states.shape[0] % 2
			or audit_periodic_distances.shape
			!= (states.shape[0] // 2, times.size)
			or not np.all(np.isfinite(audit_periodic_distances))
			or np.any(audit_periodic_distances < 0.0)
		):
			raise ValueError("Audit distances must have shape (particles, samples).")
		for value in (
			times,
			states,
			initial_state,
			audit_states,
			audit_periodic_distances,
		):
			value.setflags(write=False)
		object.__setattr__(self, "times", times)
		object.__setattr__(self, "states", states)
		object.__setattr__(self, "initial_state", initial_state)
		object.__setattr__(self, "audit_states", audit_states)
		object.__setattr__(
			self,
			"audit_periodic_distances",
			audit_periodic_distances,
		)
		object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def reference_trajectory_output_directory(
	notebook_path: str | Path,
	*,
	reference_name: str = "example_trajectory",
	version: str = "v1",
	project_root: str | Path | None = None,
) -> Path:
	"""Map an accuracy notebook to a stable versioned output directory.

	For example, a notebook below ``notebooks/developements/accuracy`` maps to
	``outputs/developements/accuracy/example_trajectory/v1``. Unlike ordinary
	diagnostic blocks, a versioned numerical reference intentionally has no date
	component so later accuracy notebooks can address it deterministically.
	"""
	if not isinstance(reference_name, str) or not _REFERENCE_NAME.fullmatch(
		reference_name
	):
		raise ValueError("`reference_name` must contain only letters, numbers, '_' and '-'.")
	if not isinstance(version, str) or not _VERSION.fullmatch(version):
		raise ValueError("`version` must have the form 'v1', 'v2', and so on.")
	root = (
		find_project_root(Path.cwd())
		if project_root is None
		else Path(project_root).expanduser().resolve()
	)
	notebook = Path(notebook_path).expanduser()
	if not notebook.is_absolute():
		notebook = root / notebook
	notebook = notebook.resolve()
	try:
		relative = notebook.relative_to(root / "notebooks")
	except ValueError as exc:
		raise ValueError("The notebook must be located below the project notebooks directory.") from exc
	if relative.suffix != ".ipynb":
		raise ValueError("`notebook_path` must identify an .ipynb file.")
	return root / "outputs" / relative.parent / reference_name / version


def _array_digest(
	*,
	times: np.ndarray,
	states: np.ndarray,
	initial_state: np.ndarray,
	audit_states: np.ndarray,
	audit_periodic_distances: np.ndarray,
) -> str:
	"""Hash array names, shapes, dtypes, and values in a stable order."""
	digest = hashlib.sha256()
	for name, value in (
		("times", times),
		("states", states),
		("initial_state", initial_state),
		("audit_states", audit_states),
		("audit_periodic_distances", audit_periodic_distances),
	):
		array = np.ascontiguousarray(value)
		digest.update(name.encode("ascii"))
		digest.update(str(array.dtype).encode("ascii"))
		digest.update(json.dumps(array.shape).encode("ascii"))
		digest.update(array.tobytes(order="C"))
	return digest.hexdigest()


def _json_default(value: object) -> object:
	"""Serialize NumPy values and paths used in reference metadata."""
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Path):
		return str(value)
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def write_reference_trajectory(
	*,
	output_directory: Path,
	times: np.ndarray,
	states: np.ndarray,
	initial_state: np.ndarray,
	audit_states: np.ndarray,
	audit_periodic_distances: np.ndarray,
	metadata: Mapping[str, Any],
	explanation: str,
	overwrite: bool = False,
) -> StoredReferenceTrajectory:
	"""Write one checksummed NPZ, JSON manifest, and explanatory README."""
	paths = ReferenceTrajectoryPaths(
		directory=Path(output_directory),
		trajectory=Path(output_directory) / "trajectory.npz",
		metadata=Path(output_directory) / "metadata.json",
		readme=Path(output_directory) / "README.md",
	)
	artifact = StoredReferenceTrajectory(
		times=times,
		states=states,
		initial_state=initial_state,
		audit_states=audit_states,
		audit_periodic_distances=audit_periodic_distances,
		metadata=metadata,
		paths=paths,
	)
	if not isinstance(explanation, str) or not explanation.strip():
		raise ValueError("A non-empty reference explanation is required.")
	if not overwrite:
		existing = [
			path
			for path in (paths.trajectory, paths.metadata, paths.readme)
			if path.exists()
		]
		if existing:
			raise FileExistsError(
				"The versioned reference already exists; select a new version or "
				"set `overwrite=True` explicitly."
			)
	paths.directory.mkdir(parents=True, exist_ok=True)
	payload = {
		**dict(metadata),
		"schema_version": 1,
		"created_at": datetime.now().astimezone().isoformat(),
		"trajectory_sha256": _array_digest(
			times=artifact.times,
			states=artifact.states,
			initial_state=artifact.initial_state,
			audit_states=artifact.audit_states,
			audit_periodic_distances=artifact.audit_periodic_distances,
		),
	}
	# Serialize before touching an existing version. Staged files are moved into
	# place atomically, with the checksum-bearing manifest published last.
	serialized_metadata = (
		json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
	)
	serialized_readme = explanation.rstrip() + "\n"
	temporary_paths: list[Path] = []
	try:
		with tempfile.NamedTemporaryFile(
			dir=paths.directory,
			prefix=".trajectory-",
			suffix=".npz",
			delete=False,
		) as temporary:
			temporary_trajectory = Path(temporary.name)
		temporary_paths.append(temporary_trajectory)
		np.savez_compressed(
			temporary_trajectory,
			times=artifact.times,
			states=artifact.states,
			initial_state=artifact.initial_state,
			audit_states=artifact.audit_states,
			audit_periodic_distances=artifact.audit_periodic_distances,
		)
		with tempfile.NamedTemporaryFile(
			mode="w",
			encoding="utf-8",
			dir=paths.directory,
			prefix=".reference-readme-",
			suffix=".md",
			delete=False,
		) as temporary:
			temporary.write(serialized_readme)
			temporary_readme = Path(temporary.name)
		temporary_paths.append(temporary_readme)
		with tempfile.NamedTemporaryFile(
			mode="w",
			encoding="utf-8",
			dir=paths.directory,
			prefix=".reference-metadata-",
			suffix=".json",
			delete=False,
		) as temporary:
			temporary.write(serialized_metadata)
			temporary_metadata = Path(temporary.name)
		temporary_paths.append(temporary_metadata)

		os.replace(temporary_trajectory, paths.trajectory)
		os.replace(temporary_readme, paths.readme)
		os.replace(temporary_metadata, paths.metadata)
	finally:
		for temporary_path in temporary_paths:
			temporary_path.unlink(missing_ok=True)
	return load_reference_trajectory(paths.directory)


def load_reference_trajectory(
	output_directory: str | Path,
) -> StoredReferenceTrajectory:
	"""Load a versioned reference and verify its manifest checksum."""
	directory = Path(output_directory).expanduser().resolve()
	paths = ReferenceTrajectoryPaths(
		directory=directory,
		trajectory=directory / "trajectory.npz",
		metadata=directory / "metadata.json",
		readme=directory / "README.md",
	)
	for path in (paths.trajectory, paths.metadata, paths.readme):
		if not path.is_file():
			raise FileNotFoundError(f"Reference artifact file not found: {path}")
	with np.load(paths.trajectory, allow_pickle=False) as archive:
		required = {
			"times",
			"states",
			"initial_state",
			"audit_states",
			"audit_periodic_distances",
		}
		if set(archive.files) != required:
			raise ValueError("The reference NPZ contains an unexpected array schema.")
		times = np.asarray(archive["times"], dtype=float)
		states = np.asarray(archive["states"], dtype=float)
		initial_state = np.asarray(archive["initial_state"], dtype=float)
		audit_states = np.asarray(archive["audit_states"], dtype=float)
		audit_periodic_distances = np.asarray(
			archive["audit_periodic_distances"],
			dtype=float,
		)
	with paths.metadata.open(encoding="utf-8") as stream:
		metadata = json.load(stream)
	if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
		raise ValueError("Unsupported reference metadata schema.")
	expected_digest = metadata.get("trajectory_sha256")
	actual_digest = _array_digest(
		times=times,
		states=states,
		initial_state=initial_state,
		audit_states=audit_states,
		audit_periodic_distances=audit_periodic_distances,
	)
	if expected_digest != actual_digest:
		raise ValueError("The reference trajectory checksum does not match its manifest.")
	return StoredReferenceTrajectory(
		times=times,
		states=states,
		initial_state=initial_state,
		audit_states=audit_states,
		audit_periodic_distances=audit_periodic_distances,
		metadata=metadata,
		paths=paths,
	)


__all__ = [
	"ReferenceTrajectoryPaths",
	"StoredReferenceTrajectory",
	"load_reference_trajectory",
	"reference_trajectory_output_directory",
	"write_reference_trajectory",
]
