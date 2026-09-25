"""Single-file CSV persistence for the five-method trajectory comparison."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
from typing import Any, TYPE_CHECKING

import numpy as np

from diagnostics.output import _json_default

from initial_conditions import GCInitialConfiguration
from solution import Solution

if TYPE_CHECKING:
	from studies import FiveMethodComparisonResult


FIVE_METHOD_COMPARISON_CSV_SCHEMA_VERSION = 1
_CORE_COLUMNS = ("schema_version", "metadata_json", "sample_index", "time")
_STEP_DIAGNOSTICS = (
	"nonlinear_iterations",
	"residual_evaluations",
	"nonlinear_residual_norms",
	"nonlinear_tolerances",
	"projection_multiplier_norms",
)


def _record_dictionary(value: object) -> dict[str, object]:
	"""Convert one dataclass result record to named scalar fields."""
	if not is_dataclass(value) or isinstance(value, type):
		raise TypeError("Comparison records must be dataclass instances.")
	return dict(asdict(value))


def _scalar_diagnostics(solution: Solution) -> dict[str, object]:
	"""Retain self-describing scalar diagnostics with the numeric series."""
	return {
		name: value.item() if isinstance(value, np.generic) else value
		for name, value in solution.diagnostics.items()
		if value is None or isinstance(value, (str, bool, int, float, np.generic))
	}


def _particle_column(scope: str, quantity: str, particle: int) -> str:
	"""Return one stable one-based particle-column name."""
	return f"{scope}.{quantity}.particle_{particle + 1}"


def _particle_history(
	value: np.ndarray,
	*,
	particle_count: int,
	sample_count: int,
	description: str,
) -> np.ndarray:
	"""Validate one finite particle-by-time history."""
	array = np.asarray(value, dtype=float)
	if array.shape != (particle_count, sample_count) or not np.all(np.isfinite(array)):
		raise ValueError(
			f"{description} must have shape ({particle_count}, {sample_count}) and be finite."
		)
	return array


def _append_particle_history(
	columns: dict[str, tuple[np.ndarray, int]],
	*,
	scope: str,
	quantity: str,
	values: np.ndarray,
) -> None:
	"""Add every particle row of a time-aligned history to CSV columns."""
	for particle, series in enumerate(values):
		columns[_particle_column(scope, quantity, particle)] = (series, 0)


class StoredFiveMethodComparison:
	"""Visualization-ready view of one persisted five-method comparison."""

	def __init__(
		self,
		*,
		path: Path,
		metadata: Mapping[str, Any],
		reference: SimpleNamespace,
		solutions: Mapping[str, Solution],
		accuracy: Mapping[str, SimpleNamespace],
		energy_accuracy: Mapping[str, SimpleNamespace],
		reference_energy_errors: np.ndarray,
		runtime_samples: Mapping[str, np.ndarray],
		wall_runtime_seconds: float,
		execution_log: tuple[SimpleNamespace, ...],
		summaries: tuple[SimpleNamespace, ...],
		nonlinear_summaries: tuple[SimpleNamespace, ...],
	) -> None:
		"""Own immutable mappings while preserving plotting-compatible views."""
		self.path = Path(path)
		self.metadata = MappingProxyType(dict(metadata))
		self.reference = reference
		self.solutions = MappingProxyType(dict(solutions))
		self.accuracy = MappingProxyType(dict(accuracy))
		self.energy_accuracy = MappingProxyType(dict(energy_accuracy))
		energy_errors = np.array(reference_energy_errors, dtype=float, copy=True)
		energy_errors.setflags(write=False)
		self.reference_energy_errors = energy_errors
		self.runtime_samples = MappingProxyType(
			{
				name: _readonly_array(values)
				for name, values in runtime_samples.items()
			}
		)
		self.wall_runtime_seconds = float(wall_runtime_seconds)
		self.execution_log = execution_log
		self._summaries = summaries
		self._nonlinear_summaries = nonlinear_summaries

	@property
	def total_study_runtime_seconds(self) -> float:
		"""Return the persisted complete study runtime."""
		return self.wall_runtime_seconds

	def summaries(self) -> tuple[SimpleNamespace, ...]:
		"""Return persisted accuracy, energy, and runtime summaries."""
		return self._summaries

	def nonlinear_work_summaries(self) -> tuple[SimpleNamespace, ...]:
		"""Return persisted Newton-work summaries for implicit methods."""
		return self._nonlinear_summaries


def _readonly_array(value: np.ndarray) -> np.ndarray:
	"""Own and freeze one finite floating-point array."""
	result = np.array(value, dtype=float, copy=True)
	if not np.all(np.isfinite(result)):
		raise ValueError("Stored comparison arrays must be finite.")
	result.setflags(write=False)
	return result


def write_five_method_comparison_csv(
	result: FiveMethodComparisonResult,
	path: str | Path,
	*,
	metadata: Mapping[str, Any] | None = None,
	overwrite: bool = False,
) -> Path:
	"""Persist all trajectories and plotted diagnostics in one wide CSV file."""
	target = Path(path)
	if target.suffix.lower() != ".csv":
		raise ValueError("The five-method comparison path must use the .csv suffix.")
	if target.exists() and not overwrite:
		raise FileExistsError(f"Comparison CSV already exists: {target}")

	times = np.asarray(result.reference.times, dtype=float)
	if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
		raise ValueError("Comparison times must be finite and strictly increasing.")
	sample_count = times.size
	particle_count = result.reference.states.shape[0] // 2
	method_names = tuple(result.solutions)
	if particle_count < 1 or not method_names:
		raise ValueError("The comparison must contain methods and trajectories.")

	columns: dict[str, tuple[np.ndarray, int]] = {}
	for reference_name, states in (
		("dop853", result.reference.states),
		("radau", result.reference.audit_states),
	):
		state_values = np.asarray(states, dtype=float)
		if state_values.shape != (2 * particle_count, sample_count):
			raise ValueError("Reference state histories must share the saved grid.")
		for quantity, values in (
			("x", state_values[:particle_count]),
			("y", state_values[particle_count:]),
		):
			_append_particle_history(
				columns,
				scope=f"reference.{reference_name}",
				quantity=quantity,
				values=values,
			)
	for quantity, values in (
		("audit_distance", result.reference.audit_distances),
		("energy_error", result.reference_energy_errors),
	):
		_append_particle_history(
			columns,
			scope="reference",
			quantity=quantity,
			values=_particle_history(
				values,
				particle_count=particle_count,
				sample_count=sample_count,
				description=f"Reference {quantity}",
			),
		)

	diagnostic_columns: dict[str, list[str]] = {}
	diagnostic_substeps: dict[str, dict[str, int]] = {}
	for method_name in method_names:
		solution = result.solutions[method_name]
		states = np.asarray(solution.states, dtype=float)
		if states.shape != (2 * particle_count, sample_count):
			raise ValueError("Every method trajectory must share the reference grid.")
		for quantity, values in (
			("x", states[:particle_count]),
			("y", states[particle_count:]),
			("distance", result.accuracy[method_name].distances),
			("energy_error", result.energy_accuracy[method_name].errors),
		):
			_append_particle_history(
				columns,
				scope=method_name,
				quantity=quantity,
				values=_particle_history(
					values,
					particle_count=particle_count,
					sample_count=sample_count,
					description=f"{method_name} {quantity}",
				),
			)
		accuracy = result.accuracy[method_name]
		for quantity in ("rms_distance", "mean_distance", "maximum_distance"):
			values = np.asarray(getattr(accuracy, quantity), dtype=float)
			if values.shape != (sample_count,) or not np.all(np.isfinite(values)):
				raise ValueError(f"{method_name} {quantity} must align with time.")
			columns[f"{method_name}.{quantity}"] = (values, 0)

		diagnostic_columns[method_name] = []
		diagnostic_substeps[method_name] = {}
		for diagnostic_name in _STEP_DIAGNOSTICS:
			source_name = diagnostic_name
			if diagnostic_name == "residual_evaluations" and (
				source_name not in solution.diagnostics
			):
				source_name = "residual_evaluations_per_step"
			if source_name not in solution.diagnostics:
				continue
			values = np.asarray(solution.diagnostics[source_name])
			step_count = result.config.step_count
			if values.shape != (step_count,) or not np.all(np.isfinite(values)):
				raise ValueError(f"{method_name} {source_name} must align with steps.")
			stride, remainder = divmod(step_count, sample_count - 1)
			if remainder or stride < 1:
				raise ValueError("Saved intervals must contain complete integration steps.")
			diagnostic_substeps[method_name][diagnostic_name] = stride
			for substep in range(stride):
				suffix = "" if substep == 0 else f".substep.{substep}"
				columns[f"{method_name}.diagnostic.{diagnostic_name}{suffix}"] = (values[substep::stride], 1)
			diagnostic_columns[method_name].append(diagnostic_name)

	payload = {
		"diagnostic_substeps": diagnostic_substeps,
		"schema_version": FIVE_METHOD_COMPARISON_CSV_SCHEMA_VERSION,
		"created_at": datetime.now().astimezone().isoformat(),
		"sample_count": sample_count,
		"particle_count": particle_count,
		"method_names": method_names,
		"implicit_method_names": tuple(
			name for name in method_names if "nonlinear_iterations" in diagnostic_columns[name]
		),
		"diagnostic_columns": diagnostic_columns,
		"config": _record_dictionary(result.config),
		"summaries": [_record_dictionary(row) for row in result.summaries()],
		"nonlinear_summaries": [
			_record_dictionary(row) for row in result.nonlinear_work_summaries()
		],
		"runtime_samples": {
			name: np.asarray(result.runtime_samples[name], dtype=float).tolist()
			for name in method_names
		},
		"wall_runtime_seconds": result.wall_runtime_seconds,
		"execution_log": [_record_dictionary(row) for row in result.execution_log],
		"reference": {
			"dop853_runtime_seconds": result.reference.dop853_runtime_seconds,
			"radau_runtime_seconds": result.reference.radau_runtime_seconds,
			"dop853_function_evaluations": result.reference.dop853_function_evaluations,
			"radau_function_evaluations": result.reference.radau_function_evaluations,
		},
		"solution_diagnostics": {
			name: _scalar_diagnostics(result.solutions[name]) for name in method_names
		},
		"experiment": dict(metadata or {}),
	}
	serialized_metadata = json.dumps(
		payload,
		sort_keys=True,
		separators=(",", ":"),
		default=_json_default,
	)

	target.parent.mkdir(parents=True, exist_ok=True)
	temporary_path: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			mode="w",
			encoding="utf-8",
			newline="",
			dir=target.parent,
			prefix=f".{target.stem}-",
			suffix=".csv",
			delete=False,
		) as stream:
			temporary_path = Path(stream.name)
			writer = csv.DictWriter(
				stream,
				fieldnames=[*_CORE_COLUMNS, *columns],
				lineterminator="\n",
			)
			writer.writeheader()
			for sample_index, time in enumerate(times):
				row: dict[str, object] = {
					"schema_version": FIVE_METHOD_COMPARISON_CSV_SCHEMA_VERSION,
					"metadata_json": serialized_metadata if sample_index == 0 else "",
					"sample_index": sample_index,
					"time": format(float(time), ".17g"),
				}
				for name, (values, offset) in columns.items():
					row[name] = (
						format(float(values[sample_index - offset]), ".17g")
						if sample_index >= offset
						else ""
					)
				writer.writerow(row)
		os.replace(temporary_path, target)
		temporary_path = None
	finally:
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)
	return target


def _stack_particle_columns(
	columns: Mapping[str, np.ndarray],
	*,
	scope: str,
	quantity: str,
	particle_count: int,
) -> np.ndarray:
	"""Rebuild one particle-by-time history from named CSV columns."""
	try:
		return np.vstack(
			[
				columns[_particle_column(scope, quantity, particle)]
				for particle in range(particle_count)
			]
		)
	except KeyError as exc:
		raise ValueError(f"Comparison CSV is missing column {exc.args[0]!r}.") from exc


def load_five_method_comparison_csv(path: str | Path) -> StoredFiveMethodComparison:
	"""Load and validate one visualization-ready five-method comparison CSV."""
	source_path = Path(path)
	if not source_path.is_file():
		raise FileNotFoundError(f"Comparison CSV not found: {source_path}")
	with source_path.open("r", encoding="utf-8", newline="") as stream:
		reader = csv.reader(stream)
		try:
			header = next(reader)
			first_row = next(reader)
		except StopIteration as exc:
			raise ValueError("Comparison CSV must contain a header and data rows.") from exc
		if len(first_row) != len(header) or len(set(header)) != len(header):
			raise ValueError("Comparison CSV has malformed or duplicate columns.")
		indices = {name: index for index, name in enumerate(header)}
		missing = [name for name in _CORE_COLUMNS if name not in indices]
		if missing:
			raise ValueError(f"Comparison CSV is missing core columns: {missing}.")
		try:
			payload = json.loads(first_row[indices["metadata_json"]])
		except (json.JSONDecodeError, TypeError) as exc:
			raise ValueError("Comparison CSV metadata is not valid JSON.") from exc
		if (
			int(first_row[indices["schema_version"]])
			!= FIVE_METHOD_COMPARISON_CSV_SCHEMA_VERSION
			or payload.get("schema_version")
			!= FIVE_METHOD_COMPARISON_CSV_SCHEMA_VERSION
		):
			raise ValueError("Unsupported comparison CSV schema version.")
		sample_count = int(payload["sample_count"])
		numeric_names = [name for name in header if name not in _CORE_COLUMNS]
		columns = {name: np.full(sample_count, np.nan) for name in numeric_names}
		times = np.empty(sample_count)

		def consume(row: list[str], expected_index: int) -> None:
			"""Parse one rectangular row into preallocated numeric columns."""
			if len(row) != len(header):
				raise ValueError("Comparison CSV contains a malformed data row.")
			if int(row[indices["sample_index"]]) != expected_index:
				raise ValueError("Comparison CSV sample indices are not contiguous.")
			if int(row[indices["schema_version"]]) != FIVE_METHOD_COMPARISON_CSV_SCHEMA_VERSION:
				raise ValueError("Comparison CSV mixes schema versions.")
			times[expected_index] = float(row[indices["time"]])
			for name in numeric_names:
				text = row[indices[name]]
				if text:
					columns[name][expected_index] = float(text)

		if sample_count < 2:
			raise ValueError("Comparison CSV must contain at least two samples.")
		consume(first_row, 0)
		rows_read = 1
		for row in reader:
			if rows_read >= sample_count:
				raise ValueError("Comparison CSV contains more rows than declared.")
			consume(row, rows_read)
			rows_read += 1
		if rows_read != sample_count:
			raise ValueError("Comparison CSV row count does not match its metadata.")

	if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
		raise ValueError("Comparison CSV times must be finite and increasing.")
	particle_count = int(payload["particle_count"])
	method_names = tuple(str(name) for name in payload["method_names"])
	if particle_count < 1 or not method_names or len(set(method_names)) != len(method_names):
		raise ValueError("Comparison CSV method or particle metadata is invalid.")

	def states_for(scope: str) -> np.ndarray:
		"""Load packed planar states for one method or reference scope."""
		return np.vstack(
			(
				_stack_particle_columns(
					columns, scope=scope, quantity="x", particle_count=particle_count
				),
				_stack_particle_columns(
					columns, scope=scope, quantity="y", particle_count=particle_count
				),
			)
		)

	reference_states = states_for("reference.dop853")
	audit_states = states_for("reference.radau")
	audit_distances = _stack_particle_columns(
		columns,
		scope="reference",
		quantity="audit_distance",
		particle_count=particle_count,
	)
	reference_energy_errors = _stack_particle_columns(
		columns,
		scope="reference",
		quantity="energy_error",
		particle_count=particle_count,
	)
	for values in (reference_states, audit_states, audit_distances, reference_energy_errors):
		if not np.all(np.isfinite(values)):
			raise ValueError("Comparison CSV contains incomplete reference histories.")
	duration = float(times[-1] - times[0])
	reference_metadata = payload["reference"]
	reference = SimpleNamespace(
		times=_readonly_array(times),
		states=_readonly_array(reference_states),
		audit_states=_readonly_array(audit_states),
		audit_distances=_readonly_array(audit_distances),
		time_integrated_rms_floor=float(
			np.sqrt(np.trapz(np.mean(audit_distances**2, axis=0), times) / duration)
		),
		dop853_runtime_seconds=float(reference_metadata["dop853_runtime_seconds"]),
		radau_runtime_seconds=float(reference_metadata["radau_runtime_seconds"]),
		dop853_function_evaluations=int(
			reference_metadata["dop853_function_evaluations"]
		),
		radau_function_evaluations=int(reference_metadata["radau_function_evaluations"]),
	)

	solutions: dict[str, Solution] = {}
	accuracy: dict[str, SimpleNamespace] = {}
	energy_accuracy: dict[str, SimpleNamespace] = {}
	source: GCInitialConfiguration | None = None
	for method_name in method_names:
		states = states_for(method_name)
		if not np.all(np.isfinite(states)):
			raise ValueError(f"Comparison CSV has incomplete {method_name} states.")
		if source is None:
			source = GCInitialConfiguration(states[:, 0])
		else:
			source_initial_state = source.initial_state
			assert source_initial_state is not None
			if not np.array_equal(states[:, 0], source_initial_state):
				raise ValueError("Stored method trajectories have different origins.")
		diagnostics = dict(payload["solution_diagnostics"][method_name])
		for diagnostic_name in payload["diagnostic_columns"][method_name]:
			column_name = f"{method_name}.diagnostic.{diagnostic_name}"
			try:
				values = columns[column_name]
			except KeyError as exc:
				raise ValueError(f"Comparison CSV is missing {column_name!r}.") from exc
			if not np.isnan(values[0]) or not np.all(np.isfinite(values[1:])):
				raise ValueError(f"Stored diagnostic {column_name!r} is misaligned.")
			stride = payload.get("diagnostic_substeps", {}).get(method_name, {}).get(diagnostic_name, 1)
			if stride > 1:
				parts = [values[1:]]
				for substep in range(1, stride):
					extra = columns[f"{column_name}.substep.{substep}"]
					if not np.isnan(extra[0]) or not np.all(np.isfinite(extra[1:])):
						raise ValueError("Stored substep diagnostics are misaligned.")
					parts.append(extra[1:])
				values = np.concatenate(([np.nan], np.column_stack(parts).ravel()))
			diagnostics[diagnostic_name] = (
				values[1:].astype(int)
				if diagnostic_name in {"nonlinear_iterations", "residual_evaluations"}
				else values[1:]
			)
		solutions[method_name] = Solution(
			t=times,
			states=states,
			source=source,
			diagnostics=diagnostics,
		)
		distances = _stack_particle_columns(
			columns,
			scope=method_name,
			quantity="distance",
			particle_count=particle_count,
		)
		accuracy[method_name] = SimpleNamespace(
			method_name=method_name,
			distances=_readonly_array(distances),
			rms_distance=_readonly_array(columns[f"{method_name}.rms_distance"]),
			mean_distance=_readonly_array(columns[f"{method_name}.mean_distance"]),
			maximum_distance=_readonly_array(columns[f"{method_name}.maximum_distance"]),
		)
		energy_errors = _stack_particle_columns(
			columns,
			scope=method_name,
			quantity="energy_error",
			particle_count=particle_count,
		)
		maximum_energy_error = np.max(np.abs(energy_errors), axis=0)
		energy_accuracy[method_name] = SimpleNamespace(
			method_name=method_name,
			errors=_readonly_array(energy_errors),
			rms_error=_readonly_array(np.sqrt(np.mean(energy_errors**2, axis=0))),
			maximum_absolute_error=_readonly_array(maximum_energy_error),
			running_maximum_absolute_error=_readonly_array(
				np.maximum.accumulate(maximum_energy_error)
			),
		)
	assert source is not None
	source_initial_state = source.initial_state
	assert source_initial_state is not None
	if not np.array_equal(reference.states[:, 0], source_initial_state):
		raise ValueError("Stored reference and method trajectories have different origins.")

	return StoredFiveMethodComparison(
		path=source_path,
		metadata=payload,
		reference=reference,
		solutions=solutions,
		accuracy=accuracy,
		energy_accuracy=energy_accuracy,
		reference_energy_errors=reference_energy_errors,
		runtime_samples={
			name: np.asarray(payload["runtime_samples"][name], dtype=float)
			for name in method_names
		},
		wall_runtime_seconds=float(payload["wall_runtime_seconds"]),
		execution_log=tuple(SimpleNamespace(**row) for row in payload["execution_log"]),
		summaries=tuple(SimpleNamespace(**row) for row in payload["summaries"]),
		nonlinear_summaries=tuple(
			SimpleNamespace(**row) for row in payload["nonlinear_summaries"]
		),
	)


__all__ = [
	"FIVE_METHOD_COMPARISON_CSV_SCHEMA_VERSION",
	"StoredFiveMethodComparison",
	"load_five_method_comparison_csv",
	"write_five_method_comparison_csv",
]
