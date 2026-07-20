"""Observe numerical Jacobians and quantify GC stage symplecticity."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from classes.system.observation import IntegrationStage, StateMap

from .paths import next_block_index, notebook_output_directory, validate_block_name


def central_difference_jacobian(
	map_state: StateMap,
	state: np.ndarray,
	*,
	relative_step: float | None = None,
) -> np.ndarray:
	"""Differentiate a packed stage map using centered finite differences.

	``relative_step`` scales independently with each state coordinate. Its default
	is the cube root of machine epsilon, which balances truncation and round-off
	for a centered first derivative. The returned matrix has shape ``(D, D)``,
	where ``D`` is the internal packed-state dimension.
	"""
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The differentiated state must be a finite, non-empty vector.")
	scale = (
		float(np.cbrt(np.finfo(float).eps))
		if relative_step is None
		else float(relative_step)
	)
	if not np.isfinite(scale) or scale <= 0:
		raise ValueError("`relative_step` must be positive and finite.")

	dimension = value.size
	jacobian = np.empty((dimension, dimension), dtype=float)
	for column in range(dimension):
		increment = scale * max(1.0, abs(float(value[column])))
		perturbation = np.zeros_like(value)
		perturbation[column] = increment
		forward = np.asarray(map_state(value + perturbation), dtype=float)
		backward = np.asarray(map_state(value - perturbation), dtype=float)
		if forward.shape != value.shape or backward.shape != value.shape:
			raise ValueError("The differentiated stage map changed the state shape.")
		jacobian[:, column] = (forward - backward) / (2 * increment)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError("The numerical stage Jacobian contains non-finite values.")
	return jacobian


def gc_extended_symplectic_form(particle_count: int) -> np.ndarray:
	"""Return the cross-coupled form for two component-major GC copies.

	The internal state order is ``[x_first, y_first, x_second, y_second]`` and
	each block contains ``particle_count`` values. The triangular extended-phase
	updates pair the two copies, giving ``[[0, omega_gc], [omega_gc, 0]]`` rather
	than two diagonal canonical forms. The result has shape ``(4N, 4N)``.
	"""
	if (
		isinstance(particle_count, (bool, np.bool_))
		or not isinstance(particle_count, (int, np.integer))
		or particle_count < 1
	):
		raise ValueError("`particle_count` must be a positive integer.")
	count = int(particle_count)
	identity = np.eye(count)
	zero = np.zeros_like(identity)
	physical_form = np.block([[zero, identity], [-identity, zero]])
	copy_zero = np.zeros_like(physical_form)
	return np.block(
		[[copy_zero, physical_form], [physical_form, copy_zero]]
	)


@dataclass(frozen=True, slots=True)
class SymplecticityRecord:
	"""Scalar diagnostics associated with one observed stage Jacobian."""

	observation_index: int
	step_index: int
	stage_index: int
	flow_name: str
	time: float
	duration: float
	state_dimension: int
	particle_count: int
	determinant: float
	determinant_error: float
	log_abs_determinant: float
	condition_number: float
	defect_frobenius: float
	relative_defect: float
	max_abs_defect: float


@dataclass(frozen=True, slots=True)
class OutputBlock:
	"""Three synchronized files written for one buffered observation block."""

	index: int
	sample_count: int
	summary_path: Path
	jacobians_path: Path
	metadata_path: Path


@dataclass(slots=True)
class _BufferedSample:
	"""Matrix-valued data retained only until the next disk flush."""

	record: SymplecticityRecord
	state_before: np.ndarray
	state_after: np.ndarray
	jacobian: np.ndarray


class SymplecticityObserver:
	"""Calculate, report and persist selected GC stage Jacobians.

	A selected complete BM4 step contributes all twelve direct/adjoint stages.
	``sample_every`` selects complete steps, while ``chunk_size`` limits the number
	of full matrices held in memory before a synchronized CSV/NPZ/JSON block is
	written. This observer expects ``check_energy=False`` because the GC diagnostic
	form covers the two physical copies, not the optional momentum without its
	explicit time coordinate.
	"""

	def __init__(
		self,
		*,
		notebook_path: str | Path,
		particle_count: int,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "symplecticity",
		sample_every: int = 1,
		chunk_size: int = 24,
		relative_step: float | None = None,
		verbose: bool = True,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure notebook-derived storage and numerical differentiation."""
		if (
			isinstance(particle_count, (bool, np.bool_))
			or not isinstance(particle_count, (int, np.integer))
			or particle_count < 1
		):
			raise ValueError("`particle_count` must be a positive integer.")
		if (
			isinstance(sample_every, (bool, np.bool_))
			or not isinstance(sample_every, (int, np.integer))
			or sample_every < 1
		):
			raise ValueError("`sample_every` must be a positive integer.")
		if (
			isinstance(chunk_size, (bool, np.bool_))
			or not isinstance(chunk_size, (int, np.integer))
			or chunk_size < 1
		):
			raise ValueError("`chunk_size` must be a positive integer.")
		if relative_step is not None and (
			not np.isfinite(float(relative_step)) or float(relative_step) <= 0
		):
			raise ValueError("`relative_step` must be positive and finite.")

		self.output_directory = notebook_output_directory(
			notebook_path,
			project_root=project_root,
			run_date=run_date,
		)
		self.block_name = validate_block_name(block_name)
		self.particle_count = int(particle_count)
		self.sample_every = int(sample_every)
		self.chunk_size = int(chunk_size)
		self.relative_step = relative_step
		self.verbose = bool(verbose)
		self.metadata = dict(metadata or {})
		self._buffer: list[_BufferedSample] = []
		self._records: list[SymplecticityRecord] = []
		self._output_blocks: list[OutputBlock] = []
		self._closed = False
		self._next_index = next_block_index(self.output_directory, self.block_name)

	@property
	def records(self) -> tuple[SymplecticityRecord, ...]:
		"""Return all scalar records collected during this observer session."""
		return tuple(self._records)

	@property
	def output_blocks(self) -> tuple[OutputBlock, ...]:
		"""Return the synchronized file groups written by this session."""
		return tuple(self._output_blocks)

	def __enter__(self) -> SymplecticityObserver:
		"""Use the observer as a context manager so its final block is flushed."""
		return self

	def __exit__(self, *_exception: object) -> None:
		"""Flush observations even when the surrounding integration raises."""
		self.close()

	def __call__(self, stage: IntegrationStage) -> None:
		"""Observe every stage of each selected complete BM4 step."""
		if self._closed:
			raise RuntimeError("This symplecticity observer is already closed.")
		if stage.step_index % self.sample_every:
			return
		if stage.system_name != "SystemGC":
			raise TypeError("SymplecticityObserver only supports SystemGC stages.")
		state = np.asarray(stage.state_before, dtype=float)
		if state.ndim != 1 or state.size != 4 * self.particle_count:
			raise ValueError(
				"GC symplecticity requires the 4N doubled physical state; "
				"run the simulation with `check_energy=False`."
			)
		jacobian = central_difference_jacobian(
			stage.map_state,
			state,
			relative_step=self.relative_step,
		)
		form = gc_extended_symplectic_form(self.particle_count)
		defect = jacobian.T @ form @ jacobian - form
		sign, log_abs_determinant = np.linalg.slogdet(jacobian)
		determinant = float(sign * np.exp(log_abs_determinant))
		defect_frobenius = float(np.linalg.norm(defect, ord="fro"))
		record = SymplecticityRecord(
			observation_index=len(self._records),
			step_index=stage.step_index,
			stage_index=stage.stage_index,
			flow_name=stage.flow_name,
			time=stage.time,
			duration=stage.duration,
			state_dimension=state.size,
			particle_count=self.particle_count,
			determinant=determinant,
			determinant_error=abs(determinant - 1.0),
			log_abs_determinant=float(log_abs_determinant),
			condition_number=float(np.linalg.cond(jacobian)),
			defect_frobenius=defect_frobenius,
			relative_defect=defect_frobenius / float(np.linalg.norm(form, ord="fro")),
			max_abs_defect=float(np.max(np.abs(defect))),
		)
		self._records.append(record)
		self._buffer.append(
			_BufferedSample(
				record=record,
				state_before=state.copy(),
				state_after=np.asarray(stage.state_after, dtype=float).copy(),
				jacobian=jacobian,
			)
		)
		if self.verbose:
			print(
				f"[symplecticity] step={record.step_index:05d} "
				f"stage={record.stage_index:02d} {record.flow_name:<12} "
				f"relative_defect={record.relative_defect:.3e} "
				f"|det(J)-1|={record.determinant_error:.3e}"
			)
		if len(self._buffer) >= self.chunk_size:
			self.flush()

	def flush(self) -> OutputBlock | None:
		"""Write the current matrix buffer as one indexed output block."""
		if not self._buffer:
			return None
		self.output_directory.mkdir(parents=True, exist_ok=True)
		block_index = self._next_index
		stem = f"{self.block_name}_{{kind}}_{block_index:05d}"
		summary_path = self.output_directory / f"{stem.format(kind='summary')}.csv"
		jacobians_path = self.output_directory / f"{stem.format(kind='jacobians')}.npz"
		metadata_path = self.output_directory / f"{stem.format(kind='metadata')}.json"

		rows = [asdict(sample.record) for sample in self._buffer]
		with summary_path.open("w", encoding="utf-8", newline="") as stream:
			writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
			writer.writeheader()
			writer.writerows(rows)
		np.savez_compressed(
			jacobians_path,
			jacobians=np.stack([sample.jacobian for sample in self._buffer]),
			states_before=np.stack([sample.state_before for sample in self._buffer]),
			states_after=np.stack([sample.state_after for sample in self._buffer]),
			observation_indices=np.asarray(
				[sample.record.observation_index for sample in self._buffer], dtype=int
			),
			step_indices=np.asarray(
				[sample.record.step_index for sample in self._buffer], dtype=int
			),
			stage_indices=np.asarray(
				[sample.record.stage_index for sample in self._buffer], dtype=int
			),
			flow_names=np.asarray([sample.record.flow_name for sample in self._buffer]),
			times=np.asarray([sample.record.time for sample in self._buffer]),
			durations=np.asarray([sample.record.duration for sample in self._buffer]),
		)
		payload = {
			"schema_version": 1,
			"objective": "GC direct/adjoint stage symplecticity",
			"created_at": datetime.now().astimezone().isoformat(),
			"block_index": block_index,
			"sample_count": len(self._buffer),
			"sample_every_complete_steps": self.sample_every,
			"particle_count": self.particle_count,
			"finite_difference_relative_step": (
				float(np.cbrt(np.finfo(float).eps))
				if self.relative_step is None
				else float(self.relative_step)
			),
			"metadata": self.metadata,
		}
		with metadata_path.open("w", encoding="utf-8") as stream:
			json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
			stream.write("\n")

		block = OutputBlock(
			index=block_index,
			sample_count=len(self._buffer),
			summary_path=summary_path,
			jacobians_path=jacobians_path,
			metadata_path=metadata_path,
		)
		self._output_blocks.append(block)
		self._buffer.clear()
		self._next_index += 1
		return block

	def close(self) -> None:
		"""Flush the final partial block and stop accepting stage events."""
		if self._closed:
			return
		self.flush()
		self._closed = True


def _json_default(value: object) -> object:
	"""Serialize common NumPy and path metadata without losing scalar values."""
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Path):
		return str(value)
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


__all__ = [
	"OutputBlock",
	"SymplecticityObserver",
	"SymplecticityRecord",
	"central_difference_jacobian",
	"gc_extended_symplectic_form",
]
