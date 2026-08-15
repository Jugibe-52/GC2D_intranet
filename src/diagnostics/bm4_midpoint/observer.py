"""Persist exact midpoint-BM4 physical-flow symplecticity diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from diagnostics.output import write_diagnostic_block
from diagnostics.paths import (
	next_block_index,
	notebook_output_directory,
	validate_block_name,
)
from dynamics import GuidingCenterJacobianSystem
from simulation import IntegrationStage

from .jacobians import (
	MIDPOINT_BM4_STAGE_COUNT,
	midpoint_bm4_stage_particle_jacobians,
)


_METHOD_NAME = "MidpointBM4"
_JACOBIAN_METHOD = "explicit_uncoupled_stage_factorization"


@dataclass(frozen=True, slots=True)
class MidpointBM4SymplecticityRecord:
	"""Averaged per-particle defects at one complete physical-flow time."""

	observation_index: int
	step_index: int
	time: float
	duration: float
	particle_count: int
	mean_local_relative_defect: float
	std_local_relative_defect: float
	max_local_relative_defect: float
	mean_accumulated_relative_defect: float
	std_accumulated_relative_defect: float
	max_accumulated_relative_defect: float
	mean_local_determinant_error: float
	mean_accumulated_determinant_error: float


@dataclass(frozen=True, slots=True)
class MidpointBM4SymplecticityOutputBlock:
	"""Synchronized scalar, Jacobian, and metadata files for one block."""

	index: int
	sample_count: int
	summary_path: Path
	jacobians_path: Path
	metadata_path: Path


@dataclass(slots=True)
class _CompletedStep:
	"""Matrix data retained until cadence selection or finalization."""

	step_index: int
	time: float
	duration: float
	state: np.ndarray
	local_jacobians: np.ndarray
	accumulated_jacobians: np.ndarray


@dataclass(slots=True)
class _BufferedSample:
	"""One persisted scalar record with its per-particle arrays."""

	record: MidpointBM4SymplecticityRecord
	state: np.ndarray
	local_jacobians: np.ndarray
	accumulated_jacobians: np.ndarray
	local_relative_defects: np.ndarray
	accumulated_relative_defects: np.ndarray
	local_determinant_errors: np.ndarray
	accumulated_determinant_errors: np.ndarray


def _positive_integer(value: int, name: str) -> int:
	"""Normalize one positive observer control."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _particle_symplecticity_metrics(
	jacobians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return relative canonical defects and determinant errors per particle."""
	values = np.asarray(jacobians, dtype=float)
	if (
		values.ndim != 3
		or values.shape[0] == 0
		or values.shape[1:] != (2, 2)
		or not np.all(np.isfinite(values))
	):
		raise ValueError("Particle Jacobians must be finite with shape (N, 2, 2).")
	form = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
	defects = np.swapaxes(values, -1, -2) @ form @ values - form
	relative_defects = np.linalg.norm(defects, axis=(-2, -1)) / np.sqrt(2.0)
	determinant_errors = np.abs(np.linalg.det(values) - 1.0)
	if not (
		np.all(np.isfinite(relative_defects))
		and np.all(np.isfinite(determinant_errors))
	):
		raise ValueError("The midpoint-BM4 symplecticity metrics are non-finite.")
	return relative_defects, determinant_errors


class MidpointBM4SymplecticityObserver:
	"""Compose exact stage tangents and average GC symplecticity defects.

	The observer consumes all twelve uncoupled stages of every complete
	:class:`~simulation.MidpointBM4` step. For each independent particle it
	propagates a ``4 x 2`` tangent from diagonal embedding through the doubled
	composition, applies arithmetic-mean projection once, and advances the
	accumulated physical ``2 x 2`` tangent. Scalar records are arithmetic means
	over particles; the per-particle values remain available in persisted arrays.
	"""

	def __init__(
		self,
		*,
		dynamics: GuidingCenterJacobianSystem,
		notebook_path: str | Path,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "midpoint_bm4_symplecticity",
		record_every: int = 1,
		chunk_size: int = 64,
		verbose: bool = True,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure exact tangent propagation and notebook-derived output."""
		if not isinstance(dynamics, GuidingCenterJacobianSystem):
			raise TypeError(
				"MidpointBM4SymplecticityObserver requires "
				"GuidingCenterJacobianSystem dynamics."
			)
		if dynamics.state_dimension != 2:
			raise TypeError(
				"MidpointBM4SymplecticityObserver requires planar dynamics."
			)
		self.dynamics = dynamics
		self.output_directory = notebook_output_directory(
			notebook_path,
			project_root=project_root,
			run_date=run_date,
		)
		self.block_name = validate_block_name(block_name)
		self.record_every = _positive_integer(record_every, "record_every")
		self.chunk_size = _positive_integer(chunk_size, "chunk_size")
		self.verbose = bool(verbose)
		self.metadata = dict(metadata or {})

		self._particle_count: int | None = None
		self._embedding: np.ndarray | None = None
		self._projection: np.ndarray | None = None
		self._local_extended_tangents: np.ndarray | None = None
		self._accumulated_jacobians: np.ndarray | None = None
		self._expected_step = 0
		self._expected_stage = 0
		self._current_step_duration = 0.0
		self._stage_time_cursor: float | None = None
		self._previous_stage_after: np.ndarray | None = None
		self._last_recorded_step = -2
		self._last_completed: _CompletedStep | None = None
		self._closed = False
		self._records: list[MidpointBM4SymplecticityRecord] = []
		self._buffer: list[_BufferedSample] = []
		self._output_blocks: list[MidpointBM4SymplecticityOutputBlock] = []
		self._next_index = next_block_index(
			self.output_directory,
			self.block_name,
		)

	@property
	def particle_count(self) -> int | None:
		"""Return the particle count after the first observed stage."""
		return self._particle_count

	@property
	def records(self) -> tuple[MidpointBM4SymplecticityRecord, ...]:
		"""Return all averaged scalar records retained for analysis."""
		return tuple(self._records)

	@property
	def output_blocks(self) -> tuple[MidpointBM4SymplecticityOutputBlock, ...]:
		"""Return synchronized output groups written by this observer."""
		return tuple(self._output_blocks)

	def __enter__(self) -> MidpointBM4SymplecticityObserver:
		"""Open a context-managed observer session."""
		return self

	def __exit__(
		self,
		exception_type: type[BaseException] | None,
		exception: BaseException | None,
		_traceback: object,
	) -> None:
		"""Finalize normally without masking an active integration exception."""
		if exception_type is None:
			self.close()
			return
		try:
			self._flush_complete_samples()
		except Exception as cleanup_error:  # pragma: no cover - exceptional I/O
			if exception is not None:
				exception.add_note(
					"Midpoint-BM4 observer cleanup also failed: "
					f"{cleanup_error!r}"
				)
		finally:
			self._closed = True

	def __call__(self, stage: IntegrationStage) -> None:
		"""Advance exact tangents with one consecutive midpoint-BM4 stage."""
		if self._closed:
			raise RuntimeError(
				"This midpoint-BM4 symplecticity observer is already closed."
			)
		if not isinstance(stage, IntegrationStage):
			raise TypeError(
				"MidpointBM4SymplecticityObserver requires IntegrationStage data."
			)
		if stage.method_name != _METHOD_NAME:
			raise TypeError(
				"MidpointBM4SymplecticityObserver only supports MidpointBM4 stages."
			)
		if stage.dynamics is not self.dynamics:
			raise TypeError(
				"The observed stages must expose the exact configured dynamics instance."
			)
		if stage.dynamics_name != type(self.dynamics).__name__:
			raise TypeError(
				"The observed stage dynamics do not match the configured dynamics."
			)
		if stage.formulation_name != "GCStageProjectedFormulation":
			raise TypeError(
				"MidpointBM4SymplecticityObserver requires uncoupled GC stages."
			)
		expected_flow = (
			"adjoint_flow" if self._expected_stage % 2 == 0 else "flow"
		)
		if (
			stage.step_index != self._expected_step
			or stage.stage_index != self._expected_stage
			or stage.flow_name != expected_flow
		):
			raise ValueError(
				"Midpoint-BM4 stages must be observed consecutively in alternating "
				"adjoint/direct order."
			)

		factor = midpoint_bm4_stage_particle_jacobians(stage, self.dynamics)
		particle_count = factor.shape[0]
		if self._particle_count is None:
			self._initialize(stage, particle_count)
		elif particle_count != self._particle_count:
			raise ValueError("The particle count changed during observation.")
		if stage.stage_index == 0:
			self._validate_step_start(stage)
			self._current_step_duration = 0.0
		else:
			self._validate_stage_continuity(stage)
		self._validate_stage_time(stage)

		assert self._local_extended_tangents is not None
		self._current_step_duration += float(stage.duration)
		self._local_extended_tangents = (
			factor @ self._local_extended_tangents
		)
		assert self._stage_time_cursor is not None
		self._stage_time_cursor += float(stage.duration)
		self._previous_stage_after = np.asarray(stage.state_after, dtype=float).copy()

		if stage.stage_index == MIDPOINT_BM4_STAGE_COUNT - 1:
			self._complete_step(stage)
			self._expected_step += 1
			self._expected_stage = 0
			self._stage_time_cursor = None
			self._previous_stage_after = None
		else:
			self._expected_stage += 1

	def _initialize(self, stage: IntegrationStage, particle_count: int) -> None:
		"""Create batched embedding and identity tangents at the initial state."""
		self._particle_count = particle_count
		identity = np.eye(2)
		self._embedding = np.vstack((identity, identity))
		self._projection = np.hstack((identity, identity)) / 2.0
		self._local_extended_tangents = np.broadcast_to(
			self._embedding,
			(particle_count, 4, 2),
		).copy()
		self._accumulated_jacobians = np.broadcast_to(
			identity,
			(particle_count, 2, 2),
		).copy()
		self._validate_diagonal_input(stage.state_before)
		initial_state = self._projected_state(stage.state_before)
		self._append_sample(
			_CompletedStep(
				step_index=-1,
				time=float(stage.time),
				duration=0.0,
				state=initial_state,
				local_jacobians=self._accumulated_jacobians.copy(),
				accumulated_jacobians=self._accumulated_jacobians.copy(),
			)
		)

	def _validate_diagonal_input(self, state: np.ndarray) -> None:
		"""Require each complete step to start from physical diagonal embedding."""
		value = np.asarray(state, dtype=float)
		if self._particle_count is None or value.size != 4 * self._particle_count:
			raise ValueError("The midpoint-BM4 doubled state has an invalid size.")
		physical_size = 2 * self._particle_count
		if not np.allclose(
			value[:physical_size],
			value[physical_size:],
			rtol=0.0,
			atol=1e-13,
		):
			raise ValueError(
				"Every midpoint-BM4 step must start from diagonal embedding."
			)

	def _validate_step_start(self, stage: IntegrationStage) -> None:
		"""Require the next cycle to start from the prior projected endpoint."""
		self._validate_diagonal_input(stage.state_before)
		state_before = np.asarray(stage.state_before, dtype=float)
		if self._last_completed is not None:
			expected = np.concatenate(
				(self._last_completed.state, self._last_completed.state)
			)
			if not np.allclose(
				state_before,
				expected,
				rtol=0.0,
				atol=1e-13,
			):
				raise ValueError(
					"A midpoint-BM4 cycle did not start from the prior projection."
				)
			tolerance = float(
				64.0
				* np.finfo(float).eps
				* max(1.0, abs(self._last_completed.time), abs(stage.time))
			)
			if not np.isclose(
				stage.time,
				self._last_completed.time,
				rtol=0.0,
				atol=tolerance,
			):
				raise ValueError(
					"Consecutive midpoint-BM4 cycle times are inconsistent."
				)
		self._stage_time_cursor = float(stage.time)
		self._previous_stage_after = None

	def _validate_stage_continuity(self, stage: IntegrationStage) -> None:
		"""Require adjacent stage snapshots to describe one state sequence."""
		if self._previous_stage_after is None or not np.array_equal(
			np.asarray(stage.state_before, dtype=float),
			self._previous_stage_after,
		):
			raise ValueError(
				"Midpoint-BM4 stage state snapshots are not continuous."
			)

	def _validate_stage_time(self, stage: IntegrationStage) -> None:
		"""Check the fixed-time convention used by the emitted stage map."""
		if self._stage_time_cursor is None:
			raise RuntimeError("The midpoint-BM4 stage time cursor is unavailable.")
		expected = (
			self._stage_time_cursor + stage.duration
			if stage.flow_name == "flow"
			else self._stage_time_cursor
		)
		tolerance = float(
			64.0
			* np.finfo(float).eps
			* max(
				1.0,
				abs(self._stage_time_cursor),
				abs(expected),
				abs(stage.time),
			)
		)
		if not np.isclose(
			stage.time,
			expected,
			rtol=0.0,
			atol=tolerance,
		):
			raise ValueError(
				"Midpoint-BM4 stage evaluation times are inconsistent."
			)

	def _projected_state(self, state: np.ndarray) -> np.ndarray:
		"""Project one doubled component-major state by the arithmetic mean."""
		value = np.asarray(state, dtype=float)
		assert self._particle_count is not None
		physical_size = 2 * self._particle_count
		if value.shape != (2 * physical_size,) or not np.all(np.isfinite(value)):
			raise ValueError("The midpoint-BM4 doubled state is invalid.")
		return np.asarray(
			(value[:physical_size] + value[physical_size:]) / 2.0,
			dtype=float,
		)

	def _complete_step(self, stage: IntegrationStage) -> None:
		"""Project the local tangent and advance the accumulated physical map."""
		assert self._projection is not None
		assert self._embedding is not None
		assert self._local_extended_tangents is not None
		assert self._accumulated_jacobians is not None
		assert self._particle_count is not None
		local_jacobians = self._projection @ self._local_extended_tangents
		self._accumulated_jacobians = (
			local_jacobians @ self._accumulated_jacobians
		)
		completed = _CompletedStep(
			step_index=stage.step_index,
			time=float(stage.time),
			duration=float(self._current_step_duration),
			state=self._projected_state(stage.state_after),
			local_jacobians=np.asarray(local_jacobians, dtype=float).copy(),
			accumulated_jacobians=self._accumulated_jacobians.copy(),
		)
		self._last_completed = completed
		if (stage.step_index + 1) % self.record_every == 0:
			self._append_sample(completed)
		self._local_extended_tangents = np.broadcast_to(
			self._embedding,
			(self._particle_count, 4, 2),
		).copy()

	def _append_sample(self, sample: _CompletedStep) -> None:
		"""Calculate per-particle metrics and buffer one selected flow time."""
		local_defects, local_determinant_errors = (
			_particle_symplecticity_metrics(sample.local_jacobians)
		)
		accumulated_defects, accumulated_determinant_errors = (
			_particle_symplecticity_metrics(sample.accumulated_jacobians)
		)
		assert self._particle_count is not None
		record = MidpointBM4SymplecticityRecord(
			observation_index=len(self._records),
			step_index=sample.step_index,
			time=sample.time,
			duration=sample.duration,
			particle_count=self._particle_count,
			mean_local_relative_defect=float(np.mean(local_defects)),
			std_local_relative_defect=float(np.std(local_defects)),
			max_local_relative_defect=float(np.max(local_defects)),
			mean_accumulated_relative_defect=float(
				np.mean(accumulated_defects)
			),
			std_accumulated_relative_defect=float(
				np.std(accumulated_defects)
			),
			max_accumulated_relative_defect=float(
				np.max(accumulated_defects)
			),
			mean_local_determinant_error=float(
				np.mean(local_determinant_errors)
			),
			mean_accumulated_determinant_error=float(
				np.mean(accumulated_determinant_errors)
			),
		)
		self._records.append(record)
		self._buffer.append(
			_BufferedSample(
				record=record,
				state=sample.state.copy(),
				local_jacobians=sample.local_jacobians.copy(),
				accumulated_jacobians=sample.accumulated_jacobians.copy(),
				local_relative_defects=local_defects.copy(),
				accumulated_relative_defects=accumulated_defects.copy(),
				local_determinant_errors=local_determinant_errors.copy(),
				accumulated_determinant_errors=(
					accumulated_determinant_errors.copy()
				),
			)
		)
		self._last_recorded_step = sample.step_index
		if self.verbose:
			print(
				f"[midpoint-bm4-symplecticity] step={record.step_index:05d} "
				f"t={record.time:.6g} "
				f"mean_local={record.mean_local_relative_defect:.3e} "
				f"mean_flow={record.mean_accumulated_relative_defect:.3e}"
			)
		if len(self._buffer) >= self.chunk_size:
			self.flush()

	def flush(self) -> MidpointBM4SymplecticityOutputBlock | None:
		"""Persist the current scalar and per-particle array buffer."""
		if not self._buffer:
			return None
		block_index = self._next_index
		paths = write_diagnostic_block(
			output_directory=self.output_directory,
			block_name=self.block_name,
			block_index=block_index,
			rows=[asdict(sample.record) for sample in self._buffer],
			arrays={
				"states": np.stack([sample.state for sample in self._buffer]),
				"local_jacobians": np.stack(
					[sample.local_jacobians for sample in self._buffer]
				),
				"accumulated_jacobians": np.stack(
					[sample.accumulated_jacobians for sample in self._buffer]
				),
				"local_relative_defects": np.stack(
					[sample.local_relative_defects for sample in self._buffer]
				),
				"accumulated_relative_defects": np.stack(
					[
						sample.accumulated_relative_defects
						for sample in self._buffer
					]
				),
				"local_determinant_errors": np.stack(
					[sample.local_determinant_errors for sample in self._buffer]
				),
				"accumulated_determinant_errors": np.stack(
					[
						sample.accumulated_determinant_errors
						for sample in self._buffer
					]
				),
				"observation_indices": np.asarray(
					[sample.record.observation_index for sample in self._buffer],
					dtype=int,
				),
				"step_indices": np.asarray(
					[sample.record.step_index for sample in self._buffer],
					dtype=int,
				),
				"times": np.asarray(
					[sample.record.time for sample in self._buffer],
					dtype=float,
				),
				"durations": np.asarray(
					[sample.record.duration for sample in self._buffer],
					dtype=float,
				),
			},
			metadata={
				"objective": (
					"Per-particle physical-flow symplecticity of midpoint BM4"
				),
				"method": _METHOD_NAME,
				"jacobian_method": _JACOBIAN_METHOD,
				"bm4_stage_count": MIDPOINT_BM4_STAGE_COUNT,
				"projection": "arithmetic_mean_once_per_complete_step",
				"particle_aggregation": "arithmetic_mean",
				"particle_count": self._particle_count,
				"record_every_complete_steps": self.record_every,
				"metadata": self.metadata,
			},
		)
		block = MidpointBM4SymplecticityOutputBlock(
			index=block_index,
			sample_count=len(self._buffer),
			summary_path=paths.summary,
			jacobians_path=paths.arrays,
			metadata_path=paths.metadata,
		)
		self._output_blocks.append(block)
		self._buffer.clear()
		self._next_index += 1
		return block

	def close(self) -> None:
		"""Record the final complete step, flush it, and reject later stages."""
		if self._closed:
			return
		if (
			self._expected_stage != 0
			or self._previous_stage_after is not None
			or self._stage_time_cursor is not None
		):
			raise RuntimeError(
				"Cannot close the observer during an incomplete midpoint-BM4 cycle."
			)
		self._flush_complete_samples()
		self._closed = True

	def _flush_complete_samples(self) -> None:
		"""Persist every complete sample while ignoring any partial tangent."""
		if (
			self._last_completed is not None
			and self._last_completed.step_index != self._last_recorded_step
		):
			self._append_sample(self._last_completed)
		self.flush()


__all__ = [
	"MidpointBM4SymplecticityObserver",
	"MidpointBM4SymplecticityOutputBlock",
	"MidpointBM4SymplecticityRecord",
]
