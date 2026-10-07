"""Persist exact per-particle symplecticity for complete physical steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import numpy as np

from diagnostics._validation import positive_integer as _positive_integer

from diagnostics.buffering import DiagnosticBuffer
from diagnostics.paths import (
	notebook_output_directory,
	validate_block_name,
)
from dynamics import GuidingCenterJacobianSystem
from initial_conditions import GCInitialConfiguration
from contracts.observation import IntegrationStep


TrajectoryJacobianCalculator: TypeAlias = Callable[[IntegrationStep], np.ndarray]


@dataclass(frozen=True, slots=True)
class TrajectorySymplecticityRecord:
	"""Arithmetic trajectory-mean defects at one physical-flow time."""

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
class TrajectorySymplecticityOutputBlock:
	"""Synchronized scalar, matrix, and metadata files for one chunk."""

	index: int
	sample_count: int
	summary_path: Path
	jacobians_path: Path
	metadata_path: Path


@dataclass(slots=True)
class _CompletedStep:
	"""One complete tangent retained until cadence selection or close."""

	step_index: int
	time: float
	duration: float
	state: np.ndarray
	local_jacobians: np.ndarray
	accumulated_jacobians: np.ndarray


@dataclass(slots=True)
class _BufferedSample:
	"""One selected record and its per-particle matrix arrays."""

	record: TrajectorySymplecticityRecord
	state: np.ndarray
	local_jacobians: np.ndarray
	accumulated_jacobians: np.ndarray
	local_relative_defects: np.ndarray
	accumulated_relative_defects: np.ndarray
	local_determinant_errors: np.ndarray
	accumulated_determinant_errors: np.ndarray


def _particle_metrics(jacobians: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Return canonical relative defects and determinant errors per particle."""
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
	relative = np.linalg.norm(defects, axis=(-2, -1)) / np.sqrt(2.0)
	determinants = np.abs(np.linalg.det(values) - 1.0)
	if not np.all(np.isfinite(relative)) or not np.all(np.isfinite(determinants)):
		raise ValueError("The trajectory symplecticity metrics are non-finite.")
	return relative, determinants


def _validated_observer_initial_state(
	dynamics: GuidingCenterJacobianSystem,
	initial_configuration: GCInitialConfiguration,
	method_name: str,
	jacobian_method: str,
	jacobian_calculator: TrajectoryJacobianCalculator,
) -> np.ndarray:
	"""Check observer capabilities and return the configured packed initial state."""
	if not isinstance(dynamics, GuidingCenterJacobianSystem):
		raise TypeError(
			"GCTrajectorySymplecticityObserver requires exact GC Jacobians."
		)
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be a GC configuration.")
	initial_state = initial_configuration.initial_state
	if initial_state is None:
		raise ValueError("The GC initial configuration has no initial state.")
	if not isinstance(method_name, str) or not method_name.strip():
		raise ValueError("`method_name` must be a non-empty string.")
	if not isinstance(jacobian_method, str) or not jacobian_method.strip():
		raise ValueError("`jacobian_method` must be a non-empty string.")
	if not callable(jacobian_calculator):
		raise TypeError("`jacobian_calculator` must be callable.")
	return initial_state


class GCTrajectorySymplecticityObserver:
	"""Accumulate exact independent planar tangents and average their defects."""

	def __init__(
		self,
		*,
		dynamics: GuidingCenterJacobianSystem,
		initial_configuration: GCInitialConfiguration,
		method_name: str,
		jacobian_method: str,
		jacobian_calculator: TrajectoryJacobianCalculator,
		notebook_path: str | Path,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "trajectory_symplecticity",
		record_every: int = 1,
		chunk_size: int = 64,
		verbose: bool = True,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure one exact complete-step tangent stream."""
		initial_state = _validated_observer_initial_state(
			dynamics, initial_configuration, method_name, jacobian_method, jacobian_calculator,
		)

		self.dynamics = dynamics
		self.initial_configuration = initial_configuration
		self.initial_state = np.asarray(initial_state, dtype=float).copy()
		self.particle_count = initial_configuration.layout.particle_count(initial_state)
		self.method_name = method_name
		self.jacobian_method = jacobian_method
		self.jacobian_calculator = jacobian_calculator
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
		self._accumulated = np.broadcast_to(
			np.eye(2),
			(self.particle_count, 2, 2),
		).copy()
		self._expected_step = 0
		self._last_state = self.initial_state.copy()
		self._last_time: float | None = None
		self._last_completed: _CompletedStep | None = None
		self._last_recorded_step = -2
		self._records: list[TrajectorySymplecticityRecord] = []
		self._output = DiagnosticBuffer[_BufferedSample, TrajectorySymplecticityOutputBlock](
			output_directory=self.output_directory,
			block_name=self.block_name,
			chunk_size=self.chunk_size,
			write_block=self._write_block,
		)

	@property
	def records(self) -> tuple[TrajectorySymplecticityRecord, ...]:
		"""Return all selected arithmetic-mean scalar records."""
		return tuple(self._records)

	@property
	def output_blocks(self) -> tuple[TrajectorySymplecticityOutputBlock, ...]:
		"""Return all persisted output chunks."""
		return self._output.blocks

	def __enter__(self) -> GCTrajectorySymplecticityObserver:
		"""Open a context-managed observation stream."""
		return self

	def __exit__(
		self,
		exception_type: type[BaseException] | None,
		exception: BaseException | None,
		_traceback: object,
	) -> None:
		"""Flush output without replacing an active integration exception."""
		self._output.exit(exception_type, exception, self._record_final_sample)

	def _validated_continuous_states(
		self, step: IntegrationStep,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Read finite physical states and require continuity with the previous step."""
		state_before = self._validated_state(step.state_before)
		state_after = self._validated_state(step.state_after)
		if not np.array_equal(state_before, self._last_state):
			raise ValueError("Observed complete-step states are not continuous.")
		if self._last_time is not None:
			tolerance = float(
				64.0
				* np.finfo(float).eps
				* max(1.0, abs(self._last_time), abs(step.start_time))
			)
			if not np.isclose(
				step.start_time,
				self._last_time,
				rtol=0.0,
				atol=tolerance,
			):
				raise ValueError("Observed complete-step times are not continuous.")
		return state_before, state_after

	def __call__(self, step: IntegrationStep) -> None:
		"""Advance the physical tangent with one consecutive complete step."""
		if self._output.closed:
			raise RuntimeError("This trajectory symplecticity observer is closed.")
		if not isinstance(step, IntegrationStep):
			raise TypeError("The observer requires IntegrationStep data.")
		if step.method_name != self.method_name:
			raise TypeError(
				f"Expected {self.method_name} steps, received {step.method_name}."
			)
		if step.dynamics is not self.dynamics:
			raise TypeError(
				"The observed step must expose the exact configured dynamics instance."
			)
		if step.step_index != self._expected_step:
			raise ValueError("Complete steps must be observed consecutively.")
		state_before, state_after = self._validated_continuous_states(step)
		if not self._records:
			self._append_sample(
				_CompletedStep(
					step_index=-1,
					time=float(step.start_time),
					duration=0.0,
					state=state_before.copy(),
					local_jacobians=self._accumulated.copy(),
					accumulated_jacobians=self._accumulated.copy(),
				)
			)

		local = np.asarray(self.jacobian_calculator(step), dtype=float)
		expected_shape = (self.particle_count, 2, 2)
		if local.shape != expected_shape or not np.all(np.isfinite(local)):
			raise ValueError(
				"The exact step Jacobian calculator must return finite blocks with "
				f"shape {expected_shape}."
			)
		self._accumulated = local @ self._accumulated
		completed = _CompletedStep(
			step_index=step.step_index,
			time=float(step.time),
			duration=float(step.duration),
			state=state_after.copy(),
			local_jacobians=local.copy(),
			accumulated_jacobians=self._accumulated.copy(),
		)
		self._last_completed = completed
		if (step.step_index + 1) % self.record_every == 0:
			self._append_sample(completed)
		self._last_state = state_after.copy()
		self._last_time = float(step.time)
		self._expected_step += 1

	def _validated_state(self, state: np.ndarray) -> np.ndarray:
		"""Require a finite physical component-major state with 2N entries."""
		value = np.asarray(state, dtype=float)
		if (
			value.shape != self.initial_state.shape
			or not np.all(np.isfinite(value))
		):
			raise ValueError("Observed physical GC states must be finite 2N vectors.")
		return value

	def _append_sample(self, sample: _CompletedStep) -> None:
		"""Calculate trajectory metrics and buffer one selected sample."""
		local_defects, local_determinants = _particle_metrics(
			sample.local_jacobians
		)
		flow_defects, flow_determinants = _particle_metrics(
			sample.accumulated_jacobians
		)
		record = TrajectorySymplecticityRecord(
			observation_index=len(self._records),
			step_index=sample.step_index,
			time=sample.time,
			duration=sample.duration,
			particle_count=self.particle_count,
			mean_local_relative_defect=float(np.mean(local_defects)),
			std_local_relative_defect=float(np.std(local_defects)),
			max_local_relative_defect=float(np.max(local_defects)),
			mean_accumulated_relative_defect=float(np.mean(flow_defects)),
			std_accumulated_relative_defect=float(np.std(flow_defects)),
			max_accumulated_relative_defect=float(np.max(flow_defects)),
			mean_local_determinant_error=float(np.mean(local_determinants)),
			mean_accumulated_determinant_error=float(
				np.mean(flow_determinants)
			),
		)
		self._records.append(record)
		self._output.append(
			_BufferedSample(
				record=record,
				state=sample.state.copy(),
				local_jacobians=sample.local_jacobians.copy(),
				accumulated_jacobians=sample.accumulated_jacobians.copy(),
				local_relative_defects=local_defects.copy(),
				accumulated_relative_defects=flow_defects.copy(),
				local_determinant_errors=local_determinants.copy(),
				accumulated_determinant_errors=flow_determinants.copy(),
			)
		)
		self._last_recorded_step = sample.step_index
		if self.verbose:
			print(
				f"[trajectory-symplecticity] {self.method_name} "
				f"step={record.step_index:05d} t={record.time:.6g} "
				f"mean_local={record.mean_local_relative_defect:.3e} "
				f"mean_flow={record.mean_accumulated_relative_defect:.3e}"
			)
		self._output.flush_if_full()

	def flush(self) -> TrajectorySymplecticityOutputBlock | None:
		"""Write pending samples through the shared diagnostic buffer."""
		return self._output.flush()

	def _write_block(
		self, index: int, samples: tuple[_BufferedSample, ...],
	) -> TrajectorySymplecticityOutputBlock:
		"""Persist the pending scalar records and particle arrays."""
		paths = self._output.write(
			block_index=index,
			rows=[asdict(sample.record) for sample in samples],
			arrays={
				"states": np.stack([sample.state for sample in samples]),
				"local_jacobians": np.stack(
					[sample.local_jacobians for sample in samples]
				),
				"accumulated_jacobians": np.stack(
					[sample.accumulated_jacobians for sample in samples]
				),
				"local_relative_defects": np.stack(
					[sample.local_relative_defects for sample in samples]
				),
				"accumulated_relative_defects": np.stack(
					[sample.accumulated_relative_defects for sample in samples]
				),
				"local_determinant_errors": np.stack(
					[sample.local_determinant_errors for sample in samples]
				),
				"accumulated_determinant_errors": np.stack(
					[
						sample.accumulated_determinant_errors
						for sample in samples
					]
				),
				"observation_indices": np.asarray(
					[sample.record.observation_index for sample in samples],
					dtype=int,
				),
				"step_indices": np.asarray(
					[sample.record.step_index for sample in samples],
					dtype=int,
				),
				"times": np.asarray(
					[sample.record.time for sample in samples],
					dtype=float,
				),
				"durations": np.asarray(
					[sample.record.duration for sample in samples],
					dtype=float,
				),
			},
			metadata={
				"objective": "Per-trajectory physical-flow symplecticity",
				"method": self.method_name,
				"jacobian_method": self.jacobian_method,
				"particle_aggregation": "arithmetic_mean",
				"particle_count": self.particle_count,
				"record_every_complete_steps": self.record_every,
				"metadata": self.metadata,
			},
		)
		block = TrajectorySymplecticityOutputBlock(
			index=index,
			sample_count=len(samples),
			summary_path=paths.summary,
			jacobians_path=paths.arrays,
			metadata_path=paths.metadata,
		)
		return block

	def close(self) -> None:
		"""Flush the final partial block and reject further events."""
		self._output.close(self._record_final_sample)

	def _record_final_sample(self) -> None:
		"""Queue the final completed step when cadence did not select it."""
		if (
			self._last_completed is not None
			and self._last_completed.step_index != self._last_recorded_step
		):
			self._append_sample(self._last_completed)


__all__ = [
	"GCTrajectorySymplecticityObserver",
	"TrajectoryJacobianCalculator",
	"TrajectorySymplecticityOutputBlock",
	"TrajectorySymplecticityRecord",
]
