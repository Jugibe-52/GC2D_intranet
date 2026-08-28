"""Physical GC area and symplecticity diagnostics for complete numerical steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from initial_conditions import Area
from simulation import ImplicitABBAIntegrationStep, IntegrationStep
from diagnostics.output import write_diagnostic_block

from diagnostics.jacobians import (
	STEP_JACOBIAN_METHODS,
	StepJacobianMethod,
	calculate_step_jacobian,
)
from .observer import gc_physical_symplectic_form
from diagnostics.paths import (
	next_block_index,
	notebook_output_directory,
	validate_block_name,
)


@dataclass(frozen=True, slots=True)
class GCAreaSymplecticityRecord:
	"""Area, local-step and accumulated-flow diagnostics at one physical state."""

	observation_index: int
	step_index: int
	time: float
	duration: float
	particle_count: int
	signed_area: float
	absolute_area_error: float
	relative_area_error: float
	local_determinant_error: float
	local_relative_defect: float
	determinant: float
	determinant_error: float
	log_abs_determinant: float
	condition_number: float
	defect_frobenius: float
	relative_defect: float
	max_abs_defect: float


@dataclass(frozen=True, slots=True)
class GCAreaSymplecticityOutputBlock:
	"""Synchronized scalar, state and Jacobian files for one buffered block."""

	index: int
	sample_count: int
	summary_path: Path
	jacobians_path: Path
	metadata_path: Path


@dataclass(slots=True)
class _BufferedSample:
	"""Matrix-valued data retained until the next output flush."""

	record: GCAreaSymplecticityRecord
	state: np.ndarray
	local_jacobian: np.ndarray
	accumulated_jacobian: np.ndarray


class GCAreaSymplecticityObserver:
	"""Propagate the Jacobian of a physical GC numerical flow.

	Each complete numerical step is differentiated by the configured diagnostic
	method. The selected matrix advances the accumulated derivative of the discrete
	flow from the initial condition. ``record_every`` controls persistence only:
	every step Jacobian still contributes to the accumulated tangent.
	"""

	def __init__(
		self,
		*,
		notebook_path: str | Path,
		area: Area,
		period: float | None = None,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "gc_area_symplecticity",
		record_every: int = 1,
		chunk_size: int = 16,
		relative_step: float | None = None,
		jacobian_method: StepJacobianMethod = "finite_difference",
		verbose: bool = True,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure physical GC diagnostics and notebook-derived persistence."""
		if not isinstance(area, Area):
			raise TypeError("`area` must be an Area instance.")
		if (
			isinstance(record_every, (bool, np.bool_))
			or not isinstance(record_every, (int, np.integer))
			or record_every < 1
		):
			raise ValueError("`record_every` must be a positive integer.")
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
		if jacobian_method not in STEP_JACOBIAN_METHODS:
			raise ValueError(
				"`jacobian_method` must be 'finite_difference', "
				"'implicit_function', or 'stage_increment'."
			)
		if jacobian_method != "finite_difference" and relative_step is not None:
			raise ValueError(
				"`relative_step` is used only when "
				"`jacobian_method='finite_difference'`."
			)

		initial_state = area.initial_state
		assert initial_state is not None
		self.area = area
		self.initial_state = np.asarray(initial_state, dtype=float).copy()
		self.particle_count = area.layout.particle_count(initial_state)
		self.physical_size = 2 * self.particle_count
		self.period = period
		self.form = gc_physical_symplectic_form(self.particle_count)
		self.form_norm = float(np.linalg.norm(self.form, ord="fro"))
		self.initial_area = float(area.calculate_area(initial_state, period=period))
		if (
			not np.isfinite(self.initial_area)
			or abs(self.initial_area) <= np.finfo(float).eps
		):
			raise ValueError("The initial boundary must have non-zero finite area.")

		self.output_directory = notebook_output_directory(
			notebook_path,
			project_root=project_root,
			run_date=run_date,
		)
		self.block_name = validate_block_name(block_name)
		self.record_every = int(record_every)
		self.chunk_size = int(chunk_size)
		self.relative_step = relative_step
		self.jacobian_method = jacobian_method
		self.verbose = bool(verbose)
		self.metadata = dict(metadata or {})
		self._accumulated_jacobian = np.eye(self.physical_size)
		self._expected_step = 0
		self._initialized = False
		self._closed = False
		self._last_recorded_step = -2
		self._last_completed: (
			tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray] | None
		) = None
		self._records: list[GCAreaSymplecticityRecord] = []
		self._buffer: list[_BufferedSample] = []
		self._output_blocks: list[GCAreaSymplecticityOutputBlock] = []
		self._next_index = next_block_index(self.output_directory, self.block_name)

	@property
	def records(self) -> tuple[GCAreaSymplecticityRecord, ...]:
		"""Return scalar diagnostics retained for interactive analysis."""
		return tuple(self._records)

	@property
	def output_blocks(self) -> tuple[GCAreaSymplecticityOutputBlock, ...]:
		"""Return synchronized file groups written during this run."""
		return tuple(self._output_blocks)

	def __enter__(self) -> GCAreaSymplecticityObserver:
		"""Use the observer as a context manager so its final block is flushed."""
		return self

	def __exit__(self, *_exception: object) -> None:
		"""Flush observations even when the surrounding integration raises."""
		self.close()

	def __call__(self, step: IntegrationStep) -> None:
		"""Advance local and accumulated diagnostics with one consecutive step."""
		if self._closed:
			raise RuntimeError("This GC area symplecticity observer is already closed.")
		if step.dynamics_name != "GuidingCenterDynamics":
			raise TypeError(
				"GCAreaSymplecticityObserver only supports guiding-centre dynamics."
			)
		if step.step_index != self._expected_step:
			raise ValueError("Numerical steps must be observed consecutively.")

		state_before = self._validated_state(step.state_before)
		state_after = self._validated_state(step.state_after)
		if not self._initialized:
			if not np.allclose(
				state_before,
				self.initial_state,
				rtol=0.0,
				atol=1e-13,
			):
				raise ValueError(
					"The observed problem and `area` must share the initial state."
				)
			identity = np.eye(self.physical_size)
			self._append_record(
				step_index=-1,
				time=(
					step.start_time
					if isinstance(step, ImplicitABBAIntegrationStep)
					else step.time - step.duration
				),
				duration=0.0,
				state=state_before,
				local_jacobian=identity,
				accumulated_jacobian=identity,
			)
			self._initialized = True

		local_jacobian = calculate_step_jacobian(
			step,
			method=self.jacobian_method,
			relative_step=self.relative_step,
		)
		self._accumulated_jacobian = (
			local_jacobian @ self._accumulated_jacobian
		)
		self._last_completed = (
			step.step_index,
			step.time,
			step.duration,
			state_after.copy(),
			local_jacobian.copy(),
			self._accumulated_jacobian.copy(),
		)
		if (step.step_index + 1) % self.record_every == 0:
			self._append_record(
				step_index=step.step_index,
				time=step.time,
				duration=step.duration,
				state=state_after,
				local_jacobian=local_jacobian,
				accumulated_jacobian=self._accumulated_jacobian,
			)
		self._expected_step += 1

	def _validated_state(self, state: np.ndarray) -> np.ndarray:
		"""Require a finite physical GC state with component-major layout."""
		value = np.asarray(state, dtype=float)
		if (
			value.ndim != 1
			or value.size != self.physical_size
			or not np.all(np.isfinite(value))
		):
			raise ValueError(
				"Physical GC diagnostics require the finite 2N state; "
				"run the method with `track_energy=False`."
			)
		return value

	def _append_record(
		self,
		*,
		step_index: int,
		time: float,
		duration: float,
		state: np.ndarray,
		local_jacobian: np.ndarray,
		accumulated_jacobian: np.ndarray,
	) -> None:
		"""Calculate scalar diagnostics and buffer one physical observation."""
		area = float(self.area.calculate_area(state, period=self.period))
		area_error = area - self.initial_area
		local_defect = local_jacobian.T @ self.form @ local_jacobian - self.form
		defect = (
			accumulated_jacobian.T @ self.form @ accumulated_jacobian - self.form
		)
		local_sign, local_log_abs_determinant = np.linalg.slogdet(local_jacobian)
		sign, log_abs_determinant = np.linalg.slogdet(accumulated_jacobian)
		local_determinant = float(
			local_sign * np.exp(local_log_abs_determinant)
		)
		determinant = float(sign * np.exp(log_abs_determinant))
		defect_frobenius = float(np.linalg.norm(defect, ord="fro"))
		record = GCAreaSymplecticityRecord(
			observation_index=len(self._records),
			step_index=step_index,
			time=float(time),
			duration=float(duration),
			particle_count=self.particle_count,
			signed_area=area,
			absolute_area_error=area_error,
			relative_area_error=area_error / abs(self.initial_area),
			local_determinant_error=abs(local_determinant - 1.0),
			local_relative_defect=(
				float(np.linalg.norm(local_defect, ord="fro")) / self.form_norm
			),
			determinant=determinant,
			determinant_error=abs(determinant - 1.0),
			log_abs_determinant=float(log_abs_determinant),
			condition_number=float(np.linalg.cond(accumulated_jacobian)),
			defect_frobenius=defect_frobenius,
			relative_defect=defect_frobenius / self.form_norm,
			max_abs_defect=float(np.max(np.abs(defect))),
		)
		self._records.append(record)
		self._buffer.append(
			_BufferedSample(
				record=record,
				state=state.copy(),
				local_jacobian=local_jacobian.copy(),
				accumulated_jacobian=accumulated_jacobian.copy(),
			)
		)
		self._last_recorded_step = step_index
		if self.verbose:
			print(
				f"[gc-area-symplecticity] step={record.step_index:05d} "
				f"t={record.time:.6g} "
				f"local_defect={record.local_relative_defect:.3e} "
				f"flow_defect={record.relative_defect:.3e} "
				f"area_error={record.relative_area_error:+.3e}"
			)
		if len(self._buffer) >= self.chunk_size:
			self.flush()

	def flush(self) -> GCAreaSymplecticityOutputBlock | None:
		"""Write the current matrix buffer as one indexed output block."""
		if not self._buffer:
			return None
		block_index = self._next_index
		rows = [asdict(sample.record) for sample in self._buffer]
		paths = write_diagnostic_block(
			output_directory=self.output_directory,
			block_name=self.block_name,
			block_index=block_index,
			rows=rows,
			arrays={
				"local_jacobians": np.stack(
				[sample.local_jacobian for sample in self._buffer]
			),
				"accumulated_jacobians": np.stack(
				[sample.accumulated_jacobian for sample in self._buffer]
			),
				"states": np.stack([sample.state for sample in self._buffer]),
				"observation_indices": np.asarray(
				[sample.record.observation_index for sample in self._buffer],
				dtype=int,
			),
				"step_indices": np.asarray(
				[sample.record.step_index for sample in self._buffer],
				dtype=int,
			),
				"times": np.asarray(
					[sample.record.time for sample in self._buffer]
				),
			},
			metadata={
				"objective": "Physical GC numerical-flow symplecticity and area",
				"particle_count": self.particle_count,
				"initial_signed_area": self.initial_area,
				"period": self.period,
				"record_every_complete_steps": self.record_every,
				"step_jacobian_method": self.jacobian_method,
				"finite_difference_relative_step": (
				None
				if self.jacobian_method != "finite_difference"
				else (
					float(np.cbrt(np.finfo(float).eps))
					if self.relative_step is None
					else float(self.relative_step)
				)
				),
				"metadata": self.metadata,
			},
		)

		block = GCAreaSymplecticityOutputBlock(
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
		"""Record the final completed step, flush it and reject further events."""
		if self._closed:
			return
		if self._last_completed is not None:
			(
				step_index,
				time,
				duration,
				state,
				local_jacobian,
				accumulated_jacobian,
			) = self._last_completed
			if step_index != self._last_recorded_step:
				self._append_record(
					step_index=step_index,
					time=time,
					duration=duration,
					state=state,
					local_jacobian=local_jacobian,
					accumulated_jacobian=accumulated_jacobian,
				)
		self.flush()
		self._closed = True

__all__ = [
	"GCAreaSymplecticityObserver",
	"GCAreaSymplecticityOutputBlock",
	"GCAreaSymplecticityRecord",
]
