"""Study the physical projection of doubled GC trajectories and their area."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from initial_conditions import Area
from simulation import IntegrationStage
from diagnostics.output import write_diagnostic_block
from diagnostics.symplecticity import (
	central_difference_jacobian,
	gc_physical_symplectic_form,
)
from diagnostics.paths import (
	next_block_index,
	notebook_output_directory,
	validate_block_name,
)


_BM4_STAGE_COUNT = 12


def _positive_particle_count(particle_count: int) -> int:
	"""Normalize a particle count shared by the projection matrices."""
	if (
		isinstance(particle_count, (bool, np.bool_))
		or not isinstance(particle_count, (int, np.integer))
		or particle_count < 1
	):
		raise ValueError("`particle_count` must be a positive integer.")
	return int(particle_count)


def gc_diagonal_embedding(particle_count: int) -> np.ndarray:
	"""Return ``E: z -> (z, z)`` from physical to doubled GC coordinates."""
	physical_identity = np.eye(2 * _positive_particle_count(particle_count))
	return np.vstack((physical_identity, physical_identity))


def gc_average_projection(particle_count: int) -> np.ndarray:
	"""Return ``P: (z_first, z_second) -> (z_first + z_second) / 2``."""
	physical_identity = np.eye(2 * _positive_particle_count(particle_count))
	return np.hstack((physical_identity, physical_identity)) / 2


@dataclass(frozen=True, slots=True)
class ProjectedAreaRecord:
	"""Scalar projection, symplecticity and polygon-area data at one time."""

	observation_index: int
	step_index: int
	time: float
	particle_count: int
	signed_area: float
	absolute_area_error: float
	relative_area_error: float
	determinant: float
	determinant_error: float
	log_abs_determinant: float
	condition_number: float
	defect_frobenius: float
	relative_defect: float
	max_abs_defect: float
	copy_separation: float
	relative_copy_separation: float


@dataclass(frozen=True, slots=True)
class ProjectedAreaOutputBlock:
	"""Synchronized projected-summary, Jacobian and metadata files."""

	index: int
	sample_count: int
	summary_path: Path
	jacobians_path: Path
	metadata_path: Path


@dataclass(slots=True)
class _BufferedProjection:
	"""Matrix-valued projected data retained until the next output block."""

	record: ProjectedAreaRecord
	projected_state: np.ndarray
	projected_jacobian: np.ndarray


class ProjectedSymplecticityAreaObserver:
	"""Propagate ``D(P Phi E)`` and observe a projected GC boundary.

	``E`` embeds the physical initial state on the diagonal of the doubled phase
	space, each numerical stage Jacobian advances that tangent, and ``P`` reads
	the physical mean after every complete twelve-stage BM4 step. For a
	stage-projected method, each observed stage map already includes the
	projection and diagonal re-embedding applied after that map. The resulting
	square Jacobian maps the initial physical boundary coordinates to their
	current projection and can therefore be tested against the physical GC
	symplectic form.

	All stage Jacobians must be evaluated to propagate the cumulative tangent.
	``record_every`` reduces only persisted observations, not differentiation cost.
	The observer requires ``track_energy=False`` and an :class:`Area` trajectory.
	"""

	def __init__(
		self,
		*,
		notebook_path: str | Path,
		area: Area,
		period: float | None = None,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "projected_symplecticity",
		record_every: int = 1,
		chunk_size: int = 16,
		relative_step: float | None = None,
		verbose: bool = True,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure cumulative tangent propagation and notebook-derived output."""
		if not isinstance(area, Area):
			raise TypeError("`area` must be an Area trajectory.")
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

		initial_state = area.state
		assert initial_state is not None
		self.area = area
		self.initial_state = initial_state.copy()
		self.particle_count = area.particle_count(initial_state)
		self.physical_size = 2 * self.particle_count
		self.extended_size = 2 * self.physical_size
		self.period = period
		self.projection = gc_average_projection(self.particle_count)
		self.embedding = gc_diagonal_embedding(self.particle_count)
		self.physical_form = gc_physical_symplectic_form(self.particle_count)
		self._extended_tangent = self.embedding.copy()

		self.initial_area = float(area.calculate_area(initial_state, period=period))
		if not np.isfinite(self.initial_area) or abs(self.initial_area) <= np.finfo(float).eps:
			raise ValueError("The initial projected boundary must have non-zero finite area.")
		self.output_directory = notebook_output_directory(
			notebook_path,
			project_root=project_root,
			run_date=run_date,
		)
		self.block_name = validate_block_name(block_name)
		self.record_every = int(record_every)
		self.chunk_size = int(chunk_size)
		self.relative_step = relative_step
		self.verbose = bool(verbose)
		self.metadata = dict(metadata or {})

		self._expected_step = 0
		self._expected_stage = 0
		self._initialized = False
		self._closed = False
		self._last_completed: tuple[int, float, np.ndarray, np.ndarray] | None = None
		self._last_recorded_step = -2
		self._records: list[ProjectedAreaRecord] = []
		self._buffer: list[_BufferedProjection] = []
		self._output_blocks: list[ProjectedAreaOutputBlock] = []
		self._next_index = next_block_index(self.output_directory, self.block_name)

	@property
	def records(self) -> tuple[ProjectedAreaRecord, ...]:
		"""Return scalar observations retained for interactive analysis."""
		return tuple(self._records)

	@property
	def output_blocks(self) -> tuple[ProjectedAreaOutputBlock, ...]:
		"""Return all synchronized file blocks written by this observer."""
		return tuple(self._output_blocks)

	def __enter__(self) -> ProjectedSymplecticityAreaObserver:
		"""Use a context manager to persist the final partial block."""
		return self

	def __exit__(self, *_exception: object) -> None:
		"""Close the observer even when integration or diagnostics raise."""
		self.close()

	def __call__(self, stage: IntegrationStage) -> None:
		"""Advance the cumulative tangent with one consecutive BM4 stage."""
		if self._closed:
			raise RuntimeError("This projected symplecticity observer is already closed.")
		if stage.dynamics_name != "GuidingCenterDynamics":
			raise TypeError(
				"ProjectedSymplecticityAreaObserver only supports "
				"guiding-centre dynamics."
			)
		if (
			stage.step_index != self._expected_step
			or stage.stage_index != self._expected_stage
		):
			raise ValueError("Projection stages must be observed consecutively in BM4 order.")

		state_before = self._validated_extended_state(stage.state_before)
		state_after = self._validated_extended_state(stage.state_after)
		if not self._initialized:
			first = state_before[: self.physical_size]
			second = state_before[self.physical_size :]
			if not np.allclose(first, second, rtol=0.0, atol=1e-13):
				raise ValueError("The initial extended GC state must lie on the diagonal.")
			if not np.allclose(first, self.initial_state, rtol=0.0, atol=1e-13):
				raise ValueError(
					"The observed problem and `area` must share the initial state."
				)
			self._append_projection(
				step_index=-1,
				time=stage.time,
				extended_state=state_before,
				projected_jacobian=np.eye(self.physical_size),
			)
			self._initialized = True

		stage_jacobian = central_difference_jacobian(
			stage.map_state,
			state_before,
			relative_step=self.relative_step,
		)
		# The tangent has shape (4N, 2N): rows are current extended variables and
		# columns are the original physical boundary coordinates.
		self._extended_tangent = stage_jacobian @ self._extended_tangent

		if stage.stage_index == _BM4_STAGE_COUNT - 1:
			projected_jacobian = self.projection @ self._extended_tangent
			self._last_completed = (
				stage.step_index,
				stage.time,
				state_after.copy(),
				projected_jacobian.copy(),
			)
			if (stage.step_index + 1) % self.record_every == 0:
				self._append_projection(
					step_index=stage.step_index,
					time=stage.time,
					extended_state=state_after,
					projected_jacobian=projected_jacobian,
				)
			self._expected_step += 1
			self._expected_stage = 0
		else:
			self._expected_stage += 1

	def _validated_extended_state(self, state: np.ndarray) -> np.ndarray:
		"""Require the doubled physical layout and reject energy augmentation."""
		value = np.asarray(state, dtype=float)
		if (
			value.ndim != 1
			or value.size != self.extended_size
			or not np.all(np.isfinite(value))
		):
			raise ValueError(
				"Projected GC diagnostics require the finite 4N doubled state; "
				"run the method with `track_energy=False`."
			)
		return value

	def _append_projection(
		self,
		*,
		step_index: int,
		time: float,
		extended_state: np.ndarray,
		projected_jacobian: np.ndarray,
	) -> None:
		"""Calculate scalar diagnostics and buffer one physical projection."""
		projected_state = self.projection @ extended_state
		first = extended_state[: self.physical_size]
		second = extended_state[self.physical_size :]
		copy_separation = float(np.linalg.norm(first - second))
		state_scale = max(
			float(np.linalg.norm(projected_state)),
			float(np.finfo(float).eps),
		)
		area = float(self.area.calculate_area(projected_state, period=self.period))
		area_error = area - self.initial_area
		defect = (
			projected_jacobian.T @ self.physical_form @ projected_jacobian
			- self.physical_form
		)
		sign, log_abs_determinant = np.linalg.slogdet(projected_jacobian)
		determinant = float(sign * np.exp(log_abs_determinant))
		defect_frobenius = float(np.linalg.norm(defect, ord="fro"))
		record = ProjectedAreaRecord(
			observation_index=len(self._records),
			step_index=step_index,
			time=float(time),
			particle_count=self.particle_count,
			signed_area=area,
			absolute_area_error=area_error,
			relative_area_error=area_error / abs(self.initial_area),
			determinant=determinant,
			determinant_error=abs(determinant - 1.0),
			log_abs_determinant=float(log_abs_determinant),
			condition_number=float(np.linalg.cond(projected_jacobian)),
			defect_frobenius=defect_frobenius,
			relative_defect=(
				defect_frobenius / float(np.linalg.norm(self.physical_form, ord="fro"))
			),
			max_abs_defect=float(np.max(np.abs(defect))),
			copy_separation=copy_separation,
			relative_copy_separation=copy_separation / state_scale,
		)
		self._records.append(record)
		self._buffer.append(
			_BufferedProjection(
				record=record,
				projected_state=projected_state,
				projected_jacobian=projected_jacobian.copy(),
			)
		)
		self._last_recorded_step = step_index
		if self.verbose:
			print(
				f"[projected] step={record.step_index:05d} t={record.time:.6g} "
				f"relative_defect={record.relative_defect:.3e} "
				f"relative_area_error={record.relative_area_error:+.3e} "
				f"copy_separation={record.copy_separation:.3e}"
			)
		if len(self._buffer) >= self.chunk_size:
			self.flush()

	def flush(self) -> ProjectedAreaOutputBlock | None:
		"""Write buffered projected states and Jacobians as one indexed block."""
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
				"projected_jacobians": np.stack(
				[sample.projected_jacobian for sample in self._buffer]
			),
				"projected_states": np.stack(
				[sample.projected_state for sample in self._buffer]
			),
				"observation_indices": np.asarray(
				[sample.record.observation_index for sample in self._buffer], dtype=int
			),
				"step_indices": np.asarray(
				[sample.record.step_index for sample in self._buffer], dtype=int
			),
				"times": np.asarray(
					[sample.record.time for sample in self._buffer]
				),
				"signed_areas": np.asarray(
				[sample.record.signed_area for sample in self._buffer]
			),
			},
			metadata={
				"objective": "Symplecticity and area of projected GC trajectories",
				"particle_count": self.particle_count,
				"initial_signed_area": self.initial_area,
				"period": self.period,
				"record_every_complete_steps": self.record_every,
				"finite_difference_relative_step": (
				float(np.cbrt(np.finfo(float).eps))
				if self.relative_step is None
				else float(self.relative_step)
				),
				"metadata": self.metadata,
			},
		)

		block = ProjectedAreaOutputBlock(
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
			step_index, time, extended_state, projected_jacobian = self._last_completed
			if step_index != self._last_recorded_step:
				self._append_projection(
					step_index=step_index,
					time=time,
					extended_state=extended_state,
					projected_jacobian=projected_jacobian,
				)
		self.flush()
		self._closed = True

__all__ = [
	"ProjectedAreaOutputBlock",
	"ProjectedAreaRecord",
	"ProjectedSymplecticityAreaObserver",
	"gc_average_projection",
	"gc_diagonal_embedding",
	"gc_physical_symplectic_form",
]
