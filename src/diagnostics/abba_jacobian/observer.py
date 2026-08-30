"""Persist local spectral analysis of complete implicit-ABBA step Jacobians."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np

from diagnostics.jacobians import (
	implicit_function_step_jacobian,
	stage_increment_step_jacobian,
)
from diagnostics.output import write_diagnostic_block
from diagnostics.paths import (
	next_block_index,
	notebook_output_directory,
	validate_block_name,
)
from simulation import ABBA2ImplicitIntegrationStep, IntegrationStep

from .analysis import (
	ParticleJacobianAnalysis,
	analyze_particle_jacobian,
	particle_jacobian_blocks,
)


ImplicitABBAJacobianMethod: TypeAlias = Literal[
	"implicit_function",
	"stage_increment",
]
IMPLICIT_ABBA_JACOBIAN_METHODS: tuple[ImplicitABBAJacobianMethod, ...] = (
	"implicit_function",
	"stage_increment",
)


@dataclass(frozen=True, slots=True)
class ImplicitABBAJacobianRecord:
	"""Scalar and flattened matrix data for one particle at one complete step."""

	observation_index: int
	step_index: int
	particle_index: int
	start_time: float
	end_time: float
	duration: float
	method_name: str
	formulation_name: str
	jacobian_method: str
	j_00: float
	j_01: float
	j_10: float
	j_11: float
	trace: float
	determinant: float
	discriminant: float
	discriminant_tolerance: float
	spectral_class: str
	condition_number: float
	spectral_radius: float
	eigenvalue_separation: float
	eigenvector_condition_number: float
	eigendirections_defined: bool
	eigenvalue_0_real: float
	eigenvalue_0_imag: float
	eigenvalue_0_modulus: float
	eigenvalue_0_argument: float
	eigenvalue_1_real: float
	eigenvalue_1_imag: float
	eigenvalue_1_modulus: float
	eigenvalue_1_argument: float
	eigenline_0_angle: float
	eigenline_1_angle: float
	singular_value_max: float
	singular_value_min: float
	singular_rate_max: float
	singular_rate_min: float
	singular_directions_defined: bool
	singular_line_max_angle: float
	singular_line_min_angle: float


@dataclass(frozen=True, slots=True)
class ImplicitABBAJacobianSample:
	"""Matrix-valued analysis retained for one observed complete ABBA step."""

	observation_index: int
	step_index: int
	start_time: float
	end_time: float
	duration: float
	method_name: str
	formulation_name: str
	state_before: np.ndarray
	state_after: np.ndarray
	jacobian: np.ndarray
	particle_jacobians: np.ndarray
	particle_analyses: tuple[ParticleJacobianAnalysis, ...]


@dataclass(frozen=True, slots=True)
class ImplicitABBAJacobianOutputBlock:
	"""Synchronized files written for one buffered group of observed steps."""

	index: int
	step_count: int
	record_count: int
	summary_path: Path
	arrays_path: Path
	metadata_path: Path


def _positive_integer(value: int, name: str) -> int:
	"""Normalize one positive integer observer control."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _record_from_analysis(
	*,
	sample: ImplicitABBAJacobianSample,
	particle_index: int,
	analysis: ParticleJacobianAnalysis,
	jacobian_method: ImplicitABBAJacobianMethod,
) -> ImplicitABBAJacobianRecord:
	"""Flatten one particle analysis for portable CSV output."""
	matrix = analysis.jacobian
	eigenvalues = analysis.eigenvalues
	singular_rates = np.log(analysis.singular_values) / abs(sample.duration)
	return ImplicitABBAJacobianRecord(
		observation_index=sample.observation_index,
		step_index=sample.step_index,
		particle_index=particle_index,
		start_time=sample.start_time,
		end_time=sample.end_time,
		duration=sample.duration,
		method_name=sample.method_name,
		formulation_name=sample.formulation_name,
		jacobian_method=jacobian_method,
		j_00=float(matrix[0, 0]),
		j_01=float(matrix[0, 1]),
		j_10=float(matrix[1, 0]),
		j_11=float(matrix[1, 1]),
		trace=analysis.trace,
		determinant=analysis.determinant,
		discriminant=analysis.discriminant,
		discriminant_tolerance=analysis.discriminant_tolerance,
		spectral_class=analysis.spectral_class,
		condition_number=analysis.condition_number,
		spectral_radius=analysis.spectral_radius,
		eigenvalue_separation=analysis.eigenvalue_separation,
		eigenvector_condition_number=analysis.eigenvector_condition_number,
		eigendirections_defined=analysis.eigendirections_defined,
		eigenvalue_0_real=float(eigenvalues[0].real),
		eigenvalue_0_imag=float(eigenvalues[0].imag),
		eigenvalue_0_modulus=float(abs(eigenvalues[0])),
		eigenvalue_0_argument=float(np.angle(eigenvalues[0])),
		eigenvalue_1_real=float(eigenvalues[1].real),
		eigenvalue_1_imag=float(eigenvalues[1].imag),
		eigenvalue_1_modulus=float(abs(eigenvalues[1])),
		eigenvalue_1_argument=float(np.angle(eigenvalues[1])),
		eigenline_0_angle=float(analysis.eigenvector_line_angles[0]),
		eigenline_1_angle=float(analysis.eigenvector_line_angles[1]),
		singular_value_max=float(analysis.singular_values[0]),
		singular_value_min=float(analysis.singular_values[1]),
		singular_rate_max=float(singular_rates[0]),
		singular_rate_min=float(singular_rates[1]),
		singular_directions_defined=analysis.singular_directions_defined,
		singular_line_max_angle=float(analysis.singular_vector_line_angles[0]),
		singular_line_min_angle=float(analysis.singular_vector_line_angles[1]),
	)


class ImplicitABBAJacobianObserver:
	"""Analyze the local physical Jacobian of selected implicit-ABBA steps.

	The observer uses converged stage snapshots emitted by either
	``ABBA2Implicit`` projection formulation. It studies only the complete
	physical map Jacobian; it does not accumulate tangent maps or calculate area
	and symplecticity quantities.
	"""

	def __init__(
		self,
		*,
		notebook_path: str | Path,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "implicit_abba_jacobian",
		sample_every: int = 1,
		chunk_size: int = 64,
		jacobian_method: ImplicitABBAJacobianMethod = "implicit_function",
		discriminant_relative_tolerance: float = 1e-10,
		verbose: bool = True,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure local Jacobian evaluation, classification, and persistence."""
		if jacobian_method not in IMPLICIT_ABBA_JACOBIAN_METHODS:
			raise ValueError(
				"`jacobian_method` must be 'implicit_function' or 'stage_increment'."
			)
		tolerance = float(discriminant_relative_tolerance)
		if not np.isfinite(tolerance) or tolerance <= 0.0:
			raise ValueError(
				"`discriminant_relative_tolerance` must be positive and finite."
			)
		self.output_directory = notebook_output_directory(
			notebook_path,
			project_root=project_root,
			run_date=run_date,
		)
		self.block_name = validate_block_name(block_name)
		self.sample_every = _positive_integer(sample_every, "sample_every")
		self.chunk_size = _positive_integer(chunk_size, "chunk_size")
		self.jacobian_method = jacobian_method
		self.discriminant_relative_tolerance = tolerance
		self.verbose = bool(verbose)
		self.metadata = dict(metadata or {})
		self._expected_step = 0
		self._particle_count: int | None = None
		self._closed = False
		self._samples: list[ImplicitABBAJacobianSample] = []
		self._records: list[ImplicitABBAJacobianRecord] = []
		self._buffer: list[ImplicitABBAJacobianSample] = []
		self._output_blocks: list[ImplicitABBAJacobianOutputBlock] = []
		self._next_index = next_block_index(self.output_directory, self.block_name)

	@property
	def particle_count(self) -> int | None:
		"""Return the particle count after the first observed step, if available."""
		return self._particle_count

	@property
	def samples(self) -> tuple[ImplicitABBAJacobianSample, ...]:
		"""Return all matrix-valued samples retained for interactive analysis."""
		return tuple(self._samples)

	@property
	def records(self) -> tuple[ImplicitABBAJacobianRecord, ...]:
		"""Return all per-particle scalar records retained in memory."""
		return tuple(self._records)

	@property
	def output_blocks(self) -> tuple[ImplicitABBAJacobianOutputBlock, ...]:
		"""Return synchronized output file groups written during this run."""
		return tuple(self._output_blocks)

	def __enter__(self) -> ImplicitABBAJacobianObserver:
		"""Open a context-managed observer session."""
		return self

	def __exit__(self, *_exception: object) -> None:
		"""Flush the final partial output block."""
		self.close()

	def __call__(self, step: IntegrationStep) -> None:
		"""Analyze one consecutive complete implicit-ABBA integration step."""
		if self._closed:
			raise RuntimeError("This implicit-ABBA Jacobian observer is already closed.")
		if not isinstance(step, ABBA2ImplicitIntegrationStep):
			raise TypeError(
				"ImplicitABBAJacobianObserver requires ABBA2ImplicitIntegrationStep data."
			)
		if step.step_index != self._expected_step:
			raise ValueError("Implicit ABBA steps must be observed consecutively.")
		self._expected_step += 1
		if step.step_index % self.sample_every:
			return

		state_before = np.asarray(step.state_before, dtype=float)
		state_after = np.asarray(step.state_after, dtype=float)
		if (
			state_before.ndim != 1
			or state_before.size == 0
			or state_before.size % 2
			or state_after.shape != state_before.shape
			or not np.all(np.isfinite(state_before))
			or not np.all(np.isfinite(state_after))
		):
			raise ValueError(
				"Implicit ABBA Jacobian analysis requires finite planar physical states."
			)
		particle_count = state_before.size // 2
		if self._particle_count is None:
			self._particle_count = particle_count
		elif particle_count != self._particle_count:
			raise ValueError("The particle count changed during Jacobian observation.")

		calculator = (
			implicit_function_step_jacobian
			if self.jacobian_method == "implicit_function"
			else stage_increment_step_jacobian
		)
		jacobian = calculator(step)
		particle_jacobians = particle_jacobian_blocks(jacobian, particle_count)
		analyses = tuple(
			analyze_particle_jacobian(
				block,
				discriminant_relative_tolerance=(
					self.discriminant_relative_tolerance
				),
			)
			for block in particle_jacobians
		)
		sample = ImplicitABBAJacobianSample(
			observation_index=len(self._samples),
			step_index=step.step_index,
			start_time=float(step.start_time),
			end_time=float(step.time),
			duration=float(step.duration),
			method_name=step.method_name,
			formulation_name=step.formulation_name,
			state_before=state_before.copy(),
			state_after=state_after.copy(),
			jacobian=np.asarray(jacobian, dtype=float).copy(),
			particle_jacobians=particle_jacobians.copy(),
			particle_analyses=analyses,
		)
		self._samples.append(sample)
		self._records.extend(
			_record_from_analysis(
				sample=sample,
				particle_index=particle,
				analysis=analysis,
				jacobian_method=self.jacobian_method,
			)
			for particle, analysis in enumerate(analyses)
		)
		self._buffer.append(sample)
		if self.verbose:
			classes = ",".join(analysis.spectral_class for analysis in analyses)
			print(
				f"[implicit-abba-jacobian] step={step.step_index:05d} "
				f"t={step.time:.6g} classes={classes}"
			)
		if len(self._buffer) >= self.chunk_size:
			self.flush()

	def flush(self) -> ImplicitABBAJacobianOutputBlock | None:
		"""Write the current local-Jacobian buffer as one indexed block."""
		if not self._buffer:
			return None
		block_index = self._next_index
		buffer_observations = {
			sample.observation_index for sample in self._buffer
		}
		buffer_records = [
			record
			for record in self._records
			if record.observation_index in buffer_observations
		]
		analyses = [sample.particle_analyses for sample in self._buffer]
		paths = write_diagnostic_block(
			output_directory=self.output_directory,
			block_name=self.block_name,
			block_index=block_index,
			rows=[asdict(record) for record in buffer_records],
			arrays={
				"jacobians": np.stack(
					[sample.jacobian for sample in self._buffer]
				),
				"particle_jacobians": np.stack(
					[sample.particle_jacobians for sample in self._buffer]
				),
				"states_before": np.stack(
					[sample.state_before for sample in self._buffer]
				),
				"states_after": np.stack(
					[sample.state_after for sample in self._buffer]
				),
				"eigenvalues": np.asarray(
					[[item.eigenvalues for item in row] for row in analyses]
				),
				"eigenvectors": np.asarray(
					[[item.eigenvectors for item in row] for row in analyses]
				),
				"eigenvector_line_angles": np.asarray(
					[
						[item.eigenvector_line_angles for item in row]
						for row in analyses
					]
				),
				"singular_values": np.asarray(
					[[item.singular_values for item in row] for row in analyses]
				),
				"right_singular_vectors": np.asarray(
					[
						[item.right_singular_vectors for item in row]
						for row in analyses
					]
				),
				"singular_vector_line_angles": np.asarray(
					[
						[item.singular_vector_line_angles for item in row]
						for row in analyses
					]
				),
				"spectral_classes": np.asarray(
					[
						[item.spectral_class for item in row]
						for row in analyses
					]
				),
				"step_indices": np.asarray(
					[sample.step_index for sample in self._buffer], dtype=int
				),
				"start_times": np.asarray(
					[sample.start_time for sample in self._buffer]
				),
				"end_times": np.asarray(
					[sample.end_time for sample in self._buffer]
				),
				"durations": np.asarray(
					[sample.duration for sample in self._buffer]
				),
			},
			metadata={
				"objective": "Local spectral analysis of implicit ABBA step Jacobians",
				"particle_count": self._particle_count,
				"sample_every_complete_steps": self.sample_every,
				"step_jacobian_method": self.jacobian_method,
				"discriminant_relative_tolerance": (
					self.discriminant_relative_tolerance
				),
				"eigenline_angle_interval": "[-pi/2, pi/2)",
				"complex_eigenvector_policy": (
					"persist_complex_vectors_without_real_line_angles"
				),
				"metadata": self.metadata,
			},
		)
		block = ImplicitABBAJacobianOutputBlock(
			index=block_index,
			step_count=len(self._buffer),
			record_count=len(buffer_records),
			summary_path=paths.summary,
			arrays_path=paths.arrays,
			metadata_path=paths.metadata,
		)
		self._output_blocks.append(block)
		self._buffer.clear()
		self._next_index += 1
		return block

	def close(self) -> None:
		"""Flush pending samples and reject subsequent observations."""
		if self._closed:
			return
		self.flush()
		self._closed = True


__all__ = [
	"IMPLICIT_ABBA_JACOBIAN_METHODS",
	"ImplicitABBAJacobianMethod",
	"ImplicitABBAJacobianObserver",
	"ImplicitABBAJacobianOutputBlock",
	"ImplicitABBAJacobianRecord",
	"ImplicitABBAJacobianSample",
]
