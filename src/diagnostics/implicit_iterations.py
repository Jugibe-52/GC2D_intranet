"""Persist nonlinear-solver work for complete accepted implicit steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Mapping, Self

import numpy as np

from simulation import (
	ImplicitABBAIntegrationStep,
	ImplicitBM4IntegrationStep,
	ImplicitIntegrationStep,
	IntegrationStep,
)

from .output import write_diagnostic_block
from .paths import (
	next_block_index,
	notebook_output_directory,
	validate_block_name,
)


@dataclass(frozen=True, slots=True)
class ImplicitIterationRecord:
	"""Nonlinear-solver metrics for one accepted complete implicit step."""

	observation_index: int
	step_index: int
	start_time: float
	end_time: float
	duration: float
	method_name: str
	formulation_name: str
	nonlinear_solver: str
	newton_iterations: int
	residual_evaluations: int
	newton_residual_norm: float
	newton_tolerance: float
	residual_to_tolerance_ratio: float
	projection_multiplier_norm: float

	@property
	def nonlinear_iterations(self) -> int:
		"""Return the correction count through solver-neutral terminology."""
		return self.newton_iterations

	@property
	def nonlinear_residual_norm(self) -> float:
		"""Return the accepted residual norm for either nonlinear solver."""
		return self.newton_residual_norm

	@property
	def nonlinear_tolerance(self) -> float:
		"""Return the effective acceptance tolerance for either solver."""
		return self.newton_tolerance


@dataclass(frozen=True, slots=True)
class ImplicitIterationOutputBlock:
	"""Files written for one buffered group of iteration records."""

	index: int
	record_count: int
	summary_path: Path
	arrays_path: Path
	metadata_path: Path


# Method-specific aliases make public result annotations explicit while the
# shared record schema remains directly comparable across ABBA and BM4.
ImplicitABBAIterationRecord = ImplicitIterationRecord
ImplicitBM4IterationRecord = ImplicitIterationRecord
ImplicitABBAIterationOutputBlock = ImplicitIterationOutputBlock
ImplicitBM4IterationOutputBlock = ImplicitIterationOutputBlock


def _positive_integer(value: int, name: str) -> int:
	"""Normalize one positive integer observer control."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _record_from_step(
	step: ImplicitIntegrationStep,
	observation_index: int,
) -> ImplicitIterationRecord:
	"""Validate and retain the solve metrics emitted with one accepted step."""
	if step.nonlinear_solver not in ("newton", "broyden"):
		raise ValueError("An implicit step reported an unknown nonlinear solver.")
	iterations = step.newton_iterations
	if (
		isinstance(iterations, (bool, np.bool_))
		or not isinstance(iterations, (int, np.integer))
		or iterations < 0
	):
		raise ValueError("Nonlinear iterations must be a non-negative integer.")
	residual_evaluations = step.residual_evaluations
	if (
		isinstance(residual_evaluations, (bool, np.bool_))
		or not isinstance(residual_evaluations, (int, np.integer))
		or residual_evaluations < 1
		or residual_evaluations < iterations + 1
	):
		raise ValueError(
			"Residual evaluations must include the initial residual and every "
			"nonlinear correction."
		)
	residual = float(step.newton_residual_norm)
	tolerance = float(step.newton_tolerance)
	multiplier = float(step.projection_multiplier_norm)
	if not np.isfinite(residual) or residual < 0.0:
		raise ValueError("The final nonlinear residual must be finite and non-negative.")
	if not np.isfinite(tolerance) or tolerance <= 0.0:
		raise ValueError("The effective nonlinear tolerance must be positive and finite.")
	if not np.isfinite(multiplier) or multiplier < 0.0:
		raise ValueError("The multiplier norm must be finite and non-negative.")
	if residual > tolerance:
		raise ValueError("An observed implicit step must satisfy its tolerance.")
	return ImplicitIterationRecord(
		observation_index=observation_index,
		step_index=step.step_index,
		start_time=float(step.start_time),
		end_time=float(step.time),
		duration=float(step.duration),
		method_name=step.method_name,
		formulation_name=step.formulation_name,
		nonlinear_solver=step.nonlinear_solver,
		newton_iterations=int(iterations),
		residual_evaluations=int(residual_evaluations),
		newton_residual_norm=residual,
		newton_tolerance=tolerance,
		residual_to_tolerance_ratio=residual / tolerance,
		projection_multiplier_norm=multiplier,
	)


class _ImplicitIterationObserver:
	"""Share sampling, validation, and persistence for implicit solvers."""

	_step_type: ClassVar[type[ImplicitIntegrationStep]]
	_diagnostic_label: ClassVar[str]
	_objective: ClassVar[str]

	def __init__(
		self,
		*,
		notebook_path: str | Path,
		project_root: str | Path | None,
		run_date: date | str | None,
		block_name: str,
		sample_every: int,
		chunk_size: int,
		verbose: bool,
		metadata: Mapping[str, Any] | None,
	) -> None:
		"""Configure common step sampling and output state."""
		self.output_directory = notebook_output_directory(
			notebook_path,
			project_root=project_root,
			run_date=run_date,
		)
		self.block_name = validate_block_name(block_name)
		self.sample_every = _positive_integer(sample_every, "sample_every")
		self.chunk_size = _positive_integer(chunk_size, "chunk_size")
		self.verbose = bool(verbose)
		self.metadata = dict(metadata or {})
		self._expected_step = 0
		self._closed = False
		self._records: list[ImplicitIterationRecord] = []
		self._buffer: list[ImplicitIterationRecord] = []
		self._output_blocks: list[ImplicitIterationOutputBlock] = []
		self._next_index = next_block_index(self.output_directory, self.block_name)

	@property
	def records(self) -> tuple[ImplicitIterationRecord, ...]:
		"""Return all retained iteration records."""
		return tuple(self._records)

	@property
	def output_blocks(self) -> tuple[ImplicitIterationOutputBlock, ...]:
		"""Return every synchronized block written during this run."""
		return tuple(self._output_blocks)

	def __enter__(self) -> Self:
		"""Open a context-managed observer session."""
		return self

	def __exit__(self, *_exception: object) -> None:
		"""Flush the final partial block."""
		self.close()

	def __call__(self, step: IntegrationStep) -> None:
		"""Record one consecutive complete implicit step of the required type."""
		if self._closed:
			raise RuntimeError(
				f"This {type(self).__name__} instance is already closed."
			)
		if not isinstance(step, self._step_type):
			raise TypeError(
				f"{type(self).__name__} requires {self._step_type.__name__} data."
			)
		implicit_step = step
		if implicit_step.step_index != self._expected_step:
			raise ValueError("Implicit steps must be observed consecutively.")
		self._expected_step += 1
		if implicit_step.step_index % self.sample_every:
			return

		record = _record_from_step(implicit_step, len(self._records))
		self._records.append(record)
		self._buffer.append(record)
		if self.verbose:
			print(
				f"[{self._diagnostic_label}] step={step.step_index:05d} "
				f"t={step.time:.6g} solver={record.nonlinear_solver} "
				f"iterations={record.newton_iterations} "
				f"residual/tolerance={record.residual_to_tolerance_ratio:.3e}"
			)
		if len(self._buffer) >= self.chunk_size:
			self.flush()

	def flush(self) -> ImplicitIterationOutputBlock | None:
		"""Write the buffered records as one indexed diagnostic block."""
		if not self._buffer:
			return None
		block_index = self._next_index
		paths = write_diagnostic_block(
			output_directory=self.output_directory,
			block_name=self.block_name,
			block_index=block_index,
			rows=[asdict(record) for record in self._buffer],
			arrays={
				"step_indices": np.asarray(
					[record.step_index for record in self._buffer], dtype=int
				),
				"start_times": np.asarray(
					[record.start_time for record in self._buffer]
				),
				"end_times": np.asarray(
					[record.end_time for record in self._buffer]
				),
				"durations": np.asarray(
					[record.duration for record in self._buffer]
				),
				"newton_iterations": np.asarray(
					[record.newton_iterations for record in self._buffer], dtype=int
				),
				"nonlinear_iterations": np.asarray(
					[record.newton_iterations for record in self._buffer], dtype=int
				),
				"residual_evaluations": np.asarray(
					[record.residual_evaluations for record in self._buffer], dtype=int
				),
				"newton_residual_norms": np.asarray(
					[record.newton_residual_norm for record in self._buffer]
				),
				"nonlinear_residual_norms": np.asarray(
					[record.newton_residual_norm for record in self._buffer]
				),
				"newton_tolerances": np.asarray(
					[record.newton_tolerance for record in self._buffer]
				),
				"nonlinear_tolerances": np.asarray(
					[record.newton_tolerance for record in self._buffer]
				),
				"residual_to_tolerance_ratios": np.asarray(
					[
						record.residual_to_tolerance_ratio
						for record in self._buffer
					]
				),
				"projection_multiplier_norms": np.asarray(
					[record.projection_multiplier_norm for record in self._buffer]
				),
			},
			metadata={
				"objective": self._objective,
				"sample_every_complete_steps": self.sample_every,
				"residual_norm": "infinity",
				"metadata": self.metadata,
			},
			arrays_kind="iterations",
		)
		block = ImplicitIterationOutputBlock(
			index=block_index,
			record_count=len(self._buffer),
			summary_path=paths.summary,
			arrays_path=paths.arrays,
			metadata_path=paths.metadata,
		)
		self._output_blocks.append(block)
		self._buffer.clear()
		self._next_index += 1
		return block

	def close(self) -> None:
		"""Flush pending records and reject subsequent observations."""
		if self._closed:
			return
		self.flush()
		self._closed = True


class ImplicitABBAIterationObserver(_ImplicitIterationObserver):
	"""Record nonlinear work for accepted implicit-ABBA steps."""

	_step_type = ImplicitABBAIntegrationStep
	_diagnostic_label = "implicit-abba-iterations"
	_objective = "Nonlinear-solver work for accepted implicit ABBA steps"

	def __init__(
		self,
		*,
		notebook_path: str | Path,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "implicit_abba_iterations",
		sample_every: int = 1,
		chunk_size: int = 256,
		verbose: bool = False,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure ABBA iteration sampling and persistence."""
		super().__init__(
			notebook_path=notebook_path,
			project_root=project_root,
			run_date=run_date,
			block_name=block_name,
			sample_every=sample_every,
			chunk_size=chunk_size,
			verbose=verbose,
			metadata=metadata,
		)


class ImplicitBM4IterationObserver(_ImplicitIterationObserver):
	"""Record nonlinear work for accepted Hairer-projected BM4 steps."""

	_step_type = ImplicitBM4IntegrationStep
	_diagnostic_label = "implicit-bm4-iterations"
	_objective = "Nonlinear-solver work for accepted implicit BM4 steps"

	def __init__(
		self,
		*,
		notebook_path: str | Path,
		project_root: str | Path | None = None,
		run_date: date | str | None = None,
		block_name: str = "implicit_bm4_iterations",
		sample_every: int = 1,
		chunk_size: int = 256,
		verbose: bool = False,
		metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Configure BM4 iteration sampling and persistence."""
		super().__init__(
			notebook_path=notebook_path,
			project_root=project_root,
			run_date=run_date,
			block_name=block_name,
			sample_every=sample_every,
			chunk_size=chunk_size,
			verbose=verbose,
			metadata=metadata,
		)


__all__ = [
	"ImplicitABBAIterationObserver",
	"ImplicitABBAIterationOutputBlock",
	"ImplicitABBAIterationRecord",
	"ImplicitBM4IterationObserver",
	"ImplicitBM4IterationOutputBlock",
	"ImplicitBM4IterationRecord",
	"ImplicitIterationOutputBlock",
	"ImplicitIterationRecord",
]
