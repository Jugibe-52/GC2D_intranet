"""Validated configuration shared by implicit ABBA public methods."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ...observation import StepObserver
from .._nonlinear import NonlinearSolver, _validate_nonlinear_solver
from ._configuration import (
    ProjectionFormulation, StateExtension, _resolved_track_energy,
    _validate_projection_formulation, _validate_state_extension,
)

def _positive_finite(value: float, name: str) -> float:
	"""Normalize a strictly positive finite solver parameter."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Normalize a strictly positive integer solver parameter."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


@dataclass(frozen=True, slots=True)
class _ABBAImplicitConfig:
	"""Validate configuration shared by projected implicit ABBA methods."""

	projection_formulation: ProjectionFormulation = "reduced_multiplier"
	state_extension: StateExtension = "physical"
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False
	step_observer: StepObserver | None = None
	track_energy: bool = False

	def __post_init__(self) -> None:
		"""Validate the nonlinear projection solver configuration."""
		object.__setattr__(
			self,
			"projection_formulation",
			_validate_projection_formulation(self.projection_formulation),
		)
		object.__setattr__(
			self,
			"state_extension",
			_validate_state_extension(self.state_extension),
		)
		object.__setattr__(
			self,
			"track_energy",
			_resolved_track_energy(self.track_energy, self.state_extension),
		)
		object.__setattr__(
			self,
			"newton_absolute_tolerance",
			_positive_finite(
				self.newton_absolute_tolerance,
				"newton_absolute_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_relative_tolerance",
			_positive_finite(
				self.newton_relative_tolerance,
				"newton_relative_tolerance",
			),
		)
		object.__setattr__(
			self,
			"newton_max_iterations",
			_positive_integer(
				self.newton_max_iterations,
				"newton_max_iterations",
			),
		)
		object.__setattr__(
			self,
			"nonlinear_solver",
			_validate_nonlinear_solver(self.nonlinear_solver),
		)


__all__: list[str] = []
