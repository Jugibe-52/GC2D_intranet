"""Canonical configuration axes shared by the public ABBA methods."""

from __future__ import annotations

from typing import Literal, TypeAlias


ProjectionFormulation: TypeAlias = Literal[
	"reduced_multiplier",
	"simultaneous_state_multiplier",
]
ABBA_PROJECTION_FORMULATIONS: tuple[ProjectionFormulation, ...] = (
	"reduced_multiplier",
	"simultaneous_state_multiplier",
)

ProjectionPlacement: TypeAlias = Literal[
	"around_complete_composition",
]
ABBA4_PROJECTION_PLACEMENTS: tuple[ProjectionPlacement, ...] = (
	"around_complete_composition",
)

StateExtension: TypeAlias = Literal[
	"physical",
]
ABBA_STATE_EXTENSIONS: tuple[StateExtension, ...] = (
	"physical",
)


def _validate_projection_formulation(value: str) -> ProjectionFormulation:
	"""Return one supported implicit-projection formulation."""
	if value not in ABBA_PROJECTION_FORMULATIONS:
		raise ValueError(
			"`projection_formulation` must be 'reduced_multiplier' or "
			"'simultaneous_state_multiplier'."
		)
	return value


def _validate_projection_placement(value: str) -> ProjectionPlacement:
	"""Return the common higher-order ABBA projection placement."""
	if value not in ABBA4_PROJECTION_PLACEMENTS:
		raise ValueError(
			"ABBA `projection_placement` must be 'around_complete_composition'; "
			"intermediate projections are not supported."
		)
	return value


def _validate_state_extension(value: str) -> StateExtension:
	"""Return one supported ABBA state-space strategy."""
	if value not in ABBA_STATE_EXTENSIONS:
		raise ValueError(
			"Only spatial duplication is supported. Use state_extension='physical' "
			"and track_energy=True for time and passive momentum per particle; "
			"the former fully_extended time/momentum projection has been removed."
		)
	return value


def _state_dimension_diagnostics(
	projection_formulation: ProjectionFormulation | None = None,
	*,
	particle_count: int = 1,
) -> dict[str, int | str]:
	"""Describe the actual accepted, splitting, and nonlinear workspaces."""
	if particle_count < 1:
		raise ValueError("`particle_count` must be a positive integer.")
	result: dict[str, int | str] = {
		"accepted_internal_state_dimension": 4 * particle_count,
		"base_splitting_state_dimension": 4 * particle_count,
		"observer_state_dimension": 2 * particle_count,
		"observer_state_kind": "physical_map",
	}
	if projection_formulation is not None:
		result["nonlinear_unknown_dimension"] = (2 if projection_formulation == "reduced_multiplier" else 6) * particle_count
	return result


__all__: list[str] = []


import numpy as np

def _positive_finite(value: float, name: str) -> float:
	"""Return a positive finite float, rejecting booleans as numeric inputs."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _nonnegative_finite(value: float, name: str) -> float:
	"""Return a non-negative finite float, rejecting boolean values."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be non-negative and finite.")
	result = float(value)
	if not np.isfinite(result) or result < 0:
		raise ValueError(f"`{name}` must be non-negative and finite.")
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Return a positive built-in integer without accepting booleans."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


