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

StateExtension: TypeAlias = Literal[
	"physical",
	"shared_time",
	"fully_extended",
]
ABBA_STATE_EXTENSIONS: tuple[StateExtension, ...] = (
	"physical",
	"shared_time",
	"fully_extended",
)


def _validate_projection_formulation(value: str) -> ProjectionFormulation:
	"""Return one supported implicit-projection formulation."""
	if value not in ABBA_PROJECTION_FORMULATIONS:
		raise ValueError(
			"`projection_formulation` must be 'reduced_multiplier' or "
			"'simultaneous_state_multiplier'."
		)
	return value


def _validate_state_extension(value: str) -> StateExtension:
	"""Return one supported ABBA state-space strategy."""
	if value not in ABBA_STATE_EXTENSIONS:
		raise ValueError(
			"`state_extension` must be 'physical', 'shared_time', or "
			"'fully_extended'."
		)
	return value


def _state_dimension_diagnostics(
	state_extension: StateExtension,
	projection_formulation: ProjectionFormulation | None = None,
	*,
	particle_count: int = 1,
) -> dict[str, int | str]:
	"""Describe the actual accepted, splitting, and nonlinear workspaces."""
	if particle_count < 1:
		raise ValueError("`particle_count` must be a positive integer.")
	if state_extension == "physical":
		accepted_dimension = 2 * particle_count
		base_dimension = 4 * particle_count
	elif state_extension == "shared_time":
		accepted_dimension = 2 * particle_count + 2
		base_dimension = 4 * particle_count + 2
	else:
		accepted_dimension = 2 * particle_count + 2
		base_dimension = 4 * particle_count + 4
	result: dict[str, int | str] = {
		"accepted_internal_state_dimension": accepted_dimension,
		"base_splitting_state_dimension": base_dimension,
		"observer_state_dimension": (
			accepted_dimension
			if state_extension != "shared_time"
			else 2 * particle_count
		),
		"observer_state_kind": (
			"accepted_internal_map"
			if state_extension == "fully_extended"
			else "physical_map"
		),
	}
	if projection_formulation is not None:
		if state_extension == "fully_extended":
			unknown_dimension = (
				accepted_dimension
				if projection_formulation == "reduced_multiplier"
				else 3 * accepted_dimension
			)
		else:
			unknown_dimension = (
				2 * particle_count
				if projection_formulation == "reduced_multiplier"
				else 6 * particle_count
			)
		result["nonlinear_unknown_dimension"] = unknown_dimension
	return result


__all__: list[str] = []
