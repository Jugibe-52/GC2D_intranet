"""Explicit initial positions that supplement saved Poincare stars.

The same cell fractions can be reused for every rho and numerical method.
Selection at one rho identifies gaps in those finite saved returns; it does
not establish an invariant empty region at other rho values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from diagnostics.persistence import StoredSolution
from initial_conditions.gc import GCInitialConfiguration
from solution import Solution
from studies.dimensional_h5_midpoint import DimensionalH5Field
from studies.poincare_rho_sweep import (
    PreparedRhoStar, RhoStarConfig, _validate_rho_cycle_solution,
    build_rho_star, folded_rho_positions, prepare_rho_star, validate_rho_solution,
)


@dataclass(frozen=True)
class GapProbeSeeds:
    """Persistent IDs and explicit (R, Z) fractions measured from the cell origin."""

    fractions: tuple[tuple[float, float], ...]
    particle_ids: tuple[int, ...]
    gap_names: tuple[str, ...]
    selection_rho: float = 0.3

    def __post_init__(self) -> None:
        """Freeze JSON sequences and reject ambiguous or duplicate seeds."""
        points = np.asarray(self.fractions, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2
                or not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= 1)):
            raise ValueError("fractions must contain at least two finite xy pairs in [0, 1).")
        if np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("Probe initial positions must be distinct.")
        ids, names = tuple(self.particle_ids), tuple(self.gap_names)
        if (len(ids) != len(points) or len(set(ids)) != len(ids)
                or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                       or value < 1 for value in ids)):
            raise ValueError("particle_ids must be distinct positive integers, one per seed.")
        if len(names) != len(points) or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("gap_names must contain a nonempty name for every seed.")
        if not np.isfinite(self.selection_rho) or self.selection_rho < 0:
            raise ValueError("selection_rho must be finite and nonnegative.")
        object.__setattr__(self, "fractions", tuple(tuple(float(value) for value in row) for row in points))
        object.__setattr__(self, "particle_ids", tuple(int(value) for value in ids))
        object.__setattr__(self, "gap_names", names)
        object.__setattr__(self, "selection_rho", float(self.selection_rho))

    def to_metadata(self) -> dict[str, Any]:
        """Return the exact JSON representation used for archive reuse checks."""
        return dict(fractions=[list(row) for row in self.fractions],
                    particle_ids=list(self.particle_ids), gap_names=list(self.gap_names),
                    selection_rho=self.selection_rho)


def _replace_star_initial_positions(prepared: PreparedRhoStar, seeds: GapProbeSeeds) -> PreparedRhoStar:
    """Reuse verified physics and sampling while replacing all star geometry."""
    if not isinstance(seeds, GapProbeSeeds):
        raise TypeError("seeds must be GapProbeSeeds.")
    if prepared.config.particles != len(seeds.fractions):
        raise ValueError("config.particles must equal the explicit probe count.")
    metadata = dict(prepared.metadata)
    # Coordinates are first defined in meters, then transformed exactly as the
    # background study: q = scale * (X - origin), in component-major xy order.
    cell = np.asarray(metadata["cell_bounds_m"], dtype=float)
    physical_xy = cell[:2] + cell[2] * np.asarray(seeds.fractions)
    xy = (physical_xy - np.asarray(metadata["spatial_origin_m"])) * metadata["spatial_scale_per_m"]
    initial = GCInitialConfiguration.from_components(x=xy[:, 0], y=xy[:, 1])
    problem = InitialValueProblem(prepared.problem.dynamics, initial)
    for name in ("star_center_m", "cell_radius_m", "initial_radii_m", "initial_radius_fraction",
                 "star_center", "initial_radii", "arm_id"):
        metadata.pop(name, None)
    metadata.update(
        study="poincare_gap_probes", probe_config=seeds.to_metadata(),
        initial_geometry="explicit_cell_fraction_gap_probes",
        initial_geometry_definition="probe_config.fractions; star arm, angle and radius controls are unused",
        initial_positions_m=physical_xy, initial_positions=xy,
        initial_positions_cell_fraction=np.asarray(seeds.fractions),
        particle_id=np.asarray(seeds.particle_ids, dtype=int),
        particle_order="explicit probe_config order, independent of radial rank",
        gap_names=list(seeds.gap_names),
        cell_center_m=cell[:2] + cell[2] / 2,
        cell_center=(cell[:2] + cell[2] / 2 - np.asarray(metadata["spatial_origin_m"]))
                    * metadata["spatial_scale_per_m"],
    )
    return replace(prepared, problem=problem, metadata=metadata)


def build_gap_probes(field: DimensionalH5Field, config: RhoStarConfig, *,
                     source_sha256: str, seeds: GapProbeSeeds) -> PreparedRhoStar:
    """Assemble explicit seeds with the background's exact dynamics and clocks."""
    prepared = build_rho_star(field, config, source_sha256=source_sha256)
    return _replace_star_initial_positions(prepared, seeds)


def prepare_gap_probes(source: str | Path, config: RhoStarConfig, *,
                       seeds: GapProbeSeeds) -> PreparedRhoStar:
    """Read and fingerprint the HDF5 source, then prepare explicit probe seeds."""
    return _replace_star_initial_positions(prepare_rho_star(source, config), seeds)


def validate_gap_solution(solution: Solution, prepared: PreparedRhoStar, *,
                          options: ExecutionOptions | None = None) -> None:
    """Audit a new probe integration against its complete prepared inputs."""
    if prepared.metadata.get("study") != "poincare_gap_probes":
        raise ValueError("The prepared problem is not an explicit gap-probe study.")
    validate_rho_solution(solution, prepared, options=options)
    validate_saved_gap_probes(StoredSolution(solution, prepared.metadata, None))


def _saved_config(metadata: dict[str, Any]) -> RhoStarConfig:
    """Restore tuple fields while retaining defaults for historical star runs."""
    values = dict(metadata["config"])
    if "source_selection" in values:
        values["source_selection"] = tuple(values["source_selection"])
    return RhoStarConfig(**values)


def validate_saved_gap_probes(stored: StoredSolution, *,
                              background: StoredSolution | None = None) -> GapProbeSeeds:
    """Validate saved probes and their optional background without integration.

    Returns the canonical immutable seed contract for matching multiple rho
    values and methods. Original particle IDs and seed positions remain separate
    from the saved star; viewers concatenate only their displayed coordinates.
    """
    metadata, solution = stored.metadata, stored.solution
    if metadata.get("study") != "poincare_gap_probes":
        raise ValueError("Expected a saved explicit gap-probe study.")
    seeds, config, xy = _validated_saved_probe_geometry(metadata)
    initial = GCInitialConfiguration.from_components(x=xy[:, 0], y=xy[:, 1])
    initial_state = initial.initial_state
    assert initial_state is not None
    options = None
    if metadata.get("execution_options", {}).get("backend") == "jax":
        options = ExecutionOptions(backend="jax", device=metadata["execution_options"].get("device", "cpu"))
    _validate_rho_cycle_solution(solution, config, output_times=np.arange(config.cycles + 1, dtype=float),
                                initial_state=initial_state, options=options)
    saved_initial_state = solution.source.initial_state
    if saved_initial_state is None or not np.array_equal(saved_initial_state, initial_state):
        raise ValueError("The saved probes' source configuration differs from their explicit seeds.")
    wrapped, fractions = folded_rho_positions(solution, metadata)
    for name, expected in (("cycle_positions_wrapped", wrapped),
                           ("cycle_positions_cell_fraction", fractions)):
        actual = solution.diagnostics.get(name)
        if actual is not None or metadata.get("saved_folded_returns"):
            if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
                raise ValueError(f"The saved probes have inconsistent {name}.")
    _validate_gap_background(background, config, solution, metadata, seeds)
    return seeds


def _validate_gap_background(
	background: StoredSolution | None, config: RhoStarConfig, solution: Solution,
	metadata: dict[str, Any], seeds: GapProbeSeeds,
) -> None:
	"""Require appended probes to share scientific inputs and cycles while preserving distinct IDs."""
	if background is not None:
		base_config = _saved_config(background.metadata)
		# Star geometry defines the original forty positions only. Every other
		# physical, temporal and numerical input must agree for appended probes.
		geometric_controls = {"particles", "arms", "outer_radius_fraction", "first_angle"}
		probe_inputs = {key: value for key, value in asdict(config).items() if key not in geometric_controls}
		base_inputs = {key: value for key, value in asdict(base_config).items() if key not in geometric_controls}
		if probe_inputs != base_inputs or not np.array_equal(solution.t, background.solution.t):
			raise ValueError("The saved probes and background use different scientific inputs or cycles.")
		for name in ("source_sha256", "source_field_indices", "cell_bounds_m", "cell_bounds",
					 "spatial_scale_per_m", "spatial_origin_m", "space_unit", "rho_runtime", "rho_m",
					 "time_unit_seconds", "hamiltonian_convention", "hamiltonian_scale_from_cycle_time"):
			if not np.array_equal(np.asarray(metadata.get(name)), np.asarray(background.metadata.get(name))):
				raise ValueError(f"The saved probes and background differ in {name}.")
		original_ids = np.asarray(background.metadata.get("particle_id", ()))
		if np.intersect1d(original_ids, seeds.particle_ids).size:
			raise ValueError("Probe particle IDs must not overlap original particle IDs.")


def _validated_saved_probe_geometry(
	metadata: dict[str, Any],
) -> tuple[GapProbeSeeds, RhoStarConfig, np.ndarray]:
	"""Restore seed and coordinate metadata and require exact physical/runtime agreement."""
	try:
		seeds = GapProbeSeeds(**metadata["probe_config"])
		config = _saved_config(metadata)
		cell = np.asarray(metadata["cell_bounds_m"], dtype=float)
		runtime_cell = np.asarray(metadata["cell_bounds"], dtype=float)
		scale = float(metadata["spatial_scale_per_m"])
		origin = np.asarray(metadata["spatial_origin_m"], dtype=float)
	except (KeyError, TypeError, OverflowError) as exc:
		raise ValueError("The saved probes lack a valid seed and coordinate contract.") from exc
	if (config.particles != len(seeds.fractions) or cell.shape != (3,)
			or runtime_cell.shape != (3,) or origin.shape != (2,)
			or not np.isfinite(cell).all() or cell[2] <= 0
			or not np.isfinite(scale) or scale <= 0 or not np.isfinite(origin).all()):
		raise ValueError("The saved probes have inconsistent particle or coordinate metadata.")
	if metadata.get("method") != config.method or not metadata.get("source_sha256"):
		raise ValueError("The saved probes lack matching method and source provenance.")
	physical_xy = cell[:2] + cell[2] * np.asarray(seeds.fractions)
	xy = (physical_xy - origin) * scale
	expected_runtime_cell = np.append((cell[:2] - origin) * scale, cell[2] * scale)
	for name, expected in (("initial_positions_m", physical_xy), ("initial_positions", xy),
						   ("initial_positions_cell_fraction", seeds.fractions),
						   ("particle_id", seeds.particle_ids), ("cell_bounds", expected_runtime_cell)):
		if not np.array_equal(np.asarray(metadata.get(name)), np.asarray(expected)):
			raise ValueError(f"The saved probes have inconsistent {name}.")
	return seeds, config, xy


__all__ = ["GapProbeSeeds", "build_gap_probes", "prepare_gap_probes",
           "validate_gap_solution", "validate_saved_gap_probes"]
