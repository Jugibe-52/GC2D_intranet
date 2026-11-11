"""Reconstruct the recorded 2026 Poincare inputs for passive transport reruns.

The historical dimensional import used the x-axis span and a normalization
roundtrip. Both differ slightly from the current general-purpose import. Keep
their arithmetic here, with exact archive checks, instead of modifying new
physical studies or silently changing their initial conditions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from functools import lru_cache
from hashlib import file_digest
from pathlib import Path
from typing import Any

import numpy as np

from dynamics.gc import GuidingCenterDynamics
from potential import GC2DH5Metadata, Grid, Potential
from problem import InitialValueProblem
from studies.dimensional_h5_midpoint import DimensionalH5Field, resolve_h5_source
from studies.poincare_gap_probes import GapProbeSeeds, build_gap_probes
from studies.poincare_rho_sweep import (
	PreparedRhoStar, RhoStarConfig, build_rho_star, rho_config_from_metadata,
)


@lru_cache(maxsize=4)
def _source_field(source: Path, fingerprint: str, source_mtime_ns: int, source_size: int,
	magnetic_field: float, characteristic_length: float, source_selection: tuple[int, ...],
	interpolation_order: int) -> DimensionalH5Field:
	"""Restore original unsmoothed samples, spatial span and amplitude arithmetic."""
	# Fingerprint and file-stat arguments invalidate the bounded field cache if
	# the input changes. Rho and numerical method do not change this common field.
	raw: Potential = Potential.load(source, indx=source_selection,
		sigma=None, interpolation_order=interpolation_order)
	provenance: object = raw.metadata
	if not isinstance(provenance, GC2DH5Metadata):
		raise ValueError('Historical reproduction requires original HDF5 provenance.')
	if provenance.source_frequencies.size != 1:
		raise ValueError('Historical Poincare reproduction requires one forcing frequency.')
	x: np.ndarray = provenance.source_x
	y: np.ndarray = provenance.source_y
	period: float = float(x.size * (x[1] - x[0]))
	grid: Grid = Grid(float(x[0]), float(y[0]), float(x[1] - x[0]),
		float(y[1] - y[0]), x.size, y.size, period)
	reference: float = float(provenance.source_frequencies[0])
	clock: float = float(2 * np.pi / reference)
	# Match the original import followed by dimensional reconstruction, including
	# floating-point operation order. These factors are not new normalizations.
	divisor: float = reference * characteristic_length**2 * magnetic_field / (2 * np.pi)**2
	mean: np.ndarray = (raw.mean / divisor) * divisor
	modes: np.ndarray = (raw.modes / divisor) * divisor
	frequencies: np.ndarray = provenance.source_frequencies / reference
	physical: Potential = Potential(grid, mean=mean, modes=modes,
		frequencies=frequencies, interpolation_order=interpolation_order)
	flow_scale: float = clock / magnetic_field
	stream: Potential = Potential(grid, mean=mean * flow_scale, modes=modes * flow_scale,
		frequencies=frequencies, interpolation_order=interpolation_order)
	return DimensionalH5Field(physical, stream, clock, magnetic_field,
		tuple(int(value) for value in provenance.source_field_indices))


def _recorded_field(source: Path, digest: str, record: Mapping[str, Any], config: RhoStarConfig) -> DimensionalH5Field:
	"""Check each archive against shared reconstruction before applying its rho."""
	stat: Any = source.stat()
	field: DimensionalH5Field = _source_field(source, digest, stat.st_mtime_ns, stat.st_size,
		config.magnetic_field, config.characteristic_length, config.source_selection, config.interpolation_order)
	if field.source_indices != tuple(record['source_field_indices']):
		raise ValueError('Recorded source fields disagree with the reconstructed selection.')
	grid: Grid = field.raw.grid
	if not np.array_equal(np.asarray(record['cell_bounds_m']), (grid.x0, grid.y0, grid.period)):
		raise ValueError('Recorded cell bounds do not match the historical x-axis grid.')
	if field.cycle_seconds != record['time_unit_seconds']:
		raise ValueError('Recorded clock does not match the historical forcing convention.')
	return field


def prepare_horizontal_reproduction(
	source: str | Path, metadata: Mapping[str, Any], *,
	cycles: int | None = None, steps_per_cycle: int | None = None,
) -> PreparedRhoStar:
	"""Restore one original star or probe group, optionally shortening validation.

	Accept either the saved metadata document or its ``experiment`` mapping.
	Coordinate arrays are compared exactly with the archived values before use.
	The returned problem contains the archived unwrapped physical initial vector;
	its request is supplied separately through ``PreparedRhoStar.request``.
	"""
	record: dict[str, Any] = dict(metadata.get('experiment', metadata))
	config: RhoStarConfig = rho_config_from_metadata(record)
	if (config.spatial_normalization != 'none' or config.hamiltonian_convention != 'radial'
		or record.get('position_wrapping') != 'none' or record.get('space_unit') != 'm'):
		raise ValueError('Reproduction requires the archived unwrapped radial study in meters.')
	if record.get('method', 'BM4Midpoint') != config.method:
		raise ValueError('The archived method and configuration disagree.')
	resolved: Path = resolve_h5_source(source)
	with resolved.open('rb') as handle:
		digest: str = file_digest(handle, 'sha256').hexdigest()
	if digest != record['source_sha256']:
		raise ValueError('HDF5 source fingerprint differs from the archived experiment.')
	field: DimensionalH5Field = _recorded_field(resolved, digest, record, config)
	prepared: PreparedRhoStar
	if 'probe_config' in record:
		prepared = build_gap_probes(field, config, source_sha256=digest,
			seeds=GapProbeSeeds(**record['probe_config']))
	else:
		prepared = build_rho_star(field, config, source_sha256=digest)
	# Exact comparisons identify changed geometry rather than treating a tiny
	# initial displacement as harmless in a long chaotic trajectory.
	geometry: tuple[str, ...] = ('cell_bounds', 'cell_bounds_m', 'initial_positions',
		'initial_positions_m', 'particle_id', 'initial_radii_m', 'initial_radius_fraction',
		'arm_id', 'initial_positions_cell_fraction')
	for key in geometry:
		if key in record and (key not in prepared.metadata or not np.array_equal(record[key], prepared.metadata[key])):
			raise ValueError(f'Historical reconstruction differs from the archived {key}.')
	xy: np.ndarray = np.asarray(record['initial_positions'], dtype=float)
	initial: np.ndarray = GuidingCenterDynamics.coordinates.pack_components(xy[:, 0], xy[:, 1])
	problem: InitialValueProblem = InitialValueProblem(prepared.problem.dynamics,
		initial, layout=GuidingCenterDynamics.coordinates)
	original_config: dict[str, Any] = asdict(config)
	if cycles is not None or steps_per_cycle is not None:
		config = replace(config, cycles=config.cycles if cycles is None else cycles,
			steps_per_cycle=config.steps_per_cycle if steps_per_cycle is None else steps_per_cycle)
		prepared = replace(prepared, config=config,
			request=replace(prepared.request, t_span=(0., float(config.cycles)),
				output_times=np.arange(config.cycles + 1, dtype=float)))
	updated: dict[str, Any] = {**prepared.metadata,
		'config': asdict(config), 'expected_step_count': config.cycles * config.steps_per_cycle,
		'saved_cycles': config.cycles,
		'historical_reproduction': {
			'source_sha256': digest,
			'original_config': original_config,
			'archived_initial_positions_exact': True,
			'archived_geometry_exact': True,
			'field_recipe': 'Original normalization roundtrip, x-axis period, then T0/B and radial 1/(2*pi).',
			'grouping': 'eight_probes' if 'probe_config' in record else 'forty_star',
			'legacy_implementation_reference': '278d964',
			'bitwise_trajectory_reproduction_claimed': False,
		},
	}
	return replace(prepared, problem=problem, metadata=updated)


__all__ = ['prepare_horizontal_reproduction']
