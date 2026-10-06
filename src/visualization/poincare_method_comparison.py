"""Compare four saved rho-star studies with shared controls and one field."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from solution import Solution
from diagnostics.persistence import StoredSolution
from studies.poincare_rho_sweep import (
    RhoStarConfig, folded_rho_positions, prepare_rho_star, sample_rho_dynamics,
    validate_rho_solution,
)
from visualization.poincare_comparison import export_poincare_panel_comparison
from visualization.poincare_gap_overlay import (
    append_gap_probe_coordinates, gap_probe_presentation, validate_gap_probe_overlay,
)
from visualization.poincare_rho_sweep import (
    _coordinate_metadata, _rho_labels, _saved_method, load_available_rho_runs,
    radius_colors,
)


_METHODS = ('BM4Midpoint', 'BM4Implicit', 'RK4', 'GaussLegendre4')
_IMPLICIT_METHODS = ('BM4Implicit', 'GaussLegendre4')


def load_rho_method_runs(
    destinations_by_method: Mapping[str, Mapping[float, str]], *, max_workers: int = 4,
) -> dict[str, dict[float, StoredSolution]]:
    """Download completed archives concurrently across the four saved studies."""
    if set(destinations_by_method) != set(_METHODS):
        raise ValueError('Provide destinations for all four supported star methods.')
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {method: pool.submit(load_available_rho_runs, destinations_by_method[method])
                   for method in _METHODS}
        return {method: future.result() for method, future in futures.items()}


def _comparison_config(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate shared scientific inputs from method identity and Newton controls."""
    config = asdict(RhoStarConfig(**record['config']))
    controls = {name: config.pop(name) for name in tuple(config) if name.startswith('newton_')}
    config.pop('method')
    config.pop('rho_hat')
    config['source_selection'] = tuple(config['source_selection'])
    return config, controls


def _require_matching_metadata(record: Mapping[str, Any], expected: Mapping[str, Any], *,
                               include_rho: bool = False) -> None:
    """Check saved geometry, time units and physical provenance before overlaying."""
    scalar_keys = ('source_sha256', 'space_unit', 'spatial_normalization',
                   'position_wrapping', 'time_unit_seconds')
    array_keys = ('source_field_indices', 'particle_id', 'arm_id',
                  'initial_positions_m', 'cell_bounds_m', 'initial_radius_fraction',
                  'initial_radii_m')
    if any(record.get(key) != expected.get(key) for key in scalar_keys):
        raise ValueError('Compared runs must share physical settings, field fingerprint and time units.')
    try:
        for key in array_keys:
            np.testing.assert_allclose(record[key], expected[key], rtol=0, atol=1e-14)
        for key in ('initial', 'cell', 'center', 'radii'):
            np.testing.assert_allclose(_coordinate_metadata(record)[key],
                                       _coordinate_metadata(expected)[key], rtol=0, atol=1e-14)
        if include_rho:
            for key in ('rho_m', 'rho_runtime'):
                np.testing.assert_allclose(record[key], expected[key], rtol=0, atol=1e-14)
    except (AssertionError, KeyError) as error:
        raise ValueError('Compared runs must share the saved geometry and physical parameters.') from error


def _available_comparison_rhos(
    runs_by_method: Mapping[str, Mapping[float, StoredSolution]], rho_values: tuple[float, ...],
) -> set[float]:
    """Require all supported methods to expose one common nonempty requested rho set."""
    if set(runs_by_method) != set(_METHODS):
        raise ValueError('Provide saved runs for all four supported star methods.')
    available = set(runs_by_method[_METHODS[0]])
    if (not available or not available.issubset(rho_values)
            or len(set(rho_values)) != len(rho_values)
            or not np.isfinite(rho_values).all() or np.any(np.asarray(rho_values) < 0)
            or any(set(runs_by_method[method]) != available for method in _METHODS)):
        raise ValueError('All four methods must share a nonempty set of unique requested rho values.')
    return available


def _validated_method_config(
    record: Mapping[str, Any], solution: Solution, method: str, rho: float,
    shared_config: Mapping[str, Any], implicit_controls: Mapping[str, Any],
    reference_metadata: Mapping[str, Any], prepared_metadata: Mapping[str, Any],
) -> RhoStarConfig:
    """Check method identity, scientific controls and provenance before overlaying a run."""
    config = RhoStarConfig(**record['config'])
    actual_config, controls = _comparison_config(record)
    if (_saved_method(record, solution.diagnostics) != method
            or actual_config != shared_config
            or not np.isclose(config.rho_hat, rho, rtol=0, atol=1e-14)):
        raise ValueError('Compared methods must share scientific inputs and correctly labeled rho values.')
    if method in _IMPLICIT_METHODS and controls != implicit_controls:
        raise ValueError('Compared implicit methods must share Newton controls.')
    _require_matching_metadata(record, reference_metadata)
    _require_matching_metadata(record, prepared_metadata, include_rho=True)
    return config


def export_rho_method_comparison(
    runs_by_method: Mapping[str, Mapping[float, StoredSolution]], path: str | Path, *,
    source: str | Path, rho_values: tuple[float, ...], selected_rho: float = 0.3,
    cycles_per_frame: int = 25, field_grid_size: int = 64,
    vector_grid_size: int = 17, phase_steps: int = 50,
    colormap: str = 'turbo', color_range: tuple[float, float] = (0.025, 0.975),
    probe_runs_by_method: Mapping[str, Mapping[float, StoredSolution]] | None = None,
) -> Path:
    """Export four horizontal method panels and a field below particle selection.

    Every saved integer-cycle state, including cycle zero, is retained. Browser
    positions are folded cell fractions; source archives remain unchanged.
    The original HDF5 field is verified and sampled once per available rho,
    with no trajectory integrations. All four methods must supply the same
    available rho set. Additional requested rho values appear unavailable.
    Optional probe archives add the same eight explicit seeds to each method,
    retaining the original star palette and all independently saved returns.
    """
    available = _available_comparison_rhos(runs_by_method, rho_values)
    labels = _rho_labels(rho_values)
    if selected_rho not in available:
        selected_rho = next(rho for rho in rho_values if rho in available)
    reference = runs_by_method[_METHODS[0]][selected_rho]
    metadata = reference.metadata
    shared_config, _ = _comparison_config(metadata)
    _, implicit_controls = _comparison_config(runs_by_method['BM4Implicit'][selected_rho].metadata)
    ids = np.asarray(metadata['particle_id'], dtype=int)
    colors = radius_colors(len(ids), colormap=colormap, color_range=color_range)
    star_ids = ids.copy()
    probes = None
    if probe_runs_by_method is not None:
        if set(probe_runs_by_method) != set(_METHODS):
            raise ValueError('Provide gap probes for all four supported methods.')
        seeds = None
        for method in _METHODS:
            seeds = validate_gap_probe_overlay(runs_by_method[method], probe_runs_by_method[method],
                                              expected_seeds=seeds)
        assert seeds is not None
        probes = gap_probe_presentation(seeds)
        ids = np.concatenate((star_ids, probes['particle_ids']))
        colors += probes['colors']
    datasets: list[dict[str, Any]] = []
    for rho in rho_values:
        dataset: dict[str, Any] = {'key': labels[rho], 'label': labels[rho]}
        if rho not in available:
            datasets.append(dataset)
            continue
        panels: list[dict[str, Any]] = []
        # Preparing the first method only reconstructs the physical field and
        # initial condition. It does not invoke an execution backend.
        record = runs_by_method[_METHODS[0]][rho].metadata
        prepared = prepare_rho_star(source, RhoStarConfig(**record['config']))
        _require_matching_metadata(record, prepared.metadata, include_rho=True)
        for method in _METHODS:
            saved = runs_by_method[method][rho]
            record, solution = saved.metadata, saved.solution
            config = _validated_method_config(
                record, solution, method, rho, shared_config, implicit_controls, metadata, prepared.metadata,
            )
            # Reuse the study's executable method and cycle-sampling contract
            # against the independently reconstructed common initial state.
            validate_rho_solution(solution, replace(prepared, config=config))
            _, xy = folded_rho_positions(solution, record)
            if probe_runs_by_method is not None:
                xy = append_gap_probe_coordinates(xy, probe_runs_by_method[method][rho], fold_to_cell=True)
            panels.append(dict(title=method, coordinates=xy, particle_ids=ids, colors=colors))
        cell = _coordinate_metadata(record)['cell']
        field = sample_rho_dynamics(prepared, grid_size=field_grid_size,
                                    vector_grid_size=vector_grid_size, phase_steps=phase_steps)
        # With q=(X-X0)/L and unchanged cycle time, Hq=HX/L² and dq/dtau=vX/L.
        field['potential'] /= cell[2]**2
        field['velocity'] /= cell[2]
        field['bounds'] = np.array((0., 0., 1.))
        field['units'] = 'cell² / cycle; velocity: cell / cycle'
        panels.append(dict(title='Effective potential and GC velocity',
                           coordinates=panels[0]['coordinates'][:1], particle_ids=ids,
                           colors=colors, static=True, field=field))
        dataset['panels'] = panels
        dataset['note'] = (f"Four methods · {config.cycles:,} cycles · {config.steps_per_cycle} steps/cycle · "
                           f"physical gyro-radius: {record['rho_m']:.9g} m · "
                           f"Hamiltonian: {config.hamiltonian_convention}")
        datasets.append(dataset)
    selected = next(item for item in datasets if item['key'] == labels[selected_rho])
    particle_labels = {
        int(pid): f"#{pid} · {100*fraction:.2f}% · {1000*radius:.2f} mm"
        for pid, fraction, radius in zip(star_ids, metadata['initial_radius_fraction'], metadata['initial_radii_m'])
    }
    groups = [dict(label=f'Arm {arm}', particle_ids=star_ids[np.asarray(metadata['arm_id']) == arm])
              for arm in range(1, shared_config['arms'] + 1)]
    particle_title = f'{len(star_ids)}-particle star'
    if probes:
        particle_labels.update(probes['labels'])
        groups += probes['groups']
        particle_title += f" + {len(probes['particle_ids'])} gap probes"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return export_poincare_panel_comparison(
        path, selected['panels'], cycles_per_frame=cycles_per_frame,
        title=f'Four methods · {particle_title} · synchronized rho sweep',
        coordinate_bounds=(0., 0., 1.), initial_view=(0., 0., 1.),
        axis_labels=('(R - R0) / L', '(Z - Z0) / L'),
        initial_cycle=shared_config['cycles'], first_cycle=0,
        datasets=datasets, dataset_label='rho (usual dimensionless value)',
        selected_dataset=labels[selected_rho], particle_labels=particle_labels,
        particle_groups=groups,
        field_placement='below_particles',
        description=(
            'BM4 midpoint, BM4 implicit, RK4 and Gauss–Legendre share rho, particles, '
            'Start cycle, current cycle, playback and viewport controls. Every saved return '
            'is retained, including cycle 0. Hollow circles mark each selected particle at '
            'Start cycle. Positions are folded into one periodic cell; original star IDs and colors follow '
            'increasing initial radius. ' + (probes['description'] if probes else '') +
            'The shared effective potential and physical GC '
            'velocity appear below particle selection. Phase explores one forcing cycle '
            'independently, including verified identical endpoints. '
            'Drag a plot to zoom; Shift-drag or right-drag to pan.'),
    )


__all__ = ['load_rho_method_runs', 'export_rho_method_comparison']
