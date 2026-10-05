"""Inspect completed rho runs in their saved spatial coordinate system."""

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

from diagnostics.persistence import StoredSolution, load_solution
from diagnostics.storage import ArtifactStore
from studies.poincare_rho_sweep import (
    PreparedRhoStar, RhoStarConfig, folded_rho_positions, prepare_rho_star, sample_rho_dynamics,
)
from visualization.poincare_comparison import export_poincare_panel_comparison
from visualization.poincare_gap_overlay import (
    append_gap_probe_coordinates, gap_probe_presentation, validate_gap_probe_overlay,
)


def _saved_method(record, diagnostics):
    """Check the archived method identity before labeling its trajectories."""
    method = record['config'].get('method', 'BM4Midpoint')
    study = record.get('study')
    valid_study = study == 'poincare_ranked_star_rho_sweep' or (
        method == 'BM4Midpoint' and study == 'poincare_bm4midpoint_ranked_star_rho_sweep')
    if not valid_study or record.get('method', 'BM4Midpoint') != method:
        raise ValueError('Saved study and method metadata do not agree.')
    if method == 'BM4Midpoint':
        valid = diagnostics.get('projection_kind') == 'arithmetic_mean'
    elif method == 'BM4Implicit':
        valid = (diagnostics.get('projection_solver_formulation') == 'bm4_implicit_reduced'
                 and diagnostics.get('nonlinear_solves_per_step') == 1
                 and diagnostics.get('nonlinear_solver') == 'newton'
                 and 'projection_kind' not in diagnostics)
    elif method == 'GaussLegendre4':
        valid = (diagnostics.get('stage_count') == 2
                 and diagnostics.get('designed_order') == 4
                 and diagnostics.get('nonlinear_solver') == 'newton'
                 and diagnostics.get('nonlinear_solves_per_step') == 1
                 and not any(key.startswith('projection_') for key in diagnostics))
    elif method == 'RK4':
        valid = not any(key.startswith(('projection_', 'nonlinear_')) for key in diagnostics)
    else:
        valid = False
    if not valid:
        raise ValueError('Saved diagnostics do not match the requested star method.')
    return method


def _rho_labels(rho_values):
    """Preserve exact rho selection while keeping familiar two-decimal labels."""
    labels = {}
    for rho in rho_values:
        rounded = f'{rho:.2f}'
        # Round-trip exact labels prevent collisions and misleading rounded
        # values when a requested grid does not have hundredth-size spacing.
        labels[rho] = rounded if float(rounded) == float(rho) else repr(float(rho))
    return labels


def _coordinate_metadata(metadata):
    """Read runtime geometry, including older archives that only stored meters."""
    normalized = metadata['spatial_normalization'] == 'characteristic_length'
    if metadata['spatial_normalization'] not in ('none', 'characteristic_length'):
        raise ValueError('Unknown spatial normalization.')
    return {
        'initial': np.asarray(metadata.get('initial_positions', metadata['initial_positions_m'])),
        'cell': np.asarray(metadata.get('cell_bounds', metadata['cell_bounds_m'])),
        'center': np.asarray(metadata.get('star_center', metadata['star_center_m'])),
        'radii': np.asarray(metadata.get('initial_radii', metadata['initial_radii_m'])),
        'axes': ('R_hat', 'Z_hat') if normalized else ('R (m)', 'Z (m)'),
        'normalized': normalized,
    }


def radius_colors(count: int, *, colormap: str = 'viridis',
                  color_range: tuple[float, float] = (0., 1.)) -> list[str]:
    """Assign a fixed, ordered color scale to increasing initial radial ranks."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    return [to_hex(colormaps[colormap](value)) for value in np.linspace(*color_range, count)]


def plot_initial_rho_star(prepared: PreparedRhoStar):
    """Show the runtime cell and radial particle IDs before cloud submission."""
    import matplotlib.pyplot as plt
    metadata = prepared.metadata
    coords = _coordinate_metadata(metadata)
    xy = coords['initial']
    colors = radius_colors(len(xy))
    x0, y0, length = coords['cell']
    center = coords['center']
    fig, ax = plt.subplots(figsize=(8, 7))
    for arm in range(prepared.config.arms):
        angle = prepared.config.first_angle + 2 * np.pi * arm / prepared.config.arms
        end = center + coords['radii'][-1] * np.array([np.cos(angle), np.sin(angle)])
        ax.plot([center[0], end[0]], [center[1], end[1]], color='#dddddd', lw=1)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=36)
    for particle, point in enumerate(xy, 1):
        ax.annotate(str(particle), point, xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set(xlabel=coords['axes'][0], ylabel=coords['axes'][1], xlim=(x0, x0 + length), ylim=(y0, y0 + length),
           title='Initial star: particle IDs increase with distance from the center')
    ax.set_aspect('equal'); ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def initial_star_table(prepared: PreparedRhoStar) -> str:
    """Tabulate IDs, arms, physical radii and runtime coordinates."""
    m = prepared.metadata
    coords = _coordinate_metadata(m)
    rows = [f"| Particle | Arm | Radius / cell radius | Radius (m) | {coords['axes'][0]} | {coords['axes'][1]} |",
            '|---:|---:|---:|---:|---:|---:|']
    for pid, arm, fraction, radius, xy in zip(m['particle_id'], m['arm_id'],
            m['initial_radius_fraction'], m['initial_radii_m'], coords['initial']):
        rows.append(f'| {pid} | {arm} | {100*fraction:.3f}% | {radius:.8f} | {xy[0]:.8f} | {xy[1]:.8f} |')
    return '\n'.join(rows)


def load_available_rho_runs(destinations: Mapping[float, str]) -> dict[float, StoredSolution]:
    """Load verified completed archives; missing rho runs stay unavailable."""
    runs = {}
    for rho, destination in destinations.items():
        if ArtifactStore(destination).exists('manifest.json'):
            stored = load_solution(destination)
            if not np.isclose(stored.metadata['config']['rho_hat'], rho, rtol=0, atol=1e-14):
                raise ValueError('The archive rho does not match its requested label.')
            runs[rho] = stored
    if not runs:
        raise FileNotFoundError('No completed rho runs exist yet. Finish a calculation before exporting the viewer.')
    return runs


def export_rho_sweep(runs: Mapping[float, StoredSolution], path: str | Path, *,
                     rho_values: tuple[float, ...], selected_rho: float = 0.3,
                     cycles_per_frame: int = 25, fold_to_cell: bool = False,
                     colormap: str = 'viridis',
                     color_range: tuple[float, float] = (0., 1.),
                     source: str | Path | None = None, field_grid_size: int = 64,
                     vector_grid_size: int = 17, phase_steps: int = 50,
                     probe_runs: Mapping[float, StoredSolution] | None = None) -> Path:
    """Export saved cycles in their runtime units; disable unavailable rho values.

    Archives retain unwrapped float64 states. By default browser copies use
    float32 in runtime units. ``fold_to_cell`` instead displays periodic
    returns relative to the cell origin, divided by the cell width.
    ``colormap`` and ``color_range`` select colors in increasing radial order.
    Supply the original HDF5 ``source`` to show the effective Hamiltonian and
    vector field on the left, sampled at ``phase_steps + 1`` phases. Its hash
    must match the saved calculation. No trajectories are integrated here.
    Optional ``probe_runs`` append eight independently saved explicit seeds;
    the original star IDs, colors and trajectories are preserved.
    """
    if (not runs or not set(runs).issubset(rho_values)
            or len(set(rho_values)) != len(rho_values)
            or not np.isfinite(rho_values).all() or np.any(np.asarray(rho_values) < 0)):
        raise ValueError('Provide completed runs labeled by unique requested rho values.')
    rho_labels = _rho_labels(rho_values)
    if selected_rho not in runs:
        selected_rho = next(iter(runs))
    reference = runs[selected_rho]
    m = reference.metadata
    method = _saved_method(m, reference.solution.diagnostics)
    ids = np.asarray(m['particle_id'], dtype=int)
    colors = radius_colors(len(ids), colormap=colormap, color_range=color_range)
    star_ids = ids.copy()
    probes = None
    if probe_runs is not None:
        probes = gap_probe_presentation(validate_gap_probe_overlay(runs, probe_runs))
        ids = np.concatenate((star_ids, probes['particle_ids']))
        colors += probes['colors']
    coords = _coordinate_metadata(m)
    initial, cell = coords['initial'], coords['cell']
    lower, upper = cell[:2].copy(), cell[:2] + cell[2]
    coordinates = {}
    def comparable_config(record):
        """Treat JSON selector lists and in-memory selector tuples identically."""
        config = asdict(RhoStarConfig(**record['config']))
        config.pop('rho_hat')
        config['source_selection'] = tuple(config['source_selection'])
        return config

    expected_config = comparable_config(m)
    for rho, saved in runs.items():
        record, solution = saved.metadata, saved.solution
        if (_saved_method(record, solution.diagnostics) != method
                or record.get('space_unit') != ('dimensionless' if coords['normalized'] else 'm')
                or record.get('spatial_normalization') != m['spatial_normalization']
                or record.get('position_wrapping') != 'none'
                or record['source_sha256'] != m['source_sha256']
                or comparable_config(record) != expected_config
                or not np.isclose(record['config']['rho_hat'], rho, rtol=0, atol=1e-14)):
            raise ValueError('Only comparable star runs from one method with the same spatial units can share this viewer.')
        x, y = solution.positions()
        xy = np.stack((x.T, y.T), axis=-1)
        if xy.shape != (record['config']['cycles'] + 1, len(star_ids), 2) or not np.isfinite(xy).all():
            raise ValueError('The saved cycle coordinates are incomplete or invalid.')
        np.testing.assert_array_equal(solution.t, np.arange(record['config']['cycles'] + 1))
        np.testing.assert_array_equal(record['particle_id'], star_ids)
        np.testing.assert_allclose(xy[0], initial, rtol=0, atol=1e-14)
        np.testing.assert_allclose(_coordinate_metadata(record)['cell'], cell, rtol=0, atol=1e-14)
        if fold_to_cell:
            _, xy = folded_rho_positions(solution, record)
        if probe_runs is not None:
            xy = append_gap_probe_coordinates(xy, probe_runs[rho], fold_to_cell=fold_to_cell)
        coordinates[rho] = xy
        lower = np.minimum(lower, xy.min(axis=(0, 1)))
        upper = np.maximum(upper, xy.max(axis=(0, 1)))
    # One square viewport contains the cell and every unwrapped return.
    span = float(np.max(upper - lower)) * (1 + 1e-12)
    lower = (lower + upper) / 2 - span / 2
    bounds = (float(lower[0]), float(lower[1]), span)
    if fold_to_cell:
        bounds = (0., 0., 1.)
    view = bounds if fold_to_cell else tuple(cell)
    axes = ('(R - R0) / L', '(Z - Z0) / L') if fold_to_cell else coords['axes']
    datasets = []
    for rho in rho_values:
        item = {'key': rho_labels[rho], 'label': rho_labels[rho]}
        if rho in runs:
            common = dict(particle_ids=ids, colors=colors)
            item['panels'] = [
                dict(common, title=('Initial star and gap probes (cycle 0)' if probes else 'Initial star (cycle 0)'),
                     coordinates=coordinates[rho][:1], static=True),
                dict(common, title=f'{method} · rho = {rho_labels[rho]}', coordinates=coordinates[rho]),
            ]
            record = runs[rho].metadata
            if source is not None:
                prepared = prepare_rho_star(source, RhoStarConfig(**record['config']))
                if prepared.metadata['source_sha256'] != record['source_sha256']:
                    raise ValueError('The viewer source does not match the saved field fingerprint.')
                field = sample_rho_dynamics(prepared, grid_size=field_grid_size,
                                           vector_grid_size=vector_grid_size, phase_steps=phase_steps)
                # In cell fractions q=(X-X0)/L, Hq=HX/L² and dq/dtau=vX/L.
                if fold_to_cell:
                    field['potential'] /= cell[2]**2
                    field['velocity'] /= cell[2]
                    field['bounds'] = np.array((0., 0., 1.))
                field['units'] = ('cell² / cycle; velocity: cell / cycle' if fold_to_cell else
                                  ('q² / cycle; velocity: q / cycle' if coords['normalized'] else
                                   'm² / cycle; velocity: m / cycle'))
                item['panels'][0].update(title='Effective potential and GC velocity', field=field)
            item['note'] = (f"Physical gyro-radius: {record['rho_m']:.9g} m · "
                            f"{record['config']['cycles']:,} cycles · "
                            f"{record['config']['steps_per_cycle']} steps/cycle · coupling = "
                            f"{record['config']['coupling_frequency']:g} · Hamiltonian: "
                            f"{record['config'].get('hamiltonian_convention', 'cycle_time')}")
        datasets.append(item)
    selected = next(d for d in datasets if d['key'] == rho_labels[selected_rho])
    labels = {int(pid): f"#{pid} · {100*fraction:.2f}% · {1000*radius:.2f} mm"
              for pid, fraction, radius in zip(star_ids, m['initial_radius_fraction'], m['initial_radii_m'])}
    groups = [dict(label=f'Arm {arm}', particle_ids=star_ids[np.asarray(m['arm_id']) == arm])
              for arm in range(1, m['config']['arms'] + 1)]
    particle_title = f'{len(star_ids)}-particle star'
    if probes:
        labels.update(probes['labels'])
        groups += probes['groups']
        particle_title += f" + {len(probes['particle_ids'])} gap probes"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return export_poincare_panel_comparison(
        path, selected['panels'], cycles_per_frame=cycles_per_frame,
        title=f'{method} · {particle_title} · rho sweep' + (' · folded cell' if fold_to_cell else
              (' · normalized space' if coords['normalized'] else '')),
        coordinate_bounds=bounds, axis_labels=axes,
        initial_view=view, initial_cycle=m['config']['cycles'], first_cycle=0,
        datasets=datasets, dataset_label='rho (usual dimensionless value)',
        selected_dataset=rho_labels[selected_rho], particle_labels=labels,
        particle_groups=groups,
        description=(f"{len(runs)} / {len(rho_values)} rho values available. "
            + ('Display positions are folded into one periodic cell and divided by its width, '
               'relative to its lower corner. Integrated states remain unwrapped. '
               if fold_to_cell else ('Positions use q = 2*pi*(X-X0)/lambda, lambda = '
               f"{m['config']['characteristic_length']:g} m; time remains tau = t/T0. No wrapping. "
               if coords['normalized'] else 'Positions are in meters, with no spatial normalization or wrapping. '))
            +
            'Original star IDs and colors follow increasing initial distance (0–85% of the cell radius). '
            + (probes['description'] if probes else '') +
            'Hollow circles mark each selected particle at Start cycle (0 is the original initial state). '
             + ('The left panel shows the rho-dependent effective Hamiltonian and GC velocity; '
               'Phase step explores one forcing cycle including its identical endpoints. '
               if source is not None else 'The left panel shows the original initial star. ') +
            'Only integer-cycle returns are shown. Play/Pause, interval sliders and particle selection '
            'control the section. Drag to zoom; Shift-drag or right-drag to pan. '
            + ('Full view and Selected region restore the periodic cell.' if fold_to_cell else
               'Full view includes every saved position; Selected region restores the original cell.')),
    )


__all__ = ['radius_colors', 'plot_initial_rho_star', 'initial_star_table',
           'load_available_rho_runs', 'export_rho_sweep']
