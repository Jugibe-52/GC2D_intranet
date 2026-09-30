"""Inspect completed rho runs in their saved spatial coordinate system."""

from pathlib import Path
from typing import Mapping

import numpy as np

from diagnostics.persistence import StoredSolution, load_solution
from diagnostics.storage import ArtifactStore
from studies.poincare_rho_sweep import PreparedRhoStar
from visualization.poincare_comparison import export_poincare_panel_comparison


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


def radius_colors(count: int) -> list[str]:
    """Assign a fixed, ordered color scale to increasing initial radial ranks."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    return [to_hex(colormaps['viridis'](value)) for value in np.linspace(0, 1, count)]


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
                     cycles_per_frame: int = 25) -> Path:
    """Export saved cycles in their runtime units; disable unavailable rho values.

    Archives are float64 and unwrapped. Browser copies use float32 without
    changing units or applying periodic folding.
    """
    if not runs or not set(runs).issubset(rho_values) or len(set(rho_values)) != len(rho_values):
        raise ValueError('Provide completed runs labeled by unique requested rho values.')
    if selected_rho not in runs:
        selected_rho = next(iter(runs))
    reference = runs[selected_rho]
    m = reference.metadata
    ids = np.asarray(m['particle_id'], dtype=int)
    colors = radius_colors(len(ids))
    coords = _coordinate_metadata(m)
    initial, cell = coords['initial'], coords['cell']
    lower, upper = cell[:2].copy(), cell[:2] + cell[2]
    coordinates = {}
    def comparable_config(record):
        """Treat JSON selector lists and in-memory selector tuples identically."""
        config = {k: v for k, v in record['config'].items() if k != 'rho_hat'}
        config.setdefault('spatial_normalization', 'none')
        config['source_selection'] = tuple(config['source_selection'])
        return config

    expected_config = comparable_config(m)
    for rho, saved in runs.items():
        record, solution = saved.metadata, saved.solution
        if (record.get('study') != 'poincare_bm4midpoint_ranked_star_rho_sweep'
                or record.get('space_unit') != ('dimensionless' if coords['normalized'] else 'm')
                or record.get('spatial_normalization') != m['spatial_normalization']
                or record.get('position_wrapping') != 'none'
                or record['source_sha256'] != m['source_sha256']
                or comparable_config(record) != expected_config
                or not np.isclose(record['config']['rho_hat'], rho, rtol=0, atol=1e-14)
                or solution.diagnostics.get('projection_kind') != 'arithmetic_mean'):
            raise ValueError('Only comparable BM4Midpoint star runs with the same spatial units can share this viewer.')
        x, y = solution.positions()
        xy = np.stack((x.T, y.T), axis=-1)
        if xy.shape != (record['config']['cycles'] + 1, len(ids), 2) or not np.isfinite(xy).all():
            raise ValueError('The saved cycle coordinates are incomplete or invalid.')
        np.testing.assert_array_equal(solution.t, np.arange(record['config']['cycles'] + 1))
        np.testing.assert_array_equal(record['particle_id'], ids)
        np.testing.assert_allclose(xy[0], initial, rtol=0, atol=1e-14)
        np.testing.assert_allclose(_coordinate_metadata(record)['cell'], cell, rtol=0, atol=1e-14)
        coordinates[rho] = xy
        lower = np.minimum(lower, xy.min(axis=(0, 1)))
        upper = np.maximum(upper, xy.max(axis=(0, 1)))
    # One square viewport contains the cell and every unwrapped return.
    span = float(np.max(upper - lower)) * (1 + 1e-12)
    lower = (lower + upper) / 2 - span / 2
    bounds = (float(lower[0]), float(lower[1]), span)
    datasets = []
    for rho in rho_values:
        item = {'key': f'{rho:.2f}', 'label': f'{rho:.2f}'}
        if rho in runs:
            common = dict(particle_ids=ids, colors=colors)
            item['panels'] = [
                dict(common, title='Initial star (cycle 0)', coordinates=coordinates[rho][:1], static=True),
                dict(common, title=f'BM4Midpoint · rho = {rho:.2f}', coordinates=coordinates[rho][1:]),
            ]
            record = runs[rho].metadata
            item['note'] = (f"Physical gyro-radius: {record['rho_m']:.9g} m · "
                            f"{record['config']['cycles']:,} cycles · "
                            f"{record['config']['steps_per_cycle']} steps/cycle · coupling = "
                            f"{record['config']['coupling_frequency']:g}")
        datasets.append(item)
    selected = next(d for d in datasets if d['key'] == f'{selected_rho:.2f}')
    labels = {int(pid): f"#{pid} · {100*fraction:.2f}% · {1000*radius:.2f} mm"
              for pid, fraction, radius in zip(ids, m['initial_radius_fraction'], m['initial_radii_m'])}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return export_poincare_panel_comparison(
        path, selected['panels'], cycles_per_frame=cycles_per_frame,
        title='BM4Midpoint · 40-particle star · rho sweep' + (' · normalized space' if coords['normalized'] else ''),
        coordinate_bounds=bounds, axis_labels=coords['axes'],
        initial_view=tuple(cell), initial_cycle=m['config']['cycles'],
        datasets=datasets, dataset_label='rho (usual dimensionless value)',
        selected_dataset=f'{selected_rho:.2f}', particle_labels=labels,
        particle_groups=[dict(label=f'Arm {arm}', particle_ids=ids[np.asarray(m['arm_id']) == arm])
                         for arm in range(1, m['config']['arms'] + 1)],
        description=(f"{len(runs)} / {len(rho_values)} rho values available. "
            + ('Positions use q = 2*pi*(X-X0)/lambda, lambda = '
               f"{m['config']['characteristic_length']:g} m; time remains tau = t/T0. No wrapping. "
               if coords['normalized'] else 'Positions are in meters, with no spatial normalization or wrapping. ')
            +
            'IDs and colors follow increasing initial distance (0–85% of the cell radius). '
            'Only integer-cycle returns are shown. Play/Pause, interval sliders and particle selection '
            'control the section. Drag to zoom; Shift-drag or right-drag to pan. '
            'Full view includes every saved position; Selected region restores the original cell.'),
    )


__all__ = ['radius_colors', 'plot_initial_rho_star', 'initial_star_table',
           'load_available_rho_runs', 'export_rho_sweep']
