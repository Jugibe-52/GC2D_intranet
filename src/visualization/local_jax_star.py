"""Read-only initial geometry, block timing and Poincare views of local runs."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np

from diagnostics.persistence import StoredSolution
from visualization.poincare_comparison import export_poincare_panel_comparison


def _check(saved: StoredSolution) -> None:
    """Require the matching archive, complete sample schedule and field."""
    if saved.metadata.get('study') != 'local_jax_bm4midpoint_star' or saved.potential is None:
        raise ValueError('Load a complete local JAX BM4Midpoint star archive.')
    config = saved.metadata['config']
    count = config['arms'] * config['particles_per_arm'] + int(config['include_center'])
    steps = config['cycles'] * config['steps_per_cycle']
    if saved.solution.states.shape != (2 * count, steps + 1):
        raise ValueError('The archive does not contain every requested state.')
    np.testing.assert_array_equal(saved.solution.t, np.arange(steps + 1) / config['steps_per_cycle'])


def local_star_summary(saved: StoredSolution) -> str:
    """Describe only values from the saved record."""
    _check(saved)
    config = saved.metadata['config']
    return '\n'.join((
        'BM4Midpoint / local JAX CPU / float64',
        f'Particles: {len(saved.metadata["particle_ids"])}; arms: {config["arms"]}; shared center: {config["include_center"]}',
        f'Cycles: {config["cycles"]}; steps per cycle: {config["steps_per_cycle"]}; saved states: {saved.solution.t.size}',
        f'R/L: {config["radius_over_period"]}; positive radii / R: {saved.metadata["positive_radii_over_R"]}',
        f'Integration including compilation: {saved.metadata["integration_seconds"]:.3f} s',
        'Poincare returns are selected at integer cycles without repeating the integration.',
    ))


def plot_local_star(saved: StoredSolution) -> Any:
    """Plot initial positions and completed-block timings from saved data."""
    _check(saved)
    assert saved.potential is not None
    grid = saved.potential.grid
    initial = (np.asarray(saved.metadata['initial_positions']) - [grid.x0, grid.y0]) / grid.period
    arms = np.asarray(saved.metadata['arm_ids'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for arm in np.unique(arms):
        selected = arms == arm
        axes[0].plot(initial[selected, 0], initial[selected, 1], 'o-',
                     color='black' if arm < 0 else plt.get_cmap('tab10')(int(arm) % 10),
                     label='Shared center' if arm < 0 else f'Arm {arm + 1}', ms=4)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='x / L', ylabel='y / L', title='Initial star')
    axes[0].legend(fontsize=8, ncol=3, loc='upper center')
    diagnostics = saved.solution.diagnostics
    axes[1].plot(diagnostics['execution_block_end_cycles'], diagnostics['execution_block_wall_seconds'], '.-')
    axes[1].set(xlabel='Completed forcing cycles', ylabel='Seconds per integration block',
                title='Local block timings (first includes compilation)')
    for ax in axes:
        ax.grid(alpha=.2)
    return fig


def export_local_star_poincare(saved: StoredSolution, path: str | Path, *, cycles_per_frame: int = 25) -> Path:
    """Export the fixed initial geometry and every saved integer-cycle return."""
    _check(saved)
    assert saved.potential is not None
    metadata, grid = saved.metadata, saved.potential.grid
    config = metadata['config']
    ids, arms = np.asarray(metadata['particle_ids']), np.asarray(metadata['arm_ids'])
    colors = ['#111111' if arm < 0 else to_hex(plt.get_cmap('tab10')(int(arm) % 10)) for arm in arms]
    # Only display arrays are wrapped and converted to the viewer's float32.
    x, y = saved.solution.positions()
    step = config['steps_per_cycle']
    positions = np.stack(((x[:, ::step].T - grid.x0) / grid.period % 1,
                          (y[:, ::step].T - grid.y0) / grid.period % 1), axis=-1)
    common = dict(particle_ids=ids, colors=colors)
    panels = [dict(common, title='Initial star', coordinates=positions[:1], static=True),
              dict(common, title='BM4Midpoint: local JAX CPU', coordinates=positions[1:])]
    groups = [dict(label='Shared center' if arm < 0 else f'Arm {arm + 1}',
                   particle_ids=ids[arms == arm]) for arm in np.unique(arms)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_poincare_panel_comparison(path, panels, cycles_per_frame=cycles_per_frame,
        initial_cycle=min(25, config['cycles']), particle_groups=groups,
        title=f'Local JAX star: {len(ids)} particles, {config["cycles"]} cycles',
        axis_labels=('x / L', 'y / L'),
        description='A single central particle and equally spaced positive arm radii. '
                    'Coordinates are folded into the periodic cell for display. '
                    'The initial panel stays fixed; the trajectory panel shows integer-cycle returns. '
                    'Select the center or an arm, drag to zoom, and Shift-drag to pan.')
    return path


__all__ = ['local_star_summary', 'plot_local_star', 'export_local_star_poincare']
