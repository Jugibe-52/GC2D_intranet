"""Plots and measured summaries for the saved Modal CPU star comparison."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np

from potential.potential import Potential
from studies.modal_cpu_comparison import ModalCPUComparison, ModalCPUStarConfig, build_star_problem
from visualization.poincare_comparison import export_poincare_panel_comparison


def plot_initial_star(potential: Potential, config: ModalCPUStarConfig) -> Any:
    """Show the initial positions in cell-period units, colored by arm."""
    problem, _ = build_star_problem(potential, config)
    x, y = problem.layout.positions(problem.initial_state)
    grid = potential.grid
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    for arm in range(config.arms):
        selected = slice(arm * config.particles_per_arm, (arm + 1) * config.particles_per_arm)
        ax.plot((x[selected] - grid.x0) / grid.period, (y[selected] - grid.y0) / grid.period,
                'o-', ms=5, lw=1, label=f'Arm {arm + 1}')
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='x / L', ylabel='y / L',
           title=f'{problem.particle_count} particles; R = {config.radius_over_period:g} L\n'
                 f'r / R = {config.inner_radius_fraction:g} to {config.outer_radius_fraction:g}')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    return fig


def comparison_table(comparison: ModalCPUComparison) -> str:
    """Generate a Markdown table from measured worker records."""
    lines = ['| CPU cores | First run (s) | Warm median (s) | Warm IQR (s) | Speedup | Max periodic discrepancy |',
             '|---:|---:|---:|---:|---:|---:|']
    for cores, record in comparison.metadata['runs'].items():
        lines.append(f'| {cores} | {record["first_seconds"]:.3f} | {record["median_seconds"]:.6f} | '
                     f'{record["q25_seconds"]:.6f}–{record["q75_seconds"]:.6f} | '
                     f'{record["speedup"]:.2f}× | {record["maximum_periodic_discrepancy"]:.3e} |')
    fastest = min(comparison.metadata['runs'], key=lambda key: comparison.metadata['runs'][key]['median_seconds'])
    lines.extend(['', f'The lowest measured warm median used **{fastest} CPU core(s)**.', '',
                  'First-run time includes JAX setup and compilation. Warm times cover complete integrations '
                  f'with synchronized NumPy outputs; each configuration has {comparison.metadata["config"]["repetitions"]} measured repetitions. '
                  'Provisioning and transfers are excluded. These short-run measurements do not establish '
                  'long-run scaling or absolute trajectory accuracy.'])
    if any(record['environment'].get('cpu_model', 'unknown') in ('', 'unknown')
           for record in comparison.metadata['runs'].values()):
        lines.extend(['', 'Modal did not expose the processor model. Container placement and hardware '
                      'may contribute to the observed runtime differences.'])
    return '\n'.join(lines)


def plot_cpu_comparison(comparison: ModalCPUComparison) -> Any:
    """Compare complete-run timings and all saved states against the 1-core run."""
    cores = list(comparison.solutions)
    records = [comparison.metadata['runs'][str(core)] for core in cores]
    medians = np.array([r['median_seconds'] for r in records])
    errors = np.array([[r['median_seconds'] - r['q25_seconds'] for r in records],
                       [r['q75_seconds'] - r['median_seconds'] for r in records]])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axes[0].bar(range(len(cores)), medians, yerr=errors, capsize=5, color='#4486a4')
    axes[0].set(xticks=range(len(cores)), xticklabels=cores, xlabel='CPU cores',
                ylabel='Seconds per complete integration', title='Warm median and interquartile range')
    axes[1].bar(range(len(cores)), [r['first_seconds'] for r in records], color='#bd7944')
    axes[1].set(xticks=range(len(cores)), xticklabels=cores, xlabel='CPU cores',
                ylabel='Seconds', title='First integration, including JAX compilation')
    baseline = comparison.solutions[1]
    period = comparison.potential.grid.period
    for core, solution in comparison.solutions.items():
        delta = (solution.states - baseline.states + period / 2) % period - period / 2
        dx, dy = solution.layout.positions(delta)
        axes[2].plot(solution.t, np.max(np.hypot(dx, dy), axis=0), label=f'{core} core(s)')
    axes[2].set(xlabel='Normalized time / forcing cycles', ylabel='Maximum periodic distance',
                title='Agreement with the 1-core run')
    axes[2].legend()
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    return fig


def export_cpu_star_poincare(comparison: ModalCPUComparison, path: str | Path) -> Path:
    """Export saved integer-cycle returns; no trajectory calculation is performed."""
    metadata = comparison.metadata
    grid = comparison.potential.grid
    ids = metadata['particle_ids']
    arm_ids = np.asarray(metadata['arm_ids'])
    colors = [to_hex(plt.get_cmap('tab10')(int(arm) % 10)) for arm in arm_ids]
    initial = (np.asarray(metadata['initial_positions']) - [grid.x0, grid.y0]) / grid.period
    panels = [dict(coordinates=initial[None], particle_ids=ids, colors=colors,
                   title='Initial star', static=True)]
    step = metadata['config']['steps_per_cycle']
    for cores, solution in comparison.solutions.items():
        x, y = solution.layout.positions(solution.states[:, step::step])
        coordinates = np.stack(((x.T - grid.x0) / grid.period % 1,
                                (y.T - grid.y0) / grid.period % 1), axis=-1)
        panels.append(dict(coordinates=coordinates, particle_ids=ids, colors=colors,
                           title=f'JAX CPU: {cores} core(s)'))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_poincare_panel_comparison(path, panels, cycles_per_frame=1,
        initial_cycle=metadata['config']['cycles'], title='BM4Midpoint: Modal JAX CPU comparison',
        axis_labels=('x / L', 'y / L'),
        description=f'{len(ids)} particles on {metadata["config"]["arms"]} arms; '
                    f'r/R from {metadata["config"]["inner_radius_fraction"]} to '
                    f'{metadata["config"]["outer_radius_fraction"]}, '
                    f'R={metadata["config"]["radius_over_period"]} L. Integer-cycle returns from '
                    f'{metadata["config"]["cycles"]} cycles, {step} steps per cycle. Colors identify initial arms.',
        particle_groups=[dict(label=f'Arm {arm + 1}', particle_ids=np.asarray(ids)[arm_ids == arm].tolist())
                         for arm in range(metadata['config']['arms'])])
    return path


__all__ = ['plot_initial_star', 'comparison_table', 'plot_cpu_comparison', 'export_cpu_star_poincare']
