"""Scientific figures for saved, same-phase periodicity studies."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from studies.poincare_periodicity import section_distances

if TYPE_CHECKING:
    from studies.poincare_periodicity import PoincarePeriodicity, SavedPoincareSection, ShortPeriodicity


def _panels(section: SavedPoincareSection) -> tuple[Figure, np.ndarray]:
    count = len(section.particle_ids)
    figure, axes = plt.subplots((count + 1) // 2, 2, figsize=(12, 4 * ((count + 1) // 2)),
                                squeeze=False, layout='constrained')
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    for axis, pid in zip(axes.flat, section.particle_ids):
        axis.set_title(f'Particle {pid}')
        axis.grid(alpha=.2)
    return figure, axes.ravel()[:count]


def plot_section_initial_positions(section: SavedPoincareSection) -> Figure:
    """Display the original geometry and IDs in the full periodic cell."""
    figure, axis = plt.subplots(figsize=(7, 6), layout='constrained')
    for j, pid in enumerate(section.particle_ids):
        point = section.positions[0, j] % 1
        axis.scatter(point[0], point[1], color=section.colors[j], s=45)
        axis.annotate(str(pid), point, xytext=(0, 10 + 12 * (j % 2)),
                      textcoords='offset points', ha='center')
    axis.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='x / L', ylabel='y / L',
             title='Selected original initial positions')
    axis.grid(alpha=.2)
    return figure


def plot_periodicity_diagnostic(
    section: SavedPoincareSection, result: PoincarePeriodicity, *, diagnostic: str,
) -> Figure:
    """Show sections, initial returns, lag defects, best-lag histories or spectra."""
    if diagnostic not in ('sections', 'initial_returns', 'lag_scan', 'best_history', 'spectra'):
        raise ValueError('Unknown periodicity diagnostic.')
    figure, axes = _panels(section)
    for j, (axis, row) in enumerate(zip(axes, result.summary)):
        if diagnostic == 'sections':
            points = section.positions[:, j] % 1
            scatter = axis.scatter(*points[1:].T, c=np.arange(1, len(points)), s=3,
                                   cmap='viridis', rasterized=True)
            axis.scatter(*points[0], marker='x', color='red', s=55, label='Initial state')
            axis.set(xlabel='x / L', ylabel='y / L', aspect='equal')
            axis.ticklabel_format(useOffset=False)
        elif diagnostic == 'initial_returns':
            axis.semilogy(np.arange(1, len(section.positions)),
                          np.maximum(result.initial_distances[1:, j], 1e-16), lw=.5)
            axis.set(xlabel='Forcing cycle n', ylabel='Distance to initial state / L')
        elif diagnostic == 'lag_scan':
            axis.semilogy(result.lags, np.maximum(result.metrics[:, j, 0], 1e-16), label='RMS')
            axis.semilogy(result.lags, np.maximum(result.metrics[:, j, 2], 1e-16),
                          label='Maximum', alpha=.6)
            for eps in result.threshold_fractions:
                axis.axhline(eps, ls='--', lw=.6, color='grey')
            axis.axvline(row['best_q'], ls=':', color='crimson', label=f"Best q={row['best_q']}")
            axis.set(xlabel='Lag q [forcing cycles]', ylabel='Repetition defect / L')
            axis.legend(fontsize=8)
        elif diagnostic == 'best_history':
            axis.plot(np.arange(len(result.best_defects[j])), result.best_defects[j], lw=.5,
                      color=section.colors[j])
            axis.set(title=f"Particle {row['particle']}: q={row['best_q']}",
                     xlabel='Starting cycle n', ylabel='d(z(n+q), z(n)) / L')
        else:
            for label, (frequency, power) in result.spectra.items():
                axis.semilogy(frequency[1:], np.maximum(power[1:, j], 1e-30), lw=.7,
                              alpha=.8, label=label.replace('_', ' ').title())
            axis.set(xlabel='Frequency [1 / forcing cycle]', ylabel='Summed spectral power')
            axis.legend(fontsize=8)
    if diagnostic == 'sections':
        figure.colorbar(scatter, ax=list(axes), label='Forcing cycle')
    return figure


def plot_residue_section(section: SavedPoincareSection, *, particle_id: int,
                         candidate_q: int, residue: int = 0) -> Figure:
    """Resolve q subsequences and the closure defect of a short resonance candidate."""
    if (isinstance(candidate_q, bool) or int(candidate_q) != candidate_q
            or not 1 <= candidate_q < len(section.positions)
            or isinstance(residue, bool) or int(residue) != residue or not 0 <= residue < candidate_q):
        raise ValueError('Require an integer lag within the record and 0 <= residue < lag.')
    j = section.particle_ids.index(particle_id)
    points = section.positions[:, j] % 1
    cycles = np.arange(len(points))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    for r in range(candidate_q):
        selected = cycles % candidate_q == r
        axes[0].scatter(*points[selected].T, s=5, label=str(r))
    axes[0].set(title=f'Particle {particle_id}: n modulo {candidate_q}')
    axes[0].legend(title='Residue', ncol=4, fontsize=8,
                   loc='upper center', bbox_to_anchor=(.5, -.16))
    selected = cycles % candidate_q == residue
    scatter = axes[1].scatter(*points[selected].T, c=cycles[selected], s=9, cmap='viridis')
    axes[1].set(title=f'Zoom: n modulo {candidate_q} = {residue}')
    figure.colorbar(scatter, ax=axes[1], label='Forcing cycle')
    for axis in axes[:2]:
        axis.set(xlabel='x / L', ylabel='y / L', aspect='equal')
        axis.ticklabel_format(useOffset=False)
    defect = section_distances(section.positions[candidate_q:, j:j + 1],
                               section.positions[:-candidate_q, j:j + 1])[:, 0]
    axes[2].plot(cycles[:-candidate_q], defect, lw=.6)
    axes[2].set(xlabel='Starting cycle n', ylabel='Repetition defect / L',
                title=f'q={candidate_q}: RMS={np.sqrt(np.mean(defect**2)):.3g} L\n'
                      f'Maximum={defect.max():.3g} L')
    return figure


def show_periodicity_figure(figure: Figure, directory: Path | None, name: str) -> None:
    """Optionally export a figure, then display and release it in a notebook."""
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(figure)


def export_interval_recurrence_viewer(
    path: str | Path, section: SavedPoincareSection, result: PoincarePeriodicity,
    *, title: str,
) -> Path:
    """Export linked spatial and initial-return views for a cycle interval.

    Particle panels are stacked vertically. A shared inclusive start/end cycle
    filters both the spatial returns and the distance-to-initial-state graph.
    The complete once-per-cycle record is embedded in a self-contained page.
    """
    positions = np.asarray(section.positions, dtype='<f4') % 1
    distances = np.asarray(result.initial_distances, dtype='<f4')
    expected = (len(positions), len(section.particle_ids))
    if positions.ndim != 3 or positions.shape[1:] != (expected[1], 2):
        raise ValueError('Section positions must have shape (cycles, particles, 2).')
    if distances.shape != expected:
        raise ValueError('Initial distances must match the section samples and particles.')
    if not np.isfinite(positions).all() or not np.isfinite(distances).all():
        raise ValueError('Viewer coordinates and distances must be finite.')

    panels = []
    for j, particle_id in enumerate(section.particle_ids):
        coordinates = np.ascontiguousarray(positions[:, j], dtype='<f4')
        values = np.ascontiguousarray(distances[:, j], dtype='<f4')
        panels.append(dict(
            particle=int(particle_id), color=str(section.colors[j]), samples=len(coordinates),
            coordinates=base64.b64encode(coordinates.tobytes()).decode('ascii'),
            distances=base64.b64encode(values.tobytes()).decode('ascii'),
        ))
    payload = json.dumps(
        dict(title=str(title), maximum_cycle=len(positions) - 1, panels=panels),
        separators=(',', ':'),
    ).replace('<', '\\u003c')
    template = Path(__file__).with_name('_poincare_interval_viewer.html').read_text(encoding='utf-8')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace('__POINCARE_INTERVAL_DATA__', payload), encoding='utf-8')
    return destination


def plot_short_periodicity(section: SavedPoincareSection, result: ShortPeriodicity,
                           *, diagnostic: str, preview_cycles: int = 80) -> Figure:
    """Display recurrence cadence, repeated shifts, temporal stability or phase classes."""
    if diagnostic not in ('signals', 'correlations', 'multiples', 'blocks', 'phase_classes'):
        raise ValueError('Unknown short-periodicity diagnostic.')
    figure, axes = _panels(section)
    for j, (axis, row) in enumerate(zip(axes, result.summary)):
        q, pid = row['short_q'], row['particle']
        if diagnostic == 'signals':
            stop = min(len(section.positions), preview_cycles+1)
            for coordinate, label in enumerate(('x / L', 'y / L')):
                axis.plot(np.arange(stop), section.positions[:stop, j, coordinate],
                          '.-', ms=3, lw=.8, label=label)
            axis.set(xlabel='Forcing cycle', ylabel='Position / L')
            axis.legend(fontsize=8)
        elif diagnostic == 'correlations':
            axis.plot(result.lags, result.correlations[:, j], '.-', ms=3, lw=.9)
            peaks = [r for r in result.peaks if r['particle'] == pid]
            axis.scatter([r['lag'] for r in peaks], [r['correlation'] for r in peaks],
                         s=28, color='crimson', label='Recurrent peaks')
            if q:
                axis.axvline(q, color='grey', ls='--', label=f'First peak: {q} cycles')
            axis.set(xlabel='Lag [forcing cycles]', ylabel='Normalized correlation', ylim=(-1.05, 1.05))
            axis.legend(fontsize=8)
        elif diagnostic == 'multiples':
            records = [r for r in result.multiples if r['particle'] == pid]
            for metric, label in (('rms_over_L', 'RMS'), ('max_over_L', 'Maximum')):
                axis.plot([r['multiple'] for r in records], [r[metric] for r in records],
                          'o-', ms=3, lw=.9, label=label)
            axis.set(xlabel=f'Number of repetitions k (lag = k × {q} cycles)', ylabel='Repetition defect / L')
            if records:
                axis.legend(fontsize=8)
        elif diagnostic == 'blocks':
            turns = [r for r in result.revolutions if r['particle'] == pid]
            blocks = [r for r in result.blocks if r['particle'] == pid and r['mean_rotation_cycles'] is not None]
            axis.plot([(r['start_cycle'] + r['end_cycle']) / 2 for r in turns],
                      [r['duration_cycles'] for r in turns], '.', ms=2, alpha=.4, label='Individual revolution')
            for k, block in enumerate(blocks):
                axis.plot([block['start_cycle'], block['stop_cycle_exclusive']],
                          [block['mean_rotation_cycles']]*2, color='crimson', lw=2,
                          label='Mean in time block' if k == 0 else None)
            if q:
                axis.axhline(q, color='grey', ls=':', label=f'{q}-cycle rhythm')
            axis.set(xlabel='Forcing cycle', ylabel='Geometric revolution [cycles]')
            if turns:
                axis.legend(fontsize=8)
        else:
            if q is None:
                axis.text(.5, .5, 'No short candidate', transform=axis.transAxes, ha='center')
                continue
            points = section.positions[:, j] % 1
            scatter = axis.scatter(*points.T, c=np.arange(len(points)) % q,
                                   cmap=plt.get_cmap('tab10', q), vmin=-.5, vmax=q-.5, s=4,
                                   rasterized=True)
            figure.colorbar(scatter, ax=axis, ticks=np.arange(q), label=f'Cycle n modulo {q}')
            axis.set(xlabel='x / L', ylabel='y / L', aspect='equal',
                     title=f"Particle {pid}: {q}-cycle phase classes")
            axis.ticklabel_format(useOffset=False)
    return figure


def export_phase_class_selector(path: str | Path, section: SavedPoincareSection,
                                result: ShortPeriodicity, *, title: str) -> Path:
    """Export four offline phase-class panels with selectable residues.

    Every saved once-per-cycle position is embedded as little-endian float32.
    Each panel keeps the short integer candidate selected by the analysis, while
    its controls independently show or hide any classes ``n modulo q``.
    """
    positions = np.asarray(section.positions, dtype='<f4') % 1
    if positions.ndim != 3 or positions.shape[1:] != (len(section.particle_ids), 2):
        raise ValueError('Section positions must have shape (cycles, particles, 2).')
    if not np.isfinite(positions).all():
        raise ValueError('Section positions must be finite.')
    rows = {int(row['particle']): row for row in result.summary}
    panels = []
    for j, particle_id in enumerate(section.particle_ids):
        q = rows[int(particle_id)]['short_q']
        if q is None or isinstance(q, bool) or int(q) != q or int(q) < 1:
            raise ValueError(f'Particle {particle_id} has no integer short-period candidate.')
        coordinates = np.ascontiguousarray(positions[:, j], dtype='<f4')
        panels.append(dict(particle=int(particle_id), q=int(q), samples=len(coordinates),
                           coordinates=base64.b64encode(coordinates.tobytes()).decode('ascii')))
    payload = json.dumps(dict(title=str(title), panels=panels), separators=(',', ':')).replace('<', '\\u003c')
    template = Path(__file__).with_name('_phase_class_selector.html').read_text(encoding='utf-8')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace('__PHASE_CLASS_DATA__', payload), encoding='utf-8')
    return destination
