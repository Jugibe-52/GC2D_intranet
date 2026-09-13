"""Focused trajectory, return and accuracy figures for one GC particle."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from studies.single_particle_recurrence import distance_to_initial, minimum_image


def display_recurrence_records(records: Any, *, columns: Any) -> None:
    """Display an event table, including an explicit empty-search outcome."""
    from .notebooks import display_records_table

    if records:
        display_records_table(records, columns=columns)
    else:
        print('No events satisfy this selection in the configured search interval.')


def _positive(values: np.ndarray) -> np.ndarray:
    """Omit exact zeros from logarithmic plots without altering stored data."""
    return np.where(values > 0, values, np.nan)


def plot_particle_recurrence(arrays: dict[str, np.ndarray], metadata: dict[str, Any],
                             original_times: np.ndarray, original_states: np.ndarray) -> Figure:
    """Show only the selected particle, its returns, and integer-phase samples."""
    period = metadata['period']
    initial = arrays['initial_state']
    times = arrays['times']
    delta = minimum_image(arrays['DOP853_refined.states'] - initial[:, None], period) / period
    path = delta.copy()
    path[:, np.r_[False, np.any(np.abs(np.diff(path, axis=1)) > .5, axis=0)]] = np.nan
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(*path, lw=.8, color='steelblue', alpha=.8, label='Refined reference')
    cycle_delta = minimum_image(arrays['DOP853_refined.cycle_states'] - initial[:, None], period) / period
    points = axes[0].scatter(*cycle_delta, c=arrays['cycle_times'], cmap='viridis', s=20, zorder=3)
    axes[0].scatter(0, 0, color='black', marker='+', s=100, label='Initial state', zorder=4)
    axes[0].add_patch(plt.Circle((0, 0), .01, fill=False, color='grey', ls='--'))
    axes[0].set(xlabel=r'$\Delta x/L$', ylabel=r'$\Delta y/L$',
                title='Particle 2: displacement from the initial point', aspect='equal')
    axes[0].legend(fontsize=8)
    figure.colorbar(points, ax=axes[0], label='Integer forcing cycle')
    valid = times >= metadata['config']['search_start']
    distances = distance_to_initial(arrays['DOP853_refined.states'], initial, period) / period
    axes[1].semilogy(times[valid], _positive(distances[valid]), lw=1.1, label='Continuous reference')
    cycle_distances = distance_to_initial(arrays['DOP853_refined.cycle_states'], initial, period) / period
    axes[1].semilogy(arrays['cycle_times'], _positive(cycle_distances), 'o', ms=3, label='Same forcing phase')
    source_valid = original_times >= metadata['config']['search_start']
    source_distances = distance_to_initial(original_states, initial, period) / period
    axes[1].semilogy(original_times[source_valid], _positive(source_distances[source_valid]),
                     '.', color='grey', ms=2, alpha=.5, label='Original BM4 samples')
    rows = metadata['returns']
    if rows:
        axes[1].semilogy([r['time'] for r in rows], [r['cell_fraction'] for r in rows],
                         'v', color='crimson', ms=4, label='Resolved spatial minima')
    axes[1].axhline(.01, ls='--', color='grey', label='Original 1% threshold')
    axes[1].set(xlabel='Time [forcing cycles]', ylabel=r'$d(t)/L$', title='Spatial and same-phase returns')
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2)
    return figure


def plot_particle_return_zoom(arrays: dict[str, np.ndarray], metadata: dict[str, Any],
                              original_times: np.ndarray, original_states: np.ndarray,
                              *, target_cycle: int, half_width: float = .5) -> Figure:
    """Resolve the selected cycle in time and in its local spatial coordinates."""
    period, initial = metadata['period'], arrays['initial_state']
    times = arrays['times']
    mask = np.abs(times - target_cycle) <= half_width
    source_mask = np.abs(original_times - target_cycle) <= half_width
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    colors = {'DOP853': 'darkorange', 'DOP853_refined': 'steelblue', 'Radau': 'seagreen'}
    for label, color in colors.items():
        state = arrays[label + '.states'][:, mask]
        distance = distance_to_initial(state, initial, period)
        axes[0].semilogy(times[mask], _positive(distance), color=color, label=label, lw=1.2)
    axes[0].semilogy(original_times[source_mask], _positive(distance_to_initial(
        original_states[:, source_mask], initial, period)), 'o', ms=4, color='grey', label='Original BM4')
    delta = minimum_image(arrays['DOP853_refined.states'][:, mask] - initial[:, None], period)
    axes[1].plot(*delta, color='steelblue', lw=1.2)
    axes[1].scatter(0, 0, color='black', marker='+', s=100, label='Initial state')
    cycle_index = int(np.flatnonzero(arrays['cycle_times'] == target_cycle)[0])
    cycle_delta = minimum_image(arrays['DOP853_refined.cycle_states'][:, cycle_index] - initial, period)
    axes[1].scatter(*cycle_delta, color='darkorange', s=45, label=f't = {target_cycle}')
    rows = [r for r in metadata['returns'] if abs(r['time'] - target_cycle) <= half_width]
    for index, row in enumerate(rows):
        label = 'Spatial minimum' if index == 0 else None
        axes[0].scatter(row['time'], row['distance'], color='crimson', marker='v', zorder=5, label=label)
        axes[1].scatter(row['dx'], row['dy'], color='crimson', marker='v', zorder=5, label=label)
        axes[1].plot([0, row['dx']], [0, row['dy']], color='crimson', ls=':', lw=1)
    axes[0].axvline(target_cycle, color='black', ls=':', lw=1)
    axes[0].set(xlabel='Time [forcing cycles]', ylabel='Periodic distance', title='Return-time detail')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel=r'$\Delta x$', ylabel=r'$\Delta y$', aspect='equal', title='Closest approach geometry')
    axes[1].legend(fontsize=8)
    for label in ('DOP853', 'Radau'):
        axes[2].semilogy(times[mask], _positive(arrays[label + '.reference_discrepancy'][mask]),
                         label=f'{label} vs refined', color=colors[label])
    axes[2].set(xlabel='Time [forcing cycles]', ylabel='Trajectory discrepancy', title='Local reference resolution')
    axes[2].legend(fontsize=8)
    return figure


def plot_particle_reference_audit(arrays: dict[str, np.ndarray], metadata: dict[str, Any],
                                  original_times: np.ndarray, original_states: np.ndarray) -> Figure:
    """Separate BM4 trajectory error, reference agreement, and velocity return."""
    times, period = arrays['times'], metadata['period']
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for label, color in (('DOP853', 'darkorange'), ('Radau', 'seagreen')):
        axes[0].semilogy(times[1:], _positive(arrays[label + '.reference_discrepancy'][1:]),
                         lw=1, color=color, label=f'{label} vs refined DOP853')
    indices = np.rint(original_times / (times[1] - times[0])).astype(int)
    if np.all(indices < len(times)) and np.allclose(times[indices], original_times, rtol=0, atol=1e-12):
        error = np.linalg.norm(minimum_image(
            original_states - arrays['DOP853_refined.states'][:, indices], period), axis=0)
        axes[0].semilogy(original_times[1:], _positive(error[1:]), color='grey', label='Original BM4 vs reference')
    axes[0].set(xlabel='Time [forcing cycles]', ylabel='Periodic trajectory discrepancy',
                title='Integration resolution over the complete horizon')
    axes[0].legend(fontsize=8)
    axes[1].plot(times, arrays['velocity_difference'], color='steelblue', lw=.9)
    axes[1].set(xlabel='Time [forcing cycles]', ylabel=r'$\|f(t,z(t))-f(0,z_0)\|$',
                title='Drift-velocity difference from its initial value')
    return figure
