"""Periodicity-only views of a single BM4 guiding-centre trajectory."""

import matplotlib.pyplot as plt
import numpy as np


def plot_bm4_periodicity(result, *, period, particle_number, threshold_fraction, candidate_cycle):
    """Show integer returns and compare the trajectory with an integer-time shift."""
    states = result.positions[0]
    stride = result.config.saved_samples_per_cycle
    cycle_times = result.times[stride::stride]
    delta = (states - states[:, :1] + period / 2) % period - period / 2
    distances = np.linalg.norm(delta, axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].semilogy(cycle_times, np.maximum(distances[stride::stride] / period, 1e-16), 'o-', ms=4)
    axes[0].axhline(threshold_fraction, ls='--', color='grey', label='Return threshold')
    axes[0].set(xlabel='Integer forcing cycle', ylabel='Distance to initial state / L',
                title=f'Particle {particle_number}: same-phase returns')
    axes[0].legend()
    shift = candidate_cycle * stride
    if shift < states.shape[1] - 1:
        difference = (states[:, shift:] - states[:, :-shift] + period / 2) % period - period / 2
        error = np.linalg.norm(difference, axis=0)
        axes[1].plot(result.times[:-shift], error / period)
        axes[1].axhline(threshold_fraction, ls='--', color='grey')
        axes[1].set(xlabel='Time t [forcing cycles]', ylabel=f'Distance between z(t+{candidate_cycle}) and z(t) / L',
                    title=f'Candidate period {candidate_cycle}: overlap test')
    else:
        axes[1].text(.5, .5, 'No positive-duration overlap for this candidate.\nIncrease cycle_count to test repetition.',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_axis_off()
    for axis in axes:
        axis.grid(True, alpha=.25)
    return figure
