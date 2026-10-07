"""Presentation of saved trajectories without repeating numerical integration."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from diagnostics.persistence import StoredSolution


def plot_stored_gc_solution(record: StoredSolution) -> Figure:
    """Show GC trajectories and physical energy from a stored base potential."""
    if record.potential is None:
        raise ValueError("A saved potential is required for the field background.")
    if record.solution.layout.state_dimension != 2:
        raise ValueError("This plot requires a guiding-center solution.")
    if record.metadata.get("potential_role") != "base_before_gyroaverage":
        raise ValueError("Metadata must identify the saved field as the base potential.")
    rho = record.metadata["dynamics"]["rho"]
    field = record.potential.gyroaverage(rho)
    solution = record.solution
    x, y = solution.positions()  # (particles, saved_times), normalized coordinates.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    background = axes[0].pcolormesh(field.grid.x, field.grid.y,
                                   field.evaluate_grid(float(solution.t[0])).T,
                                   shading="auto", cmap="viridis")
    figure.colorbar(background, ax=axes[0], label="Effective potential at initial time")
    wrapped_x, wrapped_y = field.grid.normalize(x, y)
    energy = field.evaluate(solution.t, x, y)
    for particle in range(x.shape[0]):
        label = f"Particle {particle + 1}"
        # Break curves at periodic jumps instead of drawing across the cell.
        jumps = np.hypot(np.diff(wrapped_x[particle]), np.diff(wrapped_y[particle])) > field.grid.period / 2
        px, py = wrapped_x[particle].copy(), wrapped_y[particle].copy()
        px[1:][jumps], py[1:][jumps] = np.nan, np.nan
        line, = axes[0].plot(px, py, label=label)
        axes[0].scatter(wrapped_x[particle, 0], wrapped_y[particle, 0],
                        color=line.get_color(), marker="o", edgecolor="white")
        axes[1].plot(solution.t, energy[particle], label=label)
    axes[0].set(xlabel="x (normalized)", ylabel="y (normalized)",
                title="Saved trajectories; circles mark initial positions", aspect="equal")
    axes[1].set(xlabel="Time (normalized)", ylabel="Physical Hamiltonian",
                title="Energy history in a time-dependent potential")
    axes[1].legend()
    return figure


__all__ = ["plot_stored_gc_solution"]
