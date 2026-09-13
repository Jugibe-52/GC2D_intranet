"""Geometry and independent-solver audit for radial reference studies."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from diagnostics import StoredReferenceTrajectory
from potential import Potential


def plot_radial_reference_audit(
    reference: StoredReferenceTrajectory,
    potential: Potential,
    center: np.ndarray,
) -> tuple[Figure, np.ndarray]:
    """Show initial radial coverage and the saved per-particle audit distances."""
    count = reference.initial_state.size // 2
    x, y = reference.initial_state.reshape(2, count)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot([center[0], x[-1]], [center[1], y[-1]], "k--", alpha=0.4)
    axes[0].scatter(*center, marker="+", color="black", label="Cell center")
    for i in range(count):
        axes[0].scatter(x[i], y[i], label=f"Particle {i + 1}")
        # A linear axis preserves exact zeros instead of replacing them by a floor.
        axes[1].plot(reference.times, reference.audit_distances[i], label=f"Particle {i + 1}")
    grid = potential.grid
    axes[0].set(xlim=(grid.xmin, grid.xmin + grid.period),
                ylim=(grid.ymin, grid.ymin + grid.period),
                xlabel="x", ylabel="y", title="Initial positions on one radius")
    axes[0].set_aspect("equal")
    axes[1].set(xlabel="Normalized time", ylabel="Periodic distance",
                title="DOP853–Radau discrepancy")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    return figure, axes
