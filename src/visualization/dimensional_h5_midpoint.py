"""Display dimensional H5 fields with a single BM4 midpoint trajectory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from potential import Potential


def format_dimensional_h5_metadata(metadata: Mapping[str, object], potential: Potential) -> str:
    """Format saved field and trajectory settings as a notebook Markdown table."""
    grid = potential.grid
    rows = [
        ("Source H5", f"`{Path(str(metadata['source'])).name}`"),
        ("Selected H5 channels", f"`{tuple(metadata['selectors'])}`"),
        ("Oscillatory H5 field indices", f"`{tuple(metadata['source_field_indices'])}`"),
        ("Magnetic field B", f"{float(metadata['magnetic_field_T']):g} T"),
        ("Gyro-radius rho", f"{float(metadata['rho_m']):.8g} m (rho_hat = {float(metadata['rho_hat']):g})"),
        ("Characteristic length", f"{float(metadata['characteristic_length_m']):g} m"),
        ("H5 spatial origin (R, Z)", f"({grid.x0:.8g}, {grid.y0:.8g}) m"),
        ("H5 periodic cell span", f"{grid.period:.8g} m"),
        ("H5 grid", f"{grid.nx} × {grid.ny}; spacing ({grid.dx:.8g}, {grid.dy:.8g}) m"),
        ("Potential values", "Original H5 field units; no amplitude normalization"),
        ("Normalized mode frequencies", f"`{tuple(float(v) for v in potential.frequencies)}` cycles per normalized time"),
        ("Time normalization", f"tau = t / T0; T0 = {float(metadata['time_unit_seconds']):.10g} s"),
        ("Interpolation", f"Periodic spline, order {int(metadata['interpolation_order'])}"),
        ("Initial position (R, Z)", f"`{tuple(float(v) for v in metadata['initial_xy'])}` m"),
        ("Initial radial geometry", f"{float(metadata['initial_radial_fraction']):g} cell periods at {float(metadata['initial_angle_rad']):g} rad"),
        ("BM4Midpoint coupling frequency", f"{float(metadata['coupling_frequency']):.8g} per normalized time"),
        ("Integration", f"{int(metadata['cycles'])} cycles × {int(metadata['steps_per_cycle'])} steps/cycle"),
    ]
    return "| Parameter | Saved value |\n|---|---|\n" + "\n".join(
        f"| {name} | {value} |" for name, value in rows
    )


def plot_particle_in_potential(
    potential: Potential, times: np.ndarray, states: np.ndarray, *,
    snapshot_time: float = 0.0,
) -> tuple[Figure, Axes]:
    """Show the source-unit field at one time and the full wrapped path.

    The background is a snapshot of a time-dependent field. Path color shows
    trajectory time; it does not imply that the field stayed at the snapshot.
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if times.ndim != 1 or states.shape != (2, times.size):
        raise ValueError("Expected times (samples,) and one-particle states (2, samples).")
    grid = potential.grid
    field = potential.evaluate_grid(snapshot_time)
    wrapped_x, wrapped_y = grid.normalize(states[0], states[1])
    # Break the plotted line at periodic crossings instead of drawing across
    # the whole cell, which would suggest a spurious long particle displacement.
    jumps = (np.abs(np.diff(wrapped_x)) > grid.period / 2) | (
        np.abs(np.diff(wrapped_y)) > grid.period / 2)
    plot_x, plot_y = wrapped_x.copy(), wrapped_y.copy()
    plot_x[1:][jumps] = np.nan
    plot_y[1:][jumps] = np.nan
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = axis.pcolormesh(grid.x, grid.y, field.T, shading="auto", cmap="RdBu_r")
    figure.colorbar(mesh, ax=axis, label="Potential (H5 units)")
    axis.plot(plot_x, plot_y, color="black", lw=1.2, alpha=0.85)
    points = axis.scatter(wrapped_x, wrapped_y, c=times, s=9, cmap="viridis", zorder=3)
    figure.colorbar(points, ax=axis, label="Normalized time / cycles")
    axis.scatter(wrapped_x[0], wrapped_y[0], marker="o", s=85,
                 facecolor="white", edgecolor="black", label="Start", zorder=4)
    axis.scatter(wrapped_x[-1], wrapped_y[-1], marker="X", s=90,
                 facecolor="yellow", edgecolor="black", label="End", zorder=4)
    axis.set(xlabel="R (H5 length units)", ylabel="Z (H5 length units)",
             title=f"BM4Midpoint particle in H5 potential at cycle {snapshot_time:g}",
             xlim=(grid.x0, grid.x0 + grid.period),
             ylim=(grid.y0, grid.y0 + grid.period), aspect="equal")
    axis.legend(loc="upper right")
    return figure, axis


__all__ = ["format_dimensional_h5_metadata", "plot_particle_in_potential"]
