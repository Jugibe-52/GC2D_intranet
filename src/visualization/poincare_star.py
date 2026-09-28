"""Inspect saved dimensional star trajectories with the shared Poincare viewer."""

from pathlib import Path
from typing import Any

import numpy as np

from diagnostics.persistence import StoredSolution
from visualization.poincare_comparison import export_poincare_panel_comparison


_ARM_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#17becf")


def _star_record(saved: StoredSolution) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check saved study identity, particle labels and once-per-cycle sampling."""
    metadata, solution = saved.metadata, saved.solution
    if metadata.get("study") != "poincare_star_bm4_midpoint" or saved.potential is None:
        raise ValueError("Load a complete BM4Midpoint star archive, including its potential.")
    if metadata.get("space_unit") != "m" or metadata.get("spatial_normalization") != "none":
        raise ValueError("The star viewer requires saved physical coordinates in meters.")
    if solution.diagnostics.get("projection_kind") != "arithmetic_mean":
        raise ValueError("The saved run does not identify BM4 midpoint projection.")
    ids = np.asarray(metadata["particle_id"], dtype=int)
    arms = np.asarray(metadata["arm_id"], dtype=int)
    x, y = solution.positions()
    # Coordinates have (sample, particle, R/Z) order; sample zero is initialization.
    coordinates = np.stack((x.T, y.T), axis=-1)
    config = metadata["config"]
    expected = config["arms"] * config["particles_per_arm"]
    if (coordinates.shape != (config["cycles"] + 1, expected, 2)
            or ids.shape != (expected,) or arms.shape != ids.shape
            or len(set(ids)) != expected or np.any(arms < 0)
            or np.any(arms >= config["arms"])):
        raise ValueError("Saved particle labels or sample dimensions are inconsistent.")
    np.testing.assert_array_equal(solution.t, np.arange(config["cycles"] + 1, dtype=float))
    np.testing.assert_allclose(coordinates[0], metadata["initial_positions_m"], rtol=0, atol=1e-14)
    if not np.isfinite(coordinates).all():
        raise ValueError("Saved trajectory positions must be finite.")
    return coordinates, ids, arms


def star_summary(saved: StoredSolution) -> str:
    """Describe the saved record without creating or advancing a numerical method."""
    _, ids, _ = _star_record(saved)
    m, d = saved.metadata, saved.solution.diagnostics
    config = m["config"]
    return "\n".join((
        f"Method: BM4Midpoint; execution: {d.get('execution_backend', 'unknown')} / {d.get('execution_device', 'unknown')}",
        f"Particles: {ids.size} ({config['arms']} arms x {config['particles_per_arm']} particles)",
        f"Saved cycles: {config['cycles']}; integration steps/cycle: {config['steps_per_cycle']}",
        f"Arm length: {m['arm_length_m']:.6g} m; radial spacing: {m['radial_spacing_m']:.6g} m",
        f"rho_hat: {config['rho_hat']}; physical gyro-radius: {m['rho_m']:.9g} m",
        f"Time: tau = t/T0; T0 = {m['time_unit_seconds']:.9g} s",
        "Initial positions are shown separately from the once-per-cycle returns.",
    ))


def export_star_poincare(saved: StoredSolution, path: str | Path, *,
                         cycles_per_frame: int = 1,
                         wrap_periodic: bool = True) -> Path:
    """Export fixed initial positions and animated returns, both in meters.

    Periodic folding affects only display copies. Disabling it shows the full
    unwrapped trajectory extent. The original solution and archive are untouched.
    Display coordinates use float32, following the comparison viewer; numerical
    archives retain their original float64 precision.
    """
    coordinates, ids, arms = _star_record(saved)
    assert saved.potential is not None
    grid = saved.potential.grid
    origin = np.array([grid.x0, grid.y0])
    if wrap_periodic:
        coordinates = origin + np.remainder(coordinates - origin, grid.period)
        bounds = (grid.x0, grid.y0, grid.period)
        coordinate_note = "Display positions are folded into the periodic HDF5 cell; axes remain in meters."
    else:
        lower = np.minimum(origin, coordinates.min(axis=(0, 1)))
        upper = np.maximum(origin + grid.period, coordinates.max(axis=(0, 1)))
        span = float(np.max(upper - lower))
        lower = (lower + upper) / 2 - span / 2
        bounds = (float(lower[0]), float(lower[1]), span)
        coordinate_note = "Unwrapped saved positions are displayed in meters. Full view includes every saved position."
    colors = [_ARM_COLORS[int(arm) % len(_ARM_COLORS)] for arm in arms]
    common: dict[str, Any] = {"particle_ids": ids, "colors": colors}
    panels = [
        dict(common, title="Initial star (tau = 0)", coordinates=coordinates[:1], static=True),
        dict(common, title="BM4Midpoint: once-per-cycle returns", coordinates=coordinates[1:]),
    ]
    groups = [dict(label=f"Arm {int(arm) + 1}", particle_ids=ids[arms == arm]) for arm in np.unique(arms)]
    return Path(export_poincare_panel_comparison(
        path, panels, cycles_per_frame=cycles_per_frame,
        title=f"Star Poincare section — {ids.size} particles",
        coordinate_bounds=bounds, axis_labels=("R (m)", "Z (m)"), particle_groups=groups,
        description=(coordinate_note + " Colors identify arms. The left panel remains fixed; "
                     "the right panel shows accumulated returns and the current cycle. "
                     "Select particles or isolate one arm. Drag to zoom both panels; "
                     "Shift-drag or right-drag to pan. Full view restores the complete domain."),
    ))


__all__ = ["export_star_poincare", "star_summary"]
