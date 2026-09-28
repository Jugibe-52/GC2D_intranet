"""Compose a GC study in original HDF5 space and potential units."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from methods.extended.bm4 import BM4Midpoint
from potential import Grid, Potential, load_gc2d_h5_potential
from simulation.runner import simulate
from solution import Solution


@dataclass(frozen=True, slots=True)
class DimensionalH5Field:
    """Original H5 potential and its normalized-time GC stream function.

    Coordinates and gyro-radius use the source axes' length unit. ``raw`` has
    the source potential unit, while ``stream`` has units length squared per
    normalized time. The GC vector field is J grad(stream).
    """

    raw: Potential
    stream: Potential
    cycle_seconds: float
    magnetic_field: float
    source_indices: tuple[int, ...]


def resolve_h5_source(source: str | Path) -> Path:
    """Use a checked-out H5 or its verified local Git LFS cache object."""
    path = Path(source).resolve()
    with path.open("rb") as stream:
        prefix = stream.read(200)
    if not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return path
    match = re.search(rb"oid sha256:([0-9a-f]{64})", prefix)
    if match is None:
        raise ValueError(f"Invalid Git LFS pointer: {path}")
    oid = match.group(1).decode("ascii")
    for parent in path.parents:
        cached = parent / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        if cached.is_file():
            return cached
    raise FileNotFoundError(
        f"{path} is a Git LFS pointer. Download the H5 object before running this study."
    )


def load_dimensional_h5_field(
    source: str | Path, *, magnetic_field: float = 1.5,
    characteristic_length: float = 0.06,
    selectors: tuple[int, ...] = (0, 1),
    interpolation_order: int = 3,
) -> DimensionalH5Field:
    """Restore original H5 samples after the canonical selection procedure.

    The public H5 loader selects and sorts source modes. Its normalization is
    inverted exactly at the sample level; no spatial resampling is requested.
    Time remains normalized by the selected mode's physical period.
    """
    normalized = load_gc2d_h5_potential(
        resolve_h5_source(source), B=magnetic_field, characteristic_length=characteristic_length,
        indx=selectors, interpolation_order=interpolation_order,
    )
    metadata = normalized.metadata
    if metadata is None or metadata.characteristic_period is None:
        raise ValueError("A positive-frequency H5 mode is required to normalize time.")
    x, y = metadata.source_x, metadata.source_y
    period_x = len(x) * (x[1] - x[0])
    period_y = len(y) * (y[1] - y[0])
    if not np.isclose(period_x, period_y):
        raise ValueError("The source H5 axes must have the same periodic span.")
    grid = Grid(float(x[0]), float(y[0]), float(x[1] - x[0]),
                float(y[1] - y[0]), len(x), len(y), float(period_x))
    scale = metadata.normalization_factor
    raw = Potential(grid, mean=normalized.mean * scale,
                    modes=normalized.modes * scale,
                    frequencies=normalized.frequencies,
                    interpolation_order=interpolation_order)
    # With tau=t/T0, d(R,Z)/d tau = (T0/B) J grad(Phi).
    flow_scale = metadata.characteristic_period / magnetic_field
    stream = Potential(grid, mean=raw.mean * flow_scale,
                       modes=raw.modes * flow_scale,
                       frequencies=raw.frequencies,
                       interpolation_order=interpolation_order)
    return DimensionalH5Field(raw, stream, metadata.characteristic_period,
                              magnetic_field, tuple(int(v) for v in metadata.source_field_indices))


def run_dimensional_bm4_midpoint(
    field: DimensionalH5Field, *, cycles: int = 5, steps_per_cycle: int = 40,
    initial_xy: tuple[float, float], rho: float = 0.0,
    coupling_frequency: float = float(np.pi / 8),
) -> Solution:
    """Integrate one particle on every complete normalized-time BM4 step."""
    if cycles < 1 or steps_per_cycle < 1:
        raise ValueError("Cycles and steps per cycle must be positive.")
    x0, y0 = (float(value) for value in initial_xy)
    initial = GCInitialConfiguration.from_components(
        x=np.asarray([x0]), y=np.asarray([y0]))
    problem = InitialValueProblem(GuidingCenterDynamics(field.stream, rho=rho), initial)
    count = cycles * steps_per_cycle
    request = SimulationRequest.uniform(t_span=(0.0, float(cycles)),
                                        max_step=1.0 / steps_per_cycle,
                                        sample_count=count + 1)
    solution = simulate(problem, BM4Midpoint(coupling_frequency=coupling_frequency), request)
    if solution.diagnostics["step_count"] != count or solution.states.shape != (2, count + 1):
        raise RuntimeError("BM4Midpoint did not save every complete step.")
    return solution


__all__ = ["DimensionalH5Field", "load_dimensional_h5_field", "resolve_h5_source", "run_dimensional_bm4_midpoint"]
