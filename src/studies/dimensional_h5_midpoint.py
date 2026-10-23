"""Compose a GC study in original HDF5 space and potential units."""

from __future__ import annotations

from studies._validation import require_uncoupled_bm4

from formulations.layout import GCStateLayout

from dataclasses import dataclass, replace
from pathlib import Path
import re

import h5py
import numpy as np

from problem import InitialValueProblem
from request import SimulationRequest
from dynamics import GuidingCenterDynamics

from methods.extended.bm4 import BM4Midpoint
from potential.load import DEFAULT_FIELD_INDICES
from potential import GC2DH5Metadata, Grid, Potential
from simulation.runner import simulate
from states.history import States

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


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
    path: Path = Path(source).resolve()
    with path.open("rb") as stream:
        prefix: bytes = stream.read(200)
    if not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return path
    match: re.Match[bytes] | None = re.search(rb"oid sha256:([0-9a-f]{64})", prefix)
    if match is None:
        raise ValueError(f"Invalid Git LFS pointer: {path}")
    oid: str | Any = match.group(1).decode("ascii")
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
    selectors: tuple[int, ...] = DEFAULT_FIELD_INDICES,
    interpolation_order: int = 3,
) -> DimensionalH5Field:
    """Read original samples and compose the explicit cycle-time GC dynamics.

    The public H5 loader always includes field zero and selects variable fields
    by their original source indices, preserving the requested order. No normalization is applied and undone during import.
    Time retains the archived clock 2*pi/f_ref, not the corrected physical period.
    """
    resolved: Path = resolve_h5_source(source)
    if not selectors:
        raise ValueError("A positive-frequency H5 mode is required to normalize time.")
    raw_seconds: Potential = Potential.load(resolved, indx=selectors,
                                 interpolation_order=interpolation_order)
    metadata = raw_seconds.metadata
    if not isinstance(metadata, GC2DH5Metadata):
        raise ValueError("The field requires HDF5 provenance.")
    period: float = float(2 * np.pi / metadata.source_frequencies[0])
    # Keep the established raw contract: physical amplitudes/space, cycle time.
    cyclic_rates = metadata.source_frequencies / metadata.source_frequencies[0]
    legacy_metadata: GC2DH5Metadata = replace(metadata, characteristic_period=period, time_scale=period,
                              temporal_rescaling=True, time_convention="legacy_cycles")
    raw: Potential = Potential(raw_seconds.grid, mean=raw_seconds.mean, modes=raw_seconds.modes,
                    frequencies=cyclic_rates, metadata=legacy_metadata, interpolation_order=interpolation_order)
    cycle_potential: Potential = Potential(raw.grid, mean=raw.mean * period,
                               modes=raw.modes * period, frequencies=raw.frequencies,
                               metadata=replace(legacy_metadata, normalization_factor=1.0/period),
                               interpolation_order=interpolation_order)
    stream = GuidingCenterDynamics(cycle_potential, B=magnetic_field).effective_potential
    return DimensionalH5Field(raw, stream, period,
                              magnetic_field, tuple(int(v) for v in metadata.source_field_indices))


def run_dimensional_bm4_midpoint(
    field: DimensionalH5Field, *, cycles: int = 5, steps_per_cycle: int = 40,
    initial_xy: tuple[float, float], rho: float = 0.0,
    coupling_frequency: float = 0.0,
) -> States:
    """Integrate one particle on every complete normalized-time BM4 step."""
    require_uncoupled_bm4(coupling_frequency)
    if cycles < 1 or steps_per_cycle < 1:
        raise ValueError("Cycles and steps per cycle must be positive.")
    x0: float
    y0: float
    x0, y0 = (float(value) for value in initial_xy)
    initial: np.ndarray = GCStateLayout().pack_components(np.asarray([x0]), np.asarray([y0]))
    problem: InitialValueProblem = InitialValueProblem(GuidingCenterDynamics(field.stream, rho=rho), initial, layout=GCStateLayout())
    count = cycles * steps_per_cycle
    request: SimulationRequest = SimulationRequest.uniform(t_span=(0.0, float(cycles)),
                                        sample_count=count + 1)
    solution: States = simulate(problem, BM4Midpoint(step=1.0 / steps_per_cycle), request)
    if solution.diagnostics["step_count"] != count or solution.states.shape != (2, count + 1):
        raise RuntimeError("BM4Midpoint did not save every complete step.")
    return solution


__all__ = ["DimensionalH5Field", "load_dimensional_h5_field", "resolve_h5_source", "run_dimensional_bm4_midpoint"]
