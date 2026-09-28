"""Small reproducible GC calculation for local and bucket persistence examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from diagnostics.persistence import save_solution
from dynamics.gc import GuidingCenterDynamics
from initial_conditions.gc import GCInitialConfiguration
from methods.classical.rk4 import RK4
from potential.potential import Potential
from simulation.runner import simulate
from solution import Solution


@dataclass(frozen=True)
class PersistenceDemoConfig:
    """Explicit scientific settings for a short storage demonstration.

    ``radial_fractions`` are spatial distances from the cell center divided by
    its period. ``rho`` is the normalized gyro-radius. Time and step are in
    normalized units; this study makes no comparative accuracy claims.
    """

    amplitude: float
    maximum_wave_number: int
    nx: int
    ny: int
    seed: int
    interpolation_order: int
    rho: float
    radial_fractions: tuple[float, ...]
    angle_radians: float
    t_span: tuple[float, float]
    max_step: float
    sample_count: int


def run_persistence_demo(
    config: PersistenceDemoConfig, destination: str | Path,
) -> tuple[Solution, str]:
    """Integrate all particles together and publish a self-contained result."""
    potential = Potential.random(
        A=config.amplitude, M=config.maximum_wave_number,
        nx=config.nx, ny=config.ny, seed=config.seed,
        interpolation_order=config.interpolation_order,
    )
    grid = potential.grid
    fractions = np.asarray(config.radial_fractions, dtype=float)
    if (fractions.ndim != 1 or not fractions.size or not np.all(np.isfinite(fractions))
            or np.any(fractions < 0) or np.any(fractions >= 0.5)
            or not np.isfinite(config.angle_radians)):
        raise ValueError("Use finite radial fractions in [0, 0.5) and a finite angle.")
    radii = fractions * grid.period
    initial = GCInitialConfiguration.from_components(
        x=grid.x0 + grid.period / 2 + radii * np.cos(config.angle_radians),
        y=grid.y0 + grid.period / 2 + radii * np.sin(config.angle_radians),
    )
    dynamics = GuidingCenterDynamics(potential, rho=config.rho)
    request = SimulationRequest.uniform(t_span=config.t_span, max_step=config.max_step,
                                        sample_count=config.sample_count)
    solution = simulate(InitialValueProblem(dynamics, initial), RK4(track_energy=True), request)
    # Hash the source used for this run, including uncommitted implementation
    # changes; a Git commit alone would not identify a local development run.
    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    metadata = {
        "study": "persistence_demo", "config": asdict(config),
        "method": {"name": "RK4", "track_energy": True},
        "dynamics": {"name": "GuidingCenterDynamics", "rho": config.rho},
        "potential_role": "base_before_gyroaverage",
        "potential_recipe": {"factory": "Potential.random", "A": config.amplitude,
                             "M": config.maximum_wave_number, "nx": config.nx,
                             "ny": config.ny, "seed": config.seed,
                             "interpolation_order": config.interpolation_order},
        "source_sha256": digest.hexdigest(),
        "purpose": "Storage smoke test; not a method accuracy comparison.",
    }
    location = save_solution(solution, destination, metadata=metadata, potential=potential)
    return solution, location


__all__ = ["PersistenceDemoConfig", "run_persistence_demo"]
