"""Reproducible RK4 backend equivalence and end-to-end timing study."""

from dataclasses import asdict, dataclass
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np
import scipy

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from diagnostics.persistence import load_solution, save_solution
from dynamics.gc import GuidingCenterDynamics
from initial_conditions.gc import GCInitialConfiguration
from methods.classical.rk4 import RK4
from potential.potential import Potential
from simulation.runner import simulate
from solution import Solution


@dataclass(frozen=True)
class RK4ExecutionConfig:
    """Scientific settings in normalized units; all are explicit in notebooks.

    Initial particles uniformly sample the periodic cell with a separate seed.
    This short synthetic study compares implementations of the same RK4 map;
    the CPU trajectory is an equivalence baseline, not an accuracy reference.
    """

    amplitude: float
    maximum_wave_number: int
    nx: int
    ny: int
    potential_seed: int
    interpolation_order: int
    particle_count: int
    particle_seed: int
    rho: float
    t_span: tuple[float, float]
    max_step: float
    sample_count: int
    track_energy: bool
    repetitions: int
    equivalence_rtol: float
    equivalence_atol: float


@dataclass(frozen=True)
class RK4ExecutionComparison:
    """Complete trajectories and timing/validation records ready for storage."""

    solutions: dict[str, Solution]
    potential: Potential
    metadata: dict[str, Any]


def periodic_discrepancy(reference: Solution, candidate: Solution, period: float) -> np.ndarray:
    """Per-particle minimum-image distance, shaped (particles, saved_times)."""
    delta = candidate.states - reference.states
    delta = (delta + period / 2) % period - period / 2
    dx, dy = reference.source.layout.split(delta)
    return np.hypot(dx, dy)


def run_rk4_execution_comparison(
    config: RK4ExecutionConfig, *, executions: tuple[ExecutionOptions, ...],
) -> RK4ExecutionComparison:
    """Warm each configuration, then alternate synchronized complete runs.

    Potential construction and gyroaveraging are excluded. First calls include
    any device preparation and compilation; steady timings include transfers,
    integration, energy diagnostics and immutable NumPy Solution creation.
    Explicit device requests fail if unavailable, without substitution.
    """
    _validate_execution_comparison(config, executions)
    potential = Potential.random(
        A=config.amplitude, M=config.maximum_wave_number, nx=config.nx, ny=config.ny,
        seed=config.potential_seed, interpolation_order=config.interpolation_order,
    )
    grid = potential.grid
    rng = np.random.default_rng(config.particle_seed)
    initial = GCInitialConfiguration.from_components(
        x=grid.xmin + grid.period * rng.random(config.particle_count),
        y=grid.ymin + grid.period * rng.random(config.particle_count),
    )
    problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=config.rho), initial)
    request = SimulationRequest.uniform(t_span=config.t_span, max_step=config.max_step,
                                        sample_count=config.sample_count)
    method = RK4(track_energy=config.track_energy)
    selected = {f"{item.backend}_{item.device}_{item.device_index}": item for item in executions}
    solutions: dict[str, Solution] = {}
    first_calls: dict[str, float] = {}
    timings: dict[str, list[float]] = {key: [] for key in selected}
    for key, execution in selected.items():
        start = perf_counter()
        solutions[key] = simulate(problem, method, request, options=execution)
        first_calls[key] = perf_counter() - start
    orders = []
    for repetition in range(config.repetitions):
        order = list(selected) if repetition % 2 == 0 else list(reversed(selected))
        orders.append(order)
        for key in order:
            start = perf_counter()
            solutions[key] = simulate(problem, method, request, options=selected[key])
            timings[key].append(perf_counter() - start)
    baseline = solutions["scipy_cpu_0"]
    records: dict[str, dict[str, Any]] = {}
    for key, solution in solutions.items():
        np.testing.assert_array_equal(solution.t, baseline.t)
        np.testing.assert_allclose(solution.states, baseline.states,
                                   rtol=config.equivalence_rtol, atol=config.equivalence_atol)
        for name in ("step_count", "output_interpolation_count"):
            if solution.diagnostics[name] != baseline.diagnostics[name]:
                raise AssertionError(f"Execution changed {name}.")
        if config.track_energy:
            for name in ("extended_momentum", "physical_hamiltonian", "generalized_energy_error"):
                np.testing.assert_allclose(np.asarray(solution.diagnostics[name]), np.asarray(baseline.diagnostics[name]),
                                           rtol=config.equivalence_rtol, atol=config.equivalence_atol)
        discrepancy = periodic_discrepancy(baseline, solution, grid.period)
        q25, median, q75 = np.percentile(timings[key], [25, 50, 75])
        records[key] = {
            "execution": asdict(selected[key]), "first_call_seconds": first_calls[key],
            "runtime_seconds": timings[key], "median_seconds": float(median),
            "q25_seconds": float(q25), "q75_seconds": float(q75),
            "maximum_periodic_discrepancy": float(np.max(discrepancy)),
            "rms_periodic_discrepancy": float(np.sqrt(np.mean(discrepancy**2))),
            "maximum_component_discrepancy": float(np.max(np.abs(solution.states-baseline.states))),
            "device_kind": solution.diagnostics.get("execution_device_kind", platform.processor() or platform.machine()),
        }
    for record in records.values():
        record["speedup_vs_scipy"] = records["scipy_cpu_0"]["median_seconds"] / record["median_seconds"]
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    if any(item.backend == "jax" for item in executions):
        import jax
        import jaxlib
        versions.update(jax=jax.__version__, jaxlib=jaxlib.__version__)
    return RK4ExecutionComparison(solutions, potential, {
        "study": "rk4_execution", "config": asdict(config), "runs": records,
        "timing_orders": orders, "versions": versions, "platform": platform.platform(),
        "arithmetic": "float64", "initial_geometry": "uniform_periodic_cell",
        "potential_role": "base_before_gyroaverage", "baseline": "scipy_cpu_0",
        "purpose": "Backend equivalence and timing, not an integrator accuracy comparison.",
        "timing_scope": "Complete simulate calls, including transfers and Solution construction.",
    })


def save_rk4_execution_comparison(comparison: RK4ExecutionComparison, destination: str | Path) -> str:
    """Persist canonical NPZ/JSON Solution archives; report any upload failure."""
    base = str(destination).rstrip("/")
    # Publish the baseline last so readers cannot mistake a partial run for a
    # complete comparison. Each child retains the canonical integrity manifest.
    keys = [key for key in comparison.solutions if key != "scipy_cpu_0"] + ["scipy_cpu_0"]
    for key in keys:
        save_solution(comparison.solutions[key], f"{base}/{key}",
                      metadata=comparison.metadata, potential=comparison.potential)
    return base


def load_rk4_execution_comparison(source: str | Path) -> RK4ExecutionComparison:
    """Load a completed local/bucket comparison without recomputing dynamics."""
    base = str(source).rstrip("/")
    baseline = load_solution(f"{base}/scipy_cpu_0")
    if baseline.metadata.get("study") != "rk4_execution" or baseline.potential is None:
        raise ValueError("The source is not an RK4 execution comparison.")
    solutions = {"scipy_cpu_0": baseline.solution}
    for key in baseline.metadata["runs"]:
        if key not in solutions:
            solutions[key] = load_solution(f"{base}/{key}").solution
    return RK4ExecutionComparison(solutions, baseline.potential, baseline.metadata)


def _validate_execution_comparison(
	config: RK4ExecutionConfig, executions: tuple[ExecutionOptions, ...],
) -> None:
	"""Require valid timing controls and a distinct execution set containing the CPU baseline."""
	for name in ("particle_count", "repetitions"):
		value = getattr(config, name)
		if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
			raise ValueError(f"{name} must be a positive integer.")
	if config.repetitions < 3:
		raise ValueError("Use at least three repetitions to report timing quartiles.")
	if (not np.isfinite(config.equivalence_atol) or config.equivalence_atol < 0
			or not np.isfinite(config.equivalence_rtol) or config.equivalence_rtol < 0):
		raise ValueError("Equivalence tolerances must be finite and non-negative.")
	if not executions or any(not isinstance(item, ExecutionOptions) for item in executions):
		raise TypeError("Supply an explicit tuple of ExecutionOptions configurations.")
	if len(set(executions)) != len(executions) or ExecutionOptions() not in executions:
		raise ValueError("Use distinct executions including ExecutionOptions() as the CPU baseline.")


__all__ = ["RK4ExecutionConfig", "RK4ExecutionComparison", "run_rk4_execution_comparison",
           "save_rk4_execution_comparison", "load_rk4_execution_comparison", "periodic_discrepancy"]
