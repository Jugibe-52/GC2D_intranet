"""BM4Midpoint star trajectories and matched Modal JAX CPU timing records."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from diagnostics.persistence import load_solution, save_solution
from dynamics.gc import GuidingCenterDynamics
from execution.execution_modal import Execution_Modal
from initial_conditions.star import radial_star
from methods.extended.bm4 import BM4Midpoint
from potential.potential import Potential
from simulation.runner import simulate
from solution import Solution
from studies.h5_provenance import prepare_verified_h5_field
from studies.reference_trajectory import potential_fingerprint


@dataclass(frozen=True)
class ModalCPUStarConfig:
    """Explicit physical, geometric and numerical settings in normalized units.

    The reference radius is radius_over_period * L. Particle radii interpolate
    between inner_radius_fraction and outer_radius_fraction of that radius.
    Particles are ordered by arm, then increasing radius.
    """

    magnetic_field: float
    characteristic_length: float
    source_selection: tuple[int, ...]
    interpolation_order: int
    rho: float
    coupling_frequency: float
    arms: int
    particles_per_arm: int
    radius_over_period: float
    inner_radius_fraction: float
    outer_radius_fraction: float
    center_fraction: tuple[float, float]
    first_angle: float
    cycles: int
    steps_per_cycle: int
    cpu_counts: tuple[int, ...]
    repetitions: int
    equivalence_rtol: float
    equivalence_atol: float


@dataclass(frozen=True)
class ModalCPUComparison:
    """Complete saved solutions and shared measured comparison metadata."""

    solutions: dict[int, Solution]
    potential: Potential
    metadata: dict[str, Any]


def build_star_problem(potential: Potential, config: ModalCPUStarConfig) -> tuple[InitialValueProblem, SimulationRequest]:
    """Assemble the common physical job without selecting execution resources."""
    for name in ('arms', 'particles_per_arm', 'cycles', 'steps_per_cycle', 'repetitions'):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if config.repetitions < 3 or config.cpu_counts != (1, 2, 4):
        raise ValueError('Use CPU counts (1, 2, 4) and at least three measured repetitions.')
    if not (0 < config.radius_over_period <= .5
            and 0 < config.inner_radius_fraction < config.outer_radius_fraction <= 1):
        raise ValueError('Require a positive radius no greater than L/2 and 0 < inner < outer <= 1.')
    center = np.asarray(config.center_fraction, dtype=float)
    outer = config.radius_over_period * config.outer_radius_fraction
    if (center.shape != (2,) or not np.isfinite(center).all()
            or np.any(center - outer < 0) or np.any(center + outer > 1)):
        raise ValueError('The complete initial star must lie inside the periodic cell.')
    if not np.isfinite([config.equivalence_rtol, config.equivalence_atol]).all() or min(
            config.equivalence_rtol, config.equivalence_atol) < 0:
        raise ValueError('Equivalence tolerances must be finite and non-negative.')
    if potential.frequencies.shape != (1,) or not np.allclose(potential.frequencies, [1.], rtol=0, atol=1e-14):
        raise ValueError('The study requires one mode with normalized forcing period one.')
    grid = potential.grid
    origin = np.array([grid.x0, grid.y0]) + center * grid.period
    radius = config.radius_over_period * grid.period
    initial = radial_star(center=(float(origin[0]), float(origin[1])),
                          arm_length=radius * config.outer_radius_fraction,
                          inner_radius=radius * config.inner_radius_fraction,
                          arms=config.arms, particles_per_arm=config.particles_per_arm,
                          first_angle=config.first_angle)
    problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=config.rho), initial)
    steps = config.cycles * config.steps_per_cycle
    request = SimulationRequest((0., float(config.cycles)), 1. / config.steps_per_cycle,
                                np.arange(steps + 1, dtype=float) / config.steps_per_cycle)
    return problem, request


def prepare_cpu_star(source: str | Path, config: ModalCPUStarConfig, *, expected_source_sha256: str,
                     original_provenance: str | Path) -> tuple[Potential, dict[str, Any]]:
    """Prepare the verified original field for the CPU comparison."""
    return prepare_verified_h5_field(
        source, magnetic_field=config.magnetic_field, characteristic_length=config.characteristic_length,
        source_selection=config.source_selection, interpolation_order=config.interpolation_order,
        expected_source_sha256=expected_source_sha256, original_provenance=original_provenance,
    )

def run_cpu_star_comparison(
    potential: Potential, config: ModalCPUStarConfig, provenance: dict[str, Any], *,
    app_name: str, record_directory: str | Path,
    resume_records: dict[int, str] | None = None,
) -> ModalCPUComparison:
    """Collect each complete remote job locally and compare measured results."""
    problem, request = build_star_problem(potential, config)
    method = BM4Midpoint(coupling_frequency=config.coupling_frequency, track_energy=False,
                         progress=False, step_observer=None)
    options = ExecutionOptions(backend='jax', device='cpu')
    solutions, records = {}, {}
    for cores in config.cpu_counts:
        executor = (Execution_Modal.from_record(resume_records[cores])
                    if resume_records and cores in resume_records else
                    Execution_Modal(app_name, f'integrate_cpu_{cores}', record_directory=record_directory))
        print(f'Running Modal JAX CPU benchmark: {cores} core(s).', flush=True)
        start = perf_counter()
        solution = simulate(problem, method, request, execution=executor, options=options)
        total_seconds = perf_counter() - start
        diagnostics = solution.diagnostics
        if (diagnostics['step_count'] != config.cycles * config.steps_per_cycle
                or diagnostics['output_interpolation_count'] != 0
                or diagnostics['projection_kind'] != 'arithmetic_mean'
                or diagnostics['benchmark_cpu_cores'] != cores
                or diagnostics['benchmark_repetitions'] != config.repetitions):
            raise ValueError('Remote method, sampling or benchmark controls do not match the requested study.')
        times = np.asarray(diagnostics['benchmark_wall_seconds'])
        if times.shape != (config.repetitions,) or np.any(times <= 0) or not np.isfinite(times).all():
            raise ValueError('Invalid complete-integration timing samples.')
        q25, median, q75 = np.percentile(times, [25, 50, 75])
        solutions[cores] = solution
        records[str(cores)] = {
            'cpu_request': cores, 'cpu_limit': cores, 'runtime_seconds': times,
            'first_seconds': diagnostics['benchmark_first_seconds'],
            'median_seconds': float(median), 'q25_seconds': float(q25), 'q75_seconds': float(q75),
            'client_seconds': total_seconds, 'call_id': executor.last_call_id,
            'receipt': str(executor.last_record_path),
            'environment': json.loads(str(diagnostics['benchmark_environment'])),
        }
        print(f'CPU {cores}: first {records[str(cores)]["first_seconds"]:.3f} s; median {median:.6f} s.', flush=True)
    baseline = solutions[1]
    period = potential.grid.period
    for cores, solution in solutions.items():
        np.testing.assert_allclose(solution.states, baseline.states,
                                   rtol=config.equivalence_rtol, atol=config.equivalence_atol)
        delta = (solution.states - baseline.states + period / 2) % period - period / 2
        dx, dy = solution.source.layout.split(delta)
        distances = np.hypot(dx, dy)
        records[str(cores)].update(
            maximum_periodic_discrepancy=float(np.max(distances)),
            rms_periodic_discrepancy=float(np.sqrt(np.mean(distances**2))),
            speedup=records['1']['median_seconds'] / records[str(cores)]['median_seconds'],
        )
    x, y = problem.initial_configuration.layout.positions(problem.initial_state)
    assert isinstance(problem.dynamics, GuidingCenterDynamics)
    metadata = {
        'study': 'modal_jax_cpu_star', 'config': asdict(config), 'runs': records, **provenance,
        'initial_positions': np.column_stack((x, y)), 'particle_order': 'arm-major, then increasing radius',
        'particle_ids': np.arange(1, x.size + 1),
        'arm_ids': np.repeat(np.arange(config.arms), config.particles_per_arm),
        'radii_over_R': np.linspace(config.inner_radius_fraction, config.outer_radius_fraction, config.particles_per_arm),
        'reference_radius': config.radius_over_period * period,
        'effective_field_fingerprint': potential_fingerprint(problem.dynamics.effective_potential),
        'arithmetic': 'float64', 'reference_computed': False, 'accuracy_certified': False,
        'timing_scope': 'Worker complete integration after first-call warmup; NumPy output synchronizes JAX.',
        'client_timing_scope': 'Submission, provisioning, transfers, first and repeated runs, and local Solution validation.',
        'method': 'BM4Midpoint', 'projection': 'arithmetic_mean', 'newton_solves': 0,
    }
    return ModalCPUComparison(solutions, potential, metadata)


def save_cpu_star_comparison(comparison: ModalCPUComparison, destination: str) -> None:
    """Publish the baseline last so successful loads imply all runs were saved."""
    for cores in (2, 4, 1):
        save_solution(comparison.solutions[cores], f'{destination}/cpu_{cores}',
                      metadata=comparison.metadata, potential=comparison.potential)


def load_cpu_star_comparison(source: str) -> ModalCPUComparison:
    """Load saved data independently of Modal and the calculation notebook."""
    stored = load_solution(f'{source}/cpu_1')
    if stored.metadata.get('study') != 'modal_jax_cpu_star' or stored.potential is None:
        raise ValueError('The archive is not a Modal JAX CPU star comparison.')
    solutions = {1: stored.solution}
    for cores in (2, 4):
        other = load_solution(f'{source}/cpu_{cores}')
        if other.metadata != stored.metadata:
            raise ValueError('The CPU archives have inconsistent study metadata.')
        solutions[cores] = other.solution
    return ModalCPUComparison(solutions, stored.potential, stored.metadata)


__all__ = ['ModalCPUStarConfig', 'ModalCPUComparison', 'build_star_problem', 'prepare_cpu_star',
           'run_cpu_star_comparison', 'save_cpu_star_comparison', 'load_cpu_star_comparison']
