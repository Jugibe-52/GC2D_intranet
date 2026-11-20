"""Shoot integer-cycle, zero-winding periodic orbits with fixed-step JAX maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dynamics.gc import GuidingCenterDynamics
from diagnostics.studies.periodic_particles import orbit_metrics, distinct_orbit
from execution.options import ExecutionOptions
from potential.potential import Potential
from potential.load import GC2DH5Metadata
from problem import InitialValueProblem
from request import SimulationRequest
from simulation.runner import simulate
from studies.main_campaign import campaign_method


@dataclass
class PeriodicMap:
	"""Physical-cycle returns in unwrapped coordinates, shape (q+1, N, 2)."""

	dynamics: GuidingCenterDynamics
	cycle_duration: float
	cycles: int
	steps_per_cycle: int
	method: str

	def __call__(self, points: np.ndarray) -> np.ndarray:
		points = np.asarray(points, dtype=float)
		if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
			raise ValueError('Expected finite independent points with shape (N, 2).')
		times: np.ndarray = np.arange(self.cycles + 1) * self.cycle_duration
		initial: np.ndarray = points.T.ravel()
		problem: InitialValueProblem = InitialValueProblem(self.dynamics, initial,
			layout=GuidingCenterDynamics.coordinates,
			request=SimulationRequest((0., float(times[-1])), times))
		solution = simulate(problem, campaign_method(self.method, self.cycle_duration / self.steps_per_cycle),
			options=ExecutionOptions(backend='jax', device='cpu'))
		if (solution.diagnostics['step_count'] != self.cycles * self.steps_per_cycle
				or solution.diagnostics['output_interpolation_count'] != 0
				or solution.diagnostics['execution_backend'] != 'jax'
				or solution.states.dtype != np.float64):
			raise RuntimeError('Periodic shooting requires complete fixed steps without interpolation.')
		return solution.states.reshape(-1, 2, points.shape[0]).transpose(0, 2, 1)


def refine_points(mapping: PeriodicMap, seeds: np.ndarray, *, iterations: int = 18,
		tolerance: float = 2e-11) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
	"""Batched damped Newton shooting with centered finite-difference derivatives.

	Only the algebraic Newton correction varies; every return map has the same
	fixed integration grid. The two-coordinate Jacobians are independent per seed.
	"""
	points: np.ndarray = np.array(seeds, dtype=float, copy=True)
	size: int = len(points)
	epsilon: float = 2e-6
	offsets: np.ndarray = np.array(((0., 0.), (epsilon, 0.), (-epsilon, 0.),
		(0., epsilon), (0., -epsilon)))
	alphas: np.ndarray = np.array((1., .5, .25, .125, .0625, 0.))
	trace: list[dict[str, Any]] = []
	for iteration in range(iterations):
		stencil: np.ndarray = (points[None] + offsets[:, None]).reshape(-1, 2)
		ends: np.ndarray = mapping(stencil)[-1].reshape(5, size, 2)
		residual: np.ndarray = ends[0] - points
		norms: np.ndarray = np.linalg.norm(residual, axis=1)
		trace.append(dict(iteration=iteration, residuals=norms.tolist()))
		if np.all(norms < tolerance):
			break
		jacobian: np.ndarray = np.stack(((ends[1] - ends[2]) / (2 * epsilon),
			(ends[3] - ends[4]) / (2 * epsilon)), axis=-1) - np.eye(2)
		correction: np.ndarray = np.zeros_like(points)
		for i in range(size):
			if norms[i] >= tolerance:
				correction[i] = np.linalg.lstsq(jacobian[i], residual[i], rcond=1e-10)[0]
		# Bound each correction to keep Newton in the neighborhood of the seed.
		correction *= np.minimum(1., .025 / np.maximum(np.linalg.norm(correction, axis=1), 1e-30))[:, None]
		trials: np.ndarray = points[None] - alphas[:, None, None] * correction[None]
		trial_ends: np.ndarray = mapping(trials.reshape(-1, 2))[-1].reshape(6, size, 2)
		errors: np.ndarray = np.linalg.norm(trial_ends - trials, axis=2)
		choice: np.ndarray = errors.argmin(axis=0)
		updated: np.ndarray = trials[choice, np.arange(size)]
		if np.max(np.abs(updated - points)) < 1e-14:
			break
		points = updated
	returns: np.ndarray = mapping(points)
	return points, returns, trace


def search_period(field: Potential, cycles: int, candidates: list[dict[str, Any]],
		*, steps: int = 80) -> dict[str, Any]:
	"""Find distinct candidate roots; final acceptance requires separate refinement."""
	if not isinstance(field.metadata, GC2DH5Metadata) or field.spatial_extension != 'reflected':
		raise ValueError('Periodic seeds require the reflected measured field and its physical scales.')
	dynamics: GuidingCenterDynamics = GuidingCenterDynamics(field,
		rho=.3 * .06 / (2 * np.pi * field.metadata.spatial_scale), B=1.5)
	mapping: PeriodicMap = PeriodicMap(dynamics, 1 / float(field.frequencies[0]), cycles, steps, 'RK4')
	seeds: np.ndarray = np.array([c['xy'] for c in candidates])
	points: np.ndarray
	returns: np.ndarray
	trace: list[dict[str, Any]]
	points, returns, trace = refine_points(mapping, seeds)
	orbits: list[dict[str, Any]] = []
	accepted: list[np.ndarray] = []
	for i, point in enumerate(points):
		orbit: np.ndarray = returns[:, i]
		metrics: dict[str, float] = orbit_metrics(orbit)
		if (metrics['closure'] < 1e-9 and metrics['earlier_return_min'] > 1e-4
				and metrics['cycle_excursion'] > .002 and np.all((point > 0) & (point < 1))
				and all(distinct_orbit(orbit, old) for old in accepted)):
			accepted.append(orbit)
			orbits.append(dict(period_cycles=cycles, xy=point.tolist(), returns=orbit.tolist(),
				metrics=metrics, candidate=candidates[i]))
	return dict(period_cycles=cycles, method='RK4', steps_per_cycle=steps,
		orbits=orbits, trace=trace, final_residuals=np.linalg.norm(returns[-1]-points, axis=1).tolist())


def verify_orbit(field: Potential, candidate: dict[str, Any],
		prior: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Continue one root through finer Gauss grids and check independent RK4 closure."""
	if not isinstance(field.metadata, GC2DH5Metadata) or field.spatial_extension != 'reflected':
		raise ValueError('Require a reflected measured field.')
	dynamics: GuidingCenterDynamics = GuidingCenterDynamics(field,
		rho=.3 * .06 / (2 * np.pi * field.metadata.spatial_scale), B=1.5)
	cycle_duration: float = 1 / float(field.frequencies[0])
	cycles: int = int(candidate['period_cycles'])
	points: np.ndarray = np.asarray([candidate['xy']])
	refinement: list[dict[str, Any]] = []
	if prior is not None:
		if prior['candidate'] != candidate or prior['period_cycles'] != cycles:
			raise ValueError('Continuation must use the same candidate and period.')
		refinement = list(prior['refinement'])
		points = np.asarray([prior['xy']])
	previous_steps: int = int(refinement[-1]['steps_per_cycle']) if refinement else 0
	result: dict[str, Any] = dict(candidate=candidate, accepted=False)
	for steps in (80, 160, 320, 640, 1280):
		if steps <= previous_steps:
			continue
		previous: np.ndarray = points.copy()
		mapping: PeriodicMap = PeriodicMap(dynamics, cycle_duration, cycles, steps, 'GaussLegendre4')
		returns: np.ndarray
		trace: list[dict[str, Any]]
		points, returns, trace = refine_points(mapping, points, iterations=12)
		change: float = float(np.linalg.norm(points - previous))
		closure: float = float(np.linalg.norm(returns[-1] - points))
		refinement.append(dict(method='GaussLegendre4', steps_per_cycle=steps,
			xy=points[0].tolist(), initial_position_change=change, closure=closure,
			newton_trace=trace))
		result.update(refinement=refinement, xy=points[0].tolist(), period_cycles=cycles)
		if closure > 1e-9:
			result['reason'] = 'The algebraic shooting solve did not converge.'
			return result
		if steps < 320 or change > 1e-7:
			continue
		fine: np.ndarray = PeriodicMap(dynamics, cycle_duration, cycles, 2*steps, 'GaussLegendre4')(points)
		independent: np.ndarray = PeriodicMap(dynamics, cycle_duration, cycles, 2*steps, 'RK4')(points)
		fine_error: float = float(np.linalg.norm(fine[-1] - points))
		independent_error: float = float(np.linalg.norm(independent[-1] - points))
		result.update(check_steps_per_cycle=2*steps, fine_gauss_closure=fine_error,
			independent_rk4_closure=independent_error, returns=fine[:, 0].tolist())
		if max(fine_error, independent_error) > 1e-7:
			continue
		repeated: np.ndarray = PeriodicMap(dynamics, cycle_duration, cycles*5, 2*steps, 'GaussLegendre4')(points)
		repeated_errors: np.ndarray = np.linalg.norm(repeated[::cycles, 0] - points[0], axis=1)
		metrics: dict[str, float] = orbit_metrics(fine[:, 0])
		result.update(repetition_count=5, repetition_closures=repeated_errors[1:].tolist(), metrics=metrics)
		geometry_ok: bool = bool(metrics['earlier_return_min'] > 1e-4 and metrics['cycle_excursion'] > .002
			and np.all((points > 0) & (points < 1)))
		if not geometry_ok:
			result['reason'] = 'Minimum-period or geometry check failed.'
			return result
		if repeated_errors.max() < 1e-6:
			result['accepted'] = True
			result['reason'] = 'All fixed-step checks passed.'
			return result
	result['reason'] = 'The refinement sequence did not meet the requested discrepancy threshold.'
	return result


__all__ = ['PeriodicMap', 'refine_points', 'search_period', 'verify_orbit']
