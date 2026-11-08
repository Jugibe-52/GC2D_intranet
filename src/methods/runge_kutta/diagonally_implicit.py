"""Sequential diagonal Runge--Kutta stages, including explicit zero diagonals."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any
import numpy as np

from contracts.arrays import BackendArray, array_namespace, checked_finite
from contracts.methods.step import StepValue
from methods.runge_kutta.base import _NewtonRungeKutta, _RungeKuttaStep
from methods.runge_kutta._jacobians import _checked_vector_field
from solvers.linear import solve_linear, _solve_particle_systems
from solvers.nonlinear import _NonlinearResult


@dataclass
class DiagonallyImplicitRungeKutta(_NewtonRungeKutta):
	"""Solve lower-triangular tableaux one physical stage at a time."""

	def _validate_tableau(self) -> None:
		if np.any(np.triu(self.tableau.A, k=1) != 0):
			raise ValueError('Diagonally implicit Runge--Kutta requires lower-triangular A.')

	@property
	def _nonlinear_solve_count(self) -> int:
		return int(np.count_nonzero(np.diag(self.tableau.A)))

	def _solve_stage(self, time: float, state: BackendArray, step: float, index: int,
		right_side: BackendArray, initial_field: BackendArray, *, observe: bool) -> _NonlinearResult[BackendArray]:
		"""Bind the current diagonal coefficient, preserving component-major blocks."""
		xp: ModuleType = array_namespace(state)
		diagonal: float = float(self.tableau.A[index, index])
		stage_time: float = time + step * self.tableau.C[index]
		guess: BackendArray = state + step * self.tableau.C[index] * initial_field

		def evaluate(candidate: BackendArray) -> tuple[BackendArray, BackendArray]:
			value: BackendArray = _checked_vector_field(self.f, stage_time, candidate)
			return candidate - right_side - step * diagonal * value, value

		def update(candidate: BackendArray, residual: BackendArray, value: BackendArray) -> BackendArray:
			derivative: BackendArray = self._field_jacobian(stage_time, candidate)
			correction: BackendArray
			matrix: BackendArray
			if self.resolved_jacobian_method == 'analytic':
				count: int = derivative.shape[0]
				components: int = derivative.shape[1]
				matrix = xp.broadcast_to(xp.eye(components), derivative.shape) - step * diagonal * derivative
				correction = _solve_particle_systems(matrix, -residual.reshape(components, count).T,
					singular_message='A per-particle DIRK Newton matrix is singular.').T.reshape(-1)
			else:
				matrix = xp.eye(candidate.size) - step * diagonal * derivative
				correction = solve_linear(matrix, -residual, singular_message='The DIRK Newton matrix is singular.')
			correction = checked_finite(correction, 'The DIRK Newton correction became non-finite.', RuntimeError)
			return candidate + correction

		return self.solver.solve(evaluate, guess, update, scale=state,
			context=f'{self.method_name} at stage time t={stage_time} with h={step}',
			time=time, duration=step, stage_index=index, observe=observe)

	def _solve_stages(self, time: float, state: BackendArray, step: float, *, observe: bool) -> _RungeKuttaStep:
		xp: ModuleType = array_namespace(state)
		initial_field: BackendArray = _checked_vector_field(self.f, time, state)
		stages: list[BackendArray] = []
		fields: list[BackendArray] = []
		iterations: list[int] = []
		evaluations: list[int] = []
		norms: list[Any] = []
		converged: list[Any] = []
		for i in range(self.tableau.stage_count):
			right_side: BackendArray = state + step * sum(self.tableau.A[i, j] * fields[j] for j in range(i))
			if self.tableau.A[i, i] == 0:
				stages.append(right_side)
				fields.append(_checked_vector_field(self.f, time + step * self.tableau.C[i], right_side))
				iterations.append(0)
				evaluations.append(0)
				norms.append(0.)
				converged.append(True)
				continue
			root: _NonlinearResult[BackendArray] = self._solve_stage(time, state, step, i,
				right_side, initial_field, observe=observe)
			stages.append(root.unknown)
			fields.append(root.payload)
			iterations.append(root.iterations)
			evaluations.append(root.residual_evaluations)
			norms.append(xp.linalg.norm(root.residual, ord=np.inf))
			converged.append(root.converged)
		after: BackendArray = state + step * sum(weight * value for weight, value in zip(self.tableau.B, fields))
		stage_iterations: BackendArray = xp.asarray(iterations, dtype=int)
		stage_evaluations: BackendArray = xp.asarray(evaluations, dtype=int)
		stage_norms: BackendArray = xp.asarray(norms, dtype=float)
		statistics: dict[str, StepValue] = {
			'nonlinear_iterations': xp.sum(stage_iterations),
			'residual_evaluations': xp.sum(stage_evaluations),
			'nonlinear_residual_norms': xp.max(stage_norms),
			'nonlinear_tolerances': self.solver.tolerance(state),
			'stage_nonlinear_iterations': stage_iterations,
			'stage_residual_evaluations': stage_evaluations,
			'stage_nonlinear_residual_norms': stage_norms}
		return _RungeKuttaStep(after, tuple(stages), statistics, xp.all(xp.asarray(converged)))


__all__ = ['DiagonallyImplicitRungeKutta']
