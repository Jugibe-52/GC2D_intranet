"""Fully coupled Runge--Kutta stages with dense or independent particle Newton solves."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import cast
import numpy as np

from contracts.arrays import BackendArray, array_namespace, checked_finite, map_arrays
from contracts.methods.step import StepValue
from methods.runge_kutta.base import _NewtonRungeKutta, _RungeKuttaStep
from methods.runge_kutta._jacobians import _checked_vector_field
from solvers.linear import solve_linear, _solve_particle_systems
from solvers.nonlinear import _NonlinearResult


@dataclass(frozen=True, slots=True)
class _StageEvaluation:
	"""Residuals and fields at s candidate physical stage vectors."""

	residuals: tuple[BackendArray, ...]
	fields: tuple[BackendArray, ...]


@dataclass
class ImplicitRungeKutta(_NewtonRungeKutta):
	"""Solve all stage states together; no stage count or tableau is hard-coded."""

	def _correction(self, step: float, evaluation: _StageEvaluation,
		jacobians: tuple[BackendArray, ...]) -> BackendArray:
		"""Assemble blocks delta_ij I - h A_ij J_j, preserving particle independence."""
		xp: ModuleType = array_namespace(jacobians)
		stages: int = self.tableau.stage_count
		particle: bool = self.resolved_jacobian_method == 'analytic'
		components: int = jacobians[0].shape[-1]
		identity: BackendArray = xp.eye(components)
		if particle:
			identity = xp.broadcast_to(identity, jacobians[0].shape)
		rows: list[BackendArray] = []
		for i in range(stages):
			blocks: list[BackendArray] = []
			for j in range(stages):
				block: BackendArray = -step * self.tableau.A[i, j] * jacobians[j]
				blocks.append(block + identity if i == j else block)
			rows.append(xp.concatenate(blocks, axis=-1))
		matrix: BackendArray = xp.concatenate(rows, axis=-2)
		residual: BackendArray
		correction: BackendArray
		if particle:
			residual = xp.concatenate(tuple(r.reshape(components, -1).T for r in evaluation.residuals), axis=1)
			correction = _solve_particle_systems(matrix, -residual,
				singular_message='A per-particle Runge--Kutta Newton matrix is singular.')
			return cast(BackendArray, xp.concatenate(tuple(
				correction[:, i * components:(i + 1) * components].T.reshape(-1) for i in range(stages))))
		residual = xp.concatenate(evaluation.residuals)
		return solve_linear(matrix, -residual, singular_message='The Runge--Kutta Newton matrix is singular.')

	def _solve_stages(self, time: float, state: BackendArray, step: float, *, observe: bool) -> _RungeKuttaStep:
		xp: ModuleType = array_namespace(state)
		size: int = state.size
		count: int = self.tableau.stage_count
		stage_times: BackendArray = time + step * self.tableau.C
		initial_field: BackendArray = _checked_vector_field(self.f, time, state)
		guesses: tuple[BackendArray, ...] = tuple(state + step * node * initial_field for node in self.tableau.C)

		def evaluate(unknown: BackendArray) -> tuple[BackendArray, _StageEvaluation]:
			stage_states: BackendArray = unknown.reshape(count, size)
			fields: tuple[BackendArray, ...] = tuple(map_arrays(
				lambda t, z: _checked_vector_field(self.f, t, z), stage_times, stage_states))
			residuals: tuple[BackendArray, ...] = tuple(stage_states[i] - state - step * sum(
				self.tableau.A[i, j] * fields[j] for j in range(count)) for i in range(count))
			return xp.concatenate(residuals), _StageEvaluation(residuals, fields)

		def update(unknown: BackendArray, residual: BackendArray, evaluation: _StageEvaluation) -> BackendArray:
			jacobians: tuple[BackendArray, ...] = tuple(map_arrays(
				self._field_jacobian, stage_times, unknown.reshape(count, size)))
			correction: BackendArray = self._correction(step, evaluation, jacobians)
			correction = checked_finite(correction, 'The Runge--Kutta Newton correction became non-finite.', RuntimeError)
			return unknown + correction

		root: _NonlinearResult[_StageEvaluation] = self.solver.solve(evaluate, xp.concatenate(guesses), update,
			scale=state, context=f'{self.method_name} at t={time} with h={step}',
			time=time, duration=step, observe=observe)
		after: BackendArray = state + step * sum(weight * value for weight, value in zip(self.tableau.B, root.payload.fields))
		statistics: dict[str, StepValue] = {'nonlinear_iterations': root.iterations,
			'residual_evaluations': root.residual_evaluations,
			'nonlinear_residual_norms': xp.linalg.norm(root.residual, ord=np.inf),
			'nonlinear_tolerances': self.solver.tolerance(state)}
		return _RungeKuttaStep(after, tuple(root.unknown[i * size:(i + 1) * size].copy()
			for i in range(count)), statistics, root.converged)


__all__ = ['ImplicitRungeKutta']
