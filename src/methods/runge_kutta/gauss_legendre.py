"""Two-stage Gauss--Legendre coefficients and its specialized observation."""

from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np
from contracts.methods.runge_kutta.gauss_legendre import GaussLegendre4IntegrationStep
from contracts.methods.step import StepInfo, StepResult
from methods.runge_kutta.base import _RungeKuttaStep
from methods.runge_kutta.implicit import ImplicitRungeKutta
from methods.runge_kutta.tableau import ButcherTableau
from solvers.derivatives import JacobianMethod as GaussJacobianMethod

GAUSS_JACOBIAN_METHODS: tuple[GaussJacobianMethod, ...] = ('auto', 'analytic', 'finite_difference')
_ROOT: float = float(np.sqrt(3.) / 6.)


@dataclass
class GaussLegendre4(ImplicitRungeKutta):
	"""Fourth-order symmetric Gauss recipe with inherited coupled stage solves."""

	tableau: ButcherTableau = field(default=ButcherTableau(
		((.25, .25 - _ROOT), (.25 + _ROOT, .25)),
		(.5, .5), (.5 - _ROOT, .5 + _ROOT)), init=False)
	designed_order: ClassVar[int] = 4

	def build_observation(self, info: StepInfo, step: StepResult[_RungeKuttaStep]) -> GaussLegendre4IntegrationStep:
		"""Preserve the two physical stage snapshots used by analytic diagnostics."""
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(self.advance(info.time, candidate, info.duration, observe=False).state)
		return GaussLegendre4IntegrationStep(
			dynamics_name='', method_name=type(self).__name__, step_index=info.index,
			start_time=info.time, time=info.time + info.duration, duration=info.duration,
			state_before=info.state_before.copy(), state_after=step.state.copy(), map_state=map_state,
			first_stage_time=info.time + info.duration * self.tableau.C[0],
			second_stage_time=info.time + info.duration * self.tableau.C[1],
			newton_iterations=int(step.statistics['nonlinear_iterations']),
			residual_evaluations=int(step.statistics['residual_evaluations']),
			newton_residual_norm=float(step.statistics['nonlinear_residual_norms']),
			newton_tolerance=float(step.statistics['nonlinear_tolerances']),
			first_stage_state=np.asarray(step.details.stage_states[0].copy()),
			second_stage_state=np.asarray(step.details.stage_states[1].copy()))


__all__ = ['GAUSS_JACOBIAN_METHODS', 'GaussJacobianMethod', 'GaussLegendre4']
