"""Explicit Runge--Kutta stages for any strictly lower-triangular tableau."""

from dataclasses import dataclass
import numpy as np

from contracts.arrays import BackendArray
from methods.runge_kutta.base import RungeKutta, _RungeKuttaStep
from methods.runge_kutta._jacobians import _checked_vector_field


@dataclass
class ExplicitRungeKutta(RungeKutta):
	"""Evaluate successive stages without an algebraic solver."""

	def _validate_tableau(self) -> None:
		if np.any(np.triu(self.tableau.A) != 0):
			raise ValueError('Explicit Runge--Kutta requires strictly lower-triangular A.')

	def _solve_stages(self, time: float, state: BackendArray, step: float, *, observe: bool) -> _RungeKuttaStep:
		stages: list[BackendArray] = []
		fields: list[BackendArray] = []
		for i in range(self.tableau.stage_count):
			stage: BackendArray = state + step * sum(
				self.tableau.A[i, j] * fields[j] for j in range(i) if self.tableau.A[i, j] != 0)
			stages.append(stage)
			fields.append(_checked_vector_field(self.f, time + step * self.tableau.C[i], stage))
		after: BackendArray = state + step * sum(weight * value for weight, value in zip(self.tableau.B, fields) if weight != 0)
		return _RungeKuttaStep(after, tuple(stages))


__all__ = ['ExplicitRungeKutta']
