"""Shared Runge--Kutta configuration, quadrature and numerical observations."""

from __future__ import annotations
from states.initial import InitialConditions

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar
from types import ModuleType

import numpy as np

from contracts.arrays import BackendArray, array_namespace, checked_finite
from contracts.methods.observation import IntegrationStep
from contracts.methods.quadrature import StageQuadrature
from contracts.methods.step import StepInfo, StepResult, StepValue
from contracts.states.history import DiagnosticValue
from methods.base import FixedStepMethod
from methods.runge_kutta.tableau import ButcherTableau
from methods.runge_kutta._jacobians import _checked_analytic_jacobians, _dense_finite_difference_jacobian
from solvers.derivatives import ResolvedJacobianMethod, resolve_jacobian_method
from solvers.nonlinear import NewtonSolver, validate_solver


@dataclass(frozen=True, slots=True)
class _RungeKuttaStep:
	"""Accepted physical state and s temporary physical stages, each (d*N,)."""

	state: BackendArray
	stage_states: tuple[BackendArray, ...]
	statistics: dict[str, StepValue] = field(default_factory=dict)
	converged: Any = True


@dataclass
class RungeKutta(FixedStepMethod[_RungeKuttaStep]):
	"""Common fixed-step contract; stage algorithms are supplied by subclasses."""

	tableau: ButcherTableau = field(kw_only=True)
	designed_order: ClassVar[int | None] = None
	tableau_name: ClassVar[str | None] = None
	tableau_metadata: ClassVar[tuple[tuple[str, DiagnosticValue], ...]] = ()

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		if not isinstance(self.tableau, ButcherTableau):
			raise TypeError('Supply a ButcherTableau instance.')
		self._validate_tableau()

	def _validate_tableau(self) -> None:
		"""Apply any structural restriction of the selected stage algorithm."""

	def initialize(self, initial_conditions: InitialConditions) -> None:
		"""Describe coefficients without inferring an unverified method order."""
		metadata: dict[str, DiagnosticValue] = dict(self.tableau_metadata)
		metadata['stage_count'] = self.tableau.stage_count
		metadata.update(tableau_A=self.tableau.A, tableau_B=self.tableau.B, tableau_C=self.tableau.C)
		if self.designed_order is not None:
			metadata['designed_order'] = self.designed_order
		if self.tableau_name is not None:
			metadata['tableau_name'] = self.tableau_name
		self.metadata = metadata

	@abstractmethod
	def _solve_stages(self, time: float, state: BackendArray, step: float, *, observe: bool) -> _RungeKuttaStep:
		"""Apply the tableau using explicit, diagonal or coupled stage equations."""

	def advance(self, time: float, state: BackendArray, step: float, *, observe: bool = True) -> StepResult[_RungeKuttaStep]:
		"""Advance the physical map and expose its stage quadrature."""
		xp: ModuleType = array_namespace(state)
		physical: BackendArray = xp.asarray(self._checked_state(state), dtype=float)
		result: _RungeKuttaStep = self._solve_stages(time, physical, step, observe=observe)
		after: BackendArray = checked_finite(result.state, 'The Runge--Kutta state became non-finite.', RuntimeError)
		quadrature: StageQuadrature = StageQuadrature(tuple(time + step * self.tableau.C),
			result.stage_states, tuple(step * self.tableau.B))
		return StepResult(after, result.statistics, result, quadrature, result.converged)

	def build_observation(self, info: StepInfo, result: StepResult[_RungeKuttaStep]) -> IntegrationStep:
		"""Replay the physical map silently, without adding Newton observations."""
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(self.advance(info.time, candidate, info.duration, observe=False).state)
		return IntegrationStep(dynamics_name='', method_name=type(self).__name__,
			step_index=info.index, start_time=info.time, time=info.time + info.duration,
			duration=info.duration, state_before=info.state_before.copy(),
			state_after=result.state.copy(), map_state=map_state)


@dataclass
class _NewtonRungeKutta(RungeKutta):
	"""Common algebraic preparation; solvers retain all nonlinear controls."""

	solver: NewtonSolver = field(default_factory=lambda: NewtonSolver(1e-14, 1e-13, 20), kw_only=True)
	resolved_jacobian_method: ResolvedJacobianMethod = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		RungeKutta.__post_init__(self)
		validate_solver(self.solver, newton_only=True)

	@property
	def _nonlinear_solve_count(self) -> int:
		return 1

	def initialize(self, initial_conditions: InitialConditions) -> None:
		RungeKutta.initialize(self, initial_conditions)
		self.resolved_jacobian_method = resolve_jacobian_method(self.jacobian, self.solver.jacobian_method)
		self.metadata = {**self.metadata, 'nonlinear_solver': 'newton',
			'nonlinear_solves_per_step': self._nonlinear_solve_count,
			'nonlinear_absolute_tolerance': self.solver.absolute_tolerance,
			'nonlinear_relative_tolerance': self.solver.relative_tolerance,
			'nonlinear_max_iterations': self.solver.max_iterations,
			'newton_jacobian_method': self.resolved_jacobian_method,
			'requested_newton_jacobian_method': self.solver.jacobian_method,
			'newton_jacobian_relative_step': self.solver.jacobian_relative_step}

	def _field_jacobian(self, time: float, state: BackendArray) -> BackendArray:
		"""Validate the supplied derivative or use the solver's difference scale."""
		if self.resolved_jacobian_method == 'analytic':
			assert self.jacobian is not None
			return _checked_analytic_jacobians(self.jacobian, time, state)
		return _dense_finite_difference_jacobian(self.f, time, state,
			relative_step=self.solver.jacobian_relative_step)


__all__ = ['RungeKutta']
