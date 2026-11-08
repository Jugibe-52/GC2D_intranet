"""Preparation shared by integrators that solve algebraic step equations."""

from typing import Generic, Self, TypeVar

from contracts.methods.functions import VectorField
from methods.base import IntegrationMethod
from solvers.nonlinear import AlgebraicSolver
from states.initial import InitialConditions

Detail = TypeVar('Detail')


class ImplicitIntegrationMethod(IntegrationMethod[Detail], Generic[Detail]):
	"""Bind an optional field derivative; families own their solver defaults.

	Analytic derivatives return independent particle blocks (N,d,d). The solver
	selects their use or numerical differences; implicitness does not require
	a supplied derivative. No physical function is evaluated during binding.
	"""

	jacobian: VectorField | None
	solver: AlgebraicSolver

	def new_run(
		self, f: VectorField, initial_conditions: InitialConditions,
		*, jacobian: VectorField | None = None,
	) -> Self:
		"""Bind the derivative before initializing stage or projection equations."""
		if jacobian is not None and not callable(jacobian):
			raise TypeError('Supply an optional Jacobian function.')
		method: Self = self._copy_for_run(f, initial_conditions)
		method.jacobian = jacobian
		method._initialize_run(initial_conditions)
		return method


__all__ = ['ImplicitIntegrationMethod']
