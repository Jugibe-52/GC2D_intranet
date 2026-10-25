"""Optional JAX realizations of the ordinary immutable solver configurations.

Residuals and updates keep the same meaning as on the host. Failed compiled
solves return ``converged=False`` for the execution boundary to report; no
Python callback runs in a device iteration. Payloads must be array pytrees.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from contracts.arrays import BackendArray, checked_finite

from solvers.nonlinear import NewtonSolver, BroydenSolver, _NonlinearResult, _Payload, _ResidualFunction
from solvers.derivatives import centered_difference
from solvers._jax import Root, newton, broyden, infinity_norm


def _initial(value: BackendArray) -> Any:
	"""Validate static shape and preserve non-finite failures during tracing."""
	array: Any = jnp.asarray(value, dtype=float)
	if array.ndim != 1 or array.size == 0:
		raise ValueError('The initial nonlinear unknown must be a finite vector.')
	return checked_finite(array, 'The initial nonlinear unknown must be a finite vector.')


def _evaluation(value: tuple[BackendArray, _Payload], shape: tuple[int, ...]) -> tuple[Any, _Payload]:
	"""Check a cached or freshly evaluated residual without repeating its call."""
	residual: Any = jnp.asarray(value[0], dtype=float)
	if residual.shape != shape:
		raise ValueError('The nonlinear residual must be finite and match the unknown shape.')
	return checked_finite(residual, 'The nonlinear residual must be finite and match the unknown shape.'), value[1]


class JaxNewtonSolver(NewtonSolver):
	"""Newton with device iteration and the inherited controls and solve inputs."""

	def tolerance(self, scale: BackendArray) -> Any:
		return self.absolute_tolerance + self.relative_tolerance * jnp.maximum(1., infinity_norm(scale))

	def solve(
		self, evaluate: _ResidualFunction[_Payload], initial: BackendArray,
		update: Callable[[BackendArray, BackendArray, _Payload], BackendArray] | None = None,
		*, scale: BackendArray, jacobian: Callable[[BackendArray], BackendArray] | None = None,
		context: str = 'nonlinear system',
		initial_evaluation: tuple[BackendArray, _Payload] | None = None,
		norm: Callable[[BackendArray], float] | None = None,
		update_returns_correction: bool = False,
		time: float = 0., duration: float = 0., stage_index: int | None = None,
		multiplier_slice: slice | None = None, observe: bool = True,
	) -> _NonlinearResult[_Payload]:
		"""Run a fresh solve; account for residual probes used by differences."""
		if self.iteration_observer is not None and observe:
			raise NotImplementedError('JAX solvers do not support Python iteration callbacks.')
		if update is not None and jacobian is not None:
			raise ValueError('Supply at most one of `update` or `jacobian`.')
		initial = _initial(initial)
		def checked(value: BackendArray) -> tuple[Any, _Payload]:
			return _evaluation(evaluate(value), initial.shape)
		cached: tuple[Any, _Payload] | None = (None if initial_evaluation is None
			else _evaluation(initial_evaluation, initial.shape))
		probes = 0
		if update is None:
			finite_difference = self.jacobian_method == 'finite_difference' or jacobian is None
			probes = 2 * initial.size if finite_difference else 0
			def correction(value: Any, residual: Any, payload: _Payload) -> Any:
				# An already solved initial residual requires no derivative.
				# A missing required derivative makes a needed correction invalid.
				if self.jacobian_method == 'analytic' and jacobian is None:
					return jnp.full_like(value, jnp.nan)
				matrix: Any = (centered_difference(lambda z: evaluate(z)[0], value,
					relative_step=self.jacobian_relative_step) if finite_difference else jacobian(value) if jacobian is not None else None)
				return jnp.linalg.solve(matrix, -residual)
			update = correction
			update_returns_correction = True
		root: Root = newton(checked, initial, update, self.tolerance(scale),
			self.max_iterations, norm=infinity_norm if norm is None else norm,
			backtracking=self.backtracking, initial_evaluation=cached,
			update_returns_correction=update_returns_correction)
		return _NonlinearResult(root.unknown, root.residual, root.payload, root.iterations,
			root.evaluations + probes * root.iterations, root.converged)


class JaxBroydenSolver(BroydenSolver):
	"""Good Broyden with a complete device-side secant matrix per solve."""

	def tolerance(self, scale: BackendArray) -> Any:
		return self.absolute_tolerance + self.relative_tolerance * jnp.maximum(1., infinity_norm(scale))

	def solve(
		self, evaluate: _ResidualFunction[_Payload], initial: BackendArray,
		initial_jacobian: BackendArray, *, scale: BackendArray,
		context: str = 'nonlinear system',
		initial_evaluation: tuple[BackendArray, _Payload] | None = None,
	) -> _NonlinearResult[_Payload]:
		initial = _initial(initial)
		matrix: Any = jnp.asarray(initial_jacobian, dtype=float)
		if matrix.shape != (initial.size, initial.size):
			raise ValueError('The initial Broyden Jacobian must match the unknown.')
		matrix = checked_finite(matrix, 'The initial Broyden Jacobian must be finite.')
		if not isinstance(context, str) or not context:
			raise ValueError('The Broyden solve context must be a non-empty string.')
		def checked(value: BackendArray) -> tuple[Any, _Payload]:
			return _evaluation(evaluate(value), initial.shape)
		cached: tuple[Any, _Payload] | None = (None if initial_evaluation is None
			else _evaluation(initial_evaluation, initial.shape))
		root: Root = broyden(checked, initial, matrix,
			self.tolerance(scale), self.max_iterations, initial_evaluation=cached)
		return _NonlinearResult(root.unknown, root.residual, root.payload,
			root.iterations, root.evaluations, root.converged)


__all__ = ['JaxNewtonSolver', 'JaxBroydenSolver']
