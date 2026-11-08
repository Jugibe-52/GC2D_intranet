"""Compile ordinary fixed-step advancement with prepared algebraic solvers.

This boundary validates supported implementations and packages their common
step result. Numerical stages, residuals and compositions remain in methods.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp

from contracts.methods.compiled import CompiledStep
from contracts.methods.functions import VectorField
from contracts.methods.step import StepResult
from states.initial import InitialConditions
from methods.base import IntegrationMethod
from methods.runge_kutta.euler import ExplicitEuler
from methods.runge_kutta.rk4 import RK4
from methods.runge_kutta.gauss_legendre import GaussLegendre4
from methods.runge_kutta.implicit import _StageEvaluation, ImplicitRungeKutta
from methods.runge_kutta.explicit import ExplicitRungeKutta
from methods.runge_kutta.diagonally_implicit import DiagonallyImplicitRungeKutta
from methods.runge_kutta.sdirk import SDIRK4
from methods.hbvm.order4 import HBVM42, _ResidualPayload
from methods.palindromic.abba2_implicit import ABBA2Implicit
from methods.palindromic.abba4_implicit import ABBA4Implicit
from methods.palindromic.abba2_midpoint import ABBA2Midpoint
from methods.palindromic.bm4_implicit import BM4Implicit
from methods.palindromic.bm4_midpoint import BM4Midpoint
from methods.palindromic.implicit import PalindromicImplicit
from methods.palindromic.midpoint import PalindromicMidpoint
from methods.palindromic.core.records import CompositionTrace, OperationStage


# Register numerical payloads only at the optional execution boundary. Operation
# names are static; stage values remain device arrays.
for _record, _metadata in ((_StageEvaluation, ()), (_ResidualPayload, ()),
	(CompositionTrace, ()), (OperationStage, ('operation',))):
	jax.tree_util.register_dataclass(_record,
		data_fields=[item.name for item in fields(_record) if item.name not in _metadata],
		meta_fields=list(_metadata))

_SUPPORTED_FIXED = (ExplicitRungeKutta, DiagonallyImplicitRungeKutta, ImplicitRungeKutta, ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
	ABBA2Implicit, ABBA4Implicit, ABBA2Midpoint, BM4Implicit,
	BM4Midpoint, PalindromicImplicit, PalindromicMidpoint)


def execution_mode(method: object) -> str:
	"""Own the optional compatibility boundary without replacing overrides."""
	if type(method) not in _SUPPORTED_FIXED:
		raise TypeError('JAX fixed integration requires a supported built-in method; subclass overrides are not compiled.')
	return 'device_resident'


@dataclass(frozen=True)
class PreparedStep:
	"""An isolated common method, with immutable controls determining reuse."""

	signature: tuple[Any, ...]
	f: VectorField
	device: jax.Device
	finite_difference: bool
	_run: IntegrationMethod[Any] = field(compare=False, hash=False, repr=False)

	def __call__(self, time: Any, state: Any, step: Any) -> tuple[Any, dict[str, Any], Any]:
		result: StepResult[Any] = self._run.advance(jnp.asarray(time), jnp.asarray(state), jnp.asarray(step), observe=False)
		return result.state, dict(result.statistics), jnp.asarray(result.converged)


def prepare_step(
	method: IntegrationMethod[Any], device: jax.Device,
	*, initial_conditions: InitialConditions, callbacks: Sequence[Callable[[Any], None]] = (), progress: bool = False,
) -> CompiledStep:
	"""Validate fixed execution, isolate resources and bind the same advance function."""
	if type(method) not in _SUPPORTED_FIXED:
		raise TypeError('JAX fixed integration requires a supported built-in method; subclass overrides are not compiled.')
	if (callbacks or progress
			or getattr(getattr(method, "solver", None), "iteration_observer", None) is not None):
		raise NotImplementedError(
			"JAX fixed integration requires no step or iteration callbacks "
			"and progress=False; Python callbacks "
			"are supported by the SciPy execution path."
		)
	prepared: Any = copy(method)
	solver: Any = getattr(prepared, 'solver', None)
	if solver is not None:
		prepared.solver = solver._as_jax()
		# Rebind method-owned residual closures to the prepared solver. No stages
		# or physical initial data are changed by this isolated initialization.
		prepared.initialize(initial_conditions)
	selected: str = getattr(prepared, 'resolved_jacobian_method',
		'analytic' if prepared.jacobian is not None else 'finite_difference')
	if solver is not None and solver.jacobian_method != 'auto':
		selected = solver.jacobian_method
	assert is_dataclass(method)
	controls: tuple[Any, ...] = tuple((item.name, getattr(method, item.name))
		for item in fields(method) if item.init)
	return PreparedStep((type(method), controls, method.jacobian,
		initial_conditions.values.size, selected), method.f, device,
		solver is not None and selected == 'finite_difference', prepared)



__all__: list[str] = []
