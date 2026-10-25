"""Compile ordinary fixed-step advancement with prepared algebraic solvers.

This boundary validates supported implementations and packages their common
step result. Numerical stages, residuals and compositions remain in methods.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp

from contracts.methods.compiled import CompiledStep
from contracts.methods.functions import VectorField
from contracts.methods.step import StepResult
from methods.base import IntegrationMethod
from methods.adaptive.scipy import DOP853, Radau, _AdaptiveMethod
from methods.classical.euler import ExplicitEuler
from methods.classical.rk4 import RK4
from methods.classical.gauss_legendre import GaussLegendre4, _StageEvaluation
from methods.classical.sdirk import SDIRK4
from methods.hbvm.order4 import HBVM42, _ResidualPayload
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from methods.extended.palindromic import PalindromicImplicit, PalindromicMidpoint
from methods.extended.core.records import CompositionTrace, MapStage, OperationStage


# Register numerical payloads only at the optional execution boundary. Operation
# names and grouping directions are static; stage values remain device arrays.
for _record, _metadata in ((_StageEvaluation, ()), (_ResidualPayload, ()),
	(CompositionTrace, ()), (MapStage, ('direct',)), (OperationStage, ('operation',))):
	jax.tree_util.register_dataclass(_record,
		data_fields=[item.name for item in fields(_record) if item.name not in _metadata],
		meta_fields=list(_metadata))

_SUPPORTED_FIXED = (ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
	ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint, BM4Implicit,
	BM4Midpoint, PalindromicImplicit, PalindromicMidpoint)


def execution_mode(method: object) -> str:
	"""Own the optional compatibility boundary without replacing overrides."""
	if isinstance(method, _AdaptiveMethod):
		if type(method) not in (DOP853, Radau):
			raise TypeError('Hybrid JAX execution requires a built-in adaptive method.')
		return 'hybrid_scipy_controller'
	if type(method) not in _SUPPORTED_FIXED:
		raise TypeError('JAX fixed integration requires a supported built-in method; subclass overrides are not compiled.')
	return 'device_resident'


@dataclass(frozen=True)
class PreparedStep:
	"""An isolated common method, with immutable controls determining reuse."""

	signature: tuple[Any, ...]
	f: VectorField
	device: jax.Device
	physical_dimension: int
	copies: int
	finite_difference: bool
	_run: IntegrationMethod[Any] = field(compare=False, hash=False, repr=False)

	def __call__(self, time: Any, state: Any, step: Any) -> tuple[Any, Any, dict[str, Any], Any]:
		result: StepResult[Any] = self._run.advance(jnp.asarray(time), jnp.asarray(state), jnp.asarray(step), observe=False)
		return result.state, result.quadrature, dict(result.statistics), jnp.asarray(result.converged)


def prepare_step(method: IntegrationMethod[Any], device: jax.Device) -> CompiledStep:
	"""Copy run resources, prepare the solver and bind the same advance function."""
	if execution_mode(method) != 'device_resident':
		raise TypeError('A compiled fixed step requires a fixed method.')
	prepared: Any = copy(method)
	solver: Any = getattr(prepared, 'solver', None)
	if solver is not None:
		prepared.solver = solver._as_jax()
		# Rebind method-owned residual closures to the prepared solver. No stages
		# or physical initial data are changed by this isolated initialization.
		prepared.initialize(prepared.request)
	selected: str = getattr(prepared, 'resolved_jacobian_method',
		'analytic' if prepared.jacobian is not None else 'finite_difference')
	if solver is not None and solver.jacobian_method != 'auto':
		selected = solver.jacobian_method
	assert is_dataclass(method)
	controls: tuple[Any, ...] = tuple((item.name, getattr(method, item.name))
		for item in fields(method) if item.init)
	return PreparedStep((type(method), controls, method.jacobian, method.jacobian_structure,
		method.physical_size, method.capture_quadrature, selected), method.f, device,
		method.layout.state_dimension, method.copies, solver is not None and selected == 'finite_difference', prepared)


__all__: list[str] = []
