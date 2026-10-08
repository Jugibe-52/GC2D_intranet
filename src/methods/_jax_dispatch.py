"""Typed family adapters preparing optional kernels before integration begins."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax.numpy as jnp

from contracts.compiled import CompiledStep
from contracts.execution_options import ExecutionOptions
from dynamics.protocols import HamiltonianSystem
from execution._jax import require_builtin_dynamics, resolve_device
from integration.core import IntegrationMethod
from methods._jax_options import CompositionOptions, ExplicitOptions, ImplicitOptions, NonlinearOptions
from methods.classical._jax_steps import euler_step, gauss_step, hbvm_step, rk4_step, sdirk_step
from methods.extended.core._jax_projection import extended_step
from methods.classical.euler import ExplicitEuler
from methods.classical.rk4 import RK4
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.sdirk import SDIRK4
from methods.hbvm.order4 import HBVM42
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from methods.extended.core.composition import ABBA2, BM4


Options = TypeVar("Options", ExplicitOptions, ImplicitOptions, CompositionOptions)
_SUPPORTED = (
	ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
	ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint, BM4Implicit, BM4Midpoint,
)


@dataclass(frozen=True)
class PreparedJaxStep(Generic[Options]):
	"""Hashable physical kernel and controls, with one common state-finishing map."""

	dynamics: HamiltonianSystem
	device: Any
	options: Options
	kernel: Callable[[Any, Any, Any, HamiltonianSystem, Options], tuple[Any, Any, dict[str, Any], Any]]

	@property
	def physical_dimension(self) -> int:
		return self.dynamics.state_dimension

	@property
	def copies(self) -> int:
		return self.options.copies

	@property
	def track_energy(self) -> bool:
		return self.options.track_energy

	@property
	def finite_difference(self) -> bool:
		if isinstance(self.options, ExplicitOptions):
			return False
		solve = self.options.solve
		return solve is not None and solve.jacobian == "finite_difference"

	def energy(self, times: Any, physical: Any) -> Any:
		return self.dynamics.hamiltonian(times, physical)

	def __call__(self, time: Any, state: Any, step: Any) -> tuple[Any, dict[str, Any], Any]:
		"""Finish accepted copies and passive momentum without inspecting a method."""
		n = state.size // (self.copies * self.physical_dimension + (2 if self.track_energy else 0))
		physical = state[:self.physical_dimension * n]
		after, increment, statistics, valid = self.kernel(time, physical, step, self.dynamics, self.options)
		parts = [after] * self.copies
		if self.track_energy:
			parts += [jnp.full((n,), time + step), state[-n:] + increment]
		return jnp.concatenate(parts), statistics, valid


def _explicit(method: ExplicitEuler | RK4, dynamics: HamiltonianSystem, device: Any) -> CompiledStep:
	"""Explicit kernels have no nonlinear or projection options."""
	kernel = rk4_step if isinstance(method, RK4) else euler_step
	return PreparedJaxStep(dynamics, device, ExplicitOptions(method.track_energy), kernel)


def _classical(method: GaussLegendre4 | SDIRK4, dynamics: HamiltonianSystem, device: Any) -> CompiledStep:
	"""Use the Jacobian capability resolved during CPU-side run preparation."""
	solve = NonlinearOptions(
		method.newton_absolute_tolerance, method.newton_relative_tolerance,
		method.newton_max_iterations, method.resolved_jacobian_method,
		method.newton_jacobian_relative_step,
	)
	kernel = gauss_step if isinstance(method, GaussLegendre4) else sdirk_step
	return PreparedJaxStep(dynamics, device, ImplicitOptions(method.track_energy, solve), kernel)


def _hbvm(method: HBVM42, dynamics: HamiltonianSystem, device: Any) -> CompiledStep:
	"""Keep HBVM's own damping and norm in its kernel, resolving only controls."""
	jacobian = method.jacobian_method
	if jacobian == "auto":
		jacobian = "analytic" if dynamics.state_dimension == 2 else "finite_difference"
	solve = NonlinearOptions(method.absolute_tolerance, method.relative_tolerance,
		method.max_iterations, jacobian, method.jacobian_relative_step)
	return PreparedJaxStep(dynamics, device, ImplicitOptions(method.track_energy, solve), hbvm_step)


def _abba(method: ABBA2Implicit | ABBA4Implicit | ABBA6Implicit,
		dynamics: HamiltonianSystem, device: Any) -> CompiledStep:
	"""Bind the actual signed recipe and one outer projection."""
	solve = NonlinearOptions(method.newton_absolute_tolerance, method.newton_relative_tolerance,
		method.newton_max_iterations, "analytic", 0., method.nonlinear_solver)
	options = CompositionOptions(method.track_energy, method.recipe.coefficients, None,
		method.projection_formulation, solve, publish_substeps=method.order != 2)
	return PreparedJaxStep(dynamics, device, options, extended_step)


def _bm4(method: BM4Implicit, dynamics: HamiltonianSystem, device: Any) -> CompiledStep:
	"""Bind BM4's coupling and explicit Jacobian selection."""
	solve = NonlinearOptions(method.newton_absolute_tolerance, method.newton_relative_tolerance,
		method.newton_max_iterations, method.newton_jacobian_method,
		method.newton_jacobian_relative_step, method.nonlinear_solver)
	options = CompositionOptions(method.track_energy, BM4.coefficients, method.coupling_frequency,
		"reduced_multiplier", solve)
	return PreparedJaxStep(dynamics, device, options, extended_step)


def _midpoint(method: ABBA2Midpoint | BM4Midpoint, dynamics: HamiltonianSystem, device: Any) -> CompiledStep:
	"""Arithmetic projection has no nonlinear solver configuration."""
	recipe = BM4 if isinstance(method, BM4Midpoint) else ABBA2
	coupling = method.coupling_frequency if isinstance(method, BM4Midpoint) else None
	options = CompositionOptions(method.track_energy, recipe.coefficients, coupling,
		"reduced_multiplier", None)
	return PreparedJaxStep(dynamics, device, options, extended_step)


def prepare_step(method: IntegrationMethod[Any], execution: ExecutionOptions) -> CompiledStep:
	"""Select an adapter once, refusing to discard overrides of built-in methods."""
	if type(method) not in _SUPPORTED:
		raise TypeError("JAX fixed integration requires a supported built-in method; subclass overrides are not compiled.")
	dynamics = require_builtin_dynamics(method.problem.dynamics)
	device = resolve_device(execution)
	if isinstance(method, (ExplicitEuler, RK4)):
		return _explicit(method, dynamics, device)
	if isinstance(method, (GaussLegendre4, SDIRK4)):
		return _classical(method, dynamics, device)
	if isinstance(method, HBVM42):
		return _hbvm(method, dynamics, device)
	if dynamics.state_dimension != 2:
		raise TypeError("Extended GC compositions require planar guiding-centre dynamics.")
	if isinstance(method, (ABBA2Implicit, ABBA4Implicit, ABBA6Implicit)):
		return _abba(method, dynamics, device)
	if isinstance(method, BM4Implicit):
		return _bm4(method, dynamics, device)
	if isinstance(method, (ABBA2Midpoint, BM4Midpoint)):
		return _midpoint(method, dynamics, device)
	raise TypeError("The supported method has no compiled family adapter.")


__all__: list[str] = []
