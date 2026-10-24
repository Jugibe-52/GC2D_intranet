"""Bind optional physical kernels to functions and solver-derived controls."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

from jax import Device

from contracts.methods.compiled import CompiledStep
from contracts.methods.functions import VectorField
from methods.base import IntegrationMethod
from methods._jax_options import CompositionOptions, ExplicitOptions, ImplicitOptions
from solvers._jax import NonlinearOptions
from methods.classical._jax_steps import euler_step, gauss_step, hbvm_step, rk4_step, sdirk_step
from methods.extended.core._jax_projection import extended_step
from methods.adaptive.scipy import DOP853, Radau
from methods.classical.euler import ExplicitEuler
from methods.classical.rk4 import RK4
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.sdirk import SDIRK4
from methods.hbvm.order4 import HBVM42
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from methods.extended.palindromic import PalindromicImplicit, PalindromicMidpoint, PalindromicMethod, ProjectedPalindromicMethod
from methods.projection import resolve_projection


Options = TypeVar('Options', ExplicitOptions, ImplicitOptions, CompositionOptions)
StepKernel: TypeAlias = Callable[[Any, Any, Any, VectorField, VectorField | None, int, Options],
                                tuple[Any, Any, dict[str, Any], Any]]


@dataclass(frozen=True)
class PreparedJaxStep(Generic[Options]):
	"""Hashable physical kernel retaining only supplied functions and static data."""

	f: VectorField
	jacobian: VectorField | None
	physical_dimension: int
	device: Device
	options: Options
	kernel: StepKernel[Options]

	@property
	def copies(self) -> int:
		return self.options.copies

	@property
	def finite_difference(self) -> bool:
		return not isinstance(self.options, ExplicitOptions) and self.options.solve is not None and self.options.solve.jacobian == 'finite_difference'

	def __call__(self, time: Any, state: Any, step: Any) -> tuple[Any, Any, dict[str, Any], Any]:
		return self.kernel(time, state, step, self.f, self.jacobian, self.physical_dimension, self.options)


def validate_hybrid_method(method: object) -> None:
	"""Refuse custom adaptive overrides before binding compiled field callbacks."""
	if type(method) not in (DOP853, Radau):
		raise TypeError('Hybrid JAX execution requires a built-in adaptive method.')


def prepare_step(method: IntegrationMethod[Any], device: Device) -> CompiledStep:
	"""Select one kernel, preserving exact-type restrictions and resolved policies."""
	if type(method) in (ExplicitEuler, RK4) and isinstance(method, (ExplicitEuler, RK4)):
		kernel: StepKernel[ExplicitOptions] = rk4_step if isinstance(method, RK4) else euler_step
		return PreparedJaxStep(method.f, method.jacobian, method.layout.state_dimension, device,
			ExplicitOptions(method.capture_quadrature), kernel)
	if type(method) in (GaussLegendre4, SDIRK4, HBVM42) and isinstance(method, (GaussLegendre4, SDIRK4, HBVM42)):
		if isinstance(method, HBVM42):
			selected = method.solver.jacobian_method
			if selected == 'auto':
				selected = 'analytic' if method.jacobian is not None else 'finite_difference'
		else:
			selected = method.resolved_jacobian_method
		solve = NonlinearOptions.from_solver(method.solver, selected)
		implicit_kernel: StepKernel[ImplicitOptions] = hbvm_step if isinstance(method, HBVM42) else (gauss_step if isinstance(method, GaussLegendre4) else sdirk_step)
		return PreparedJaxStep(method.f, method.jacobian, method.layout.state_dimension, device,
			ImplicitOptions(method.capture_quadrature, solve), implicit_kernel)
	if type(method) in (ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Implicit, ABBA2Midpoint, BM4Midpoint, PalindromicImplicit, PalindromicMidpoint) and isinstance(method, PalindromicMethod):
		configuration = (NonlinearOptions.from_solver(method.solver, method.resolved_jacobian_method)
			if isinstance(method, ProjectedPalindromicMethod) else None)
		projection = resolve_projection(getattr(method, 'projection_formulation', 'reduced_multiplier')).name
		options = CompositionOptions(method.capture_quadrature, method.recipe,
			method.formulation.coupling_frequency, projection, configuration,
			publish_substeps=isinstance(method, (ABBA4Implicit, ABBA6Implicit)))
		return PreparedJaxStep(method.f, method.jacobian, method.layout.state_dimension, device, options, extended_step)
	raise TypeError('JAX fixed integration requires a supported built-in method; subclass overrides are not compiled.')


__all__: list[str] = []
