"""Bind supported method configurations to one optional compiled fixed-step map."""

from typing import Any

import jax.numpy as jnp

from dynamics._jax import JaxDynamics
from methods._jax_common import MethodOptions
from methods.classical._jax_steps import classical_step
from methods.extended.core._jax_projection import extended_step
from methods.classical.euler import ExplicitEuler
from methods.classical.rk4 import RK4
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.sdirk import SDIRK4
from methods.hbvm.order4 import HBVM42
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from methods.extended.core.composition import ABBA2, BM4


_SUPPORTED = (ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
              ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint, BM4Implicit, BM4Midpoint)


def method_options(method: Any, dynamics: JaxDynamics) -> MethodOptions:
    """Snapshot validated controls, refusing to discard subclass overrides."""
    if type(method) not in _SUPPORTED:
        raise TypeError('JAX fixed integration requires a supported built-in method; subclass overrides are not compiled.')
    name = type(method).__name__
    parameters: dict[str, Any] = {
        'name': name, 'track_energy': method.track_energy,
        'copies': method.state_formulation.copies,
    }
    if isinstance(method, (GaussLegendre4, SDIRK4, BM4Implicit, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit)):
        parameters.update(atol=method.newton_absolute_tolerance, rtol=method.newton_relative_tolerance,
                          max_iterations=method.newton_max_iterations,
                          relative_step=getattr(method, 'newton_jacobian_relative_step', 0.),
                          solver=getattr(method, 'nonlinear_solver', 'newton'),
                          jacobian=getattr(method, 'resolved_jacobian_method', getattr(method, 'newton_jacobian_method', 'analytic')))
    if isinstance(method, HBVM42):
        jacobian = method.jacobian_method
        if jacobian == 'auto':
            jacobian = 'analytic' if dynamics.dimension == 2 else 'finite_difference'
        parameters.update(atol=method.absolute_tolerance, rtol=method.relative_tolerance,
                          max_iterations=method.max_iterations, relative_step=method.jacobian_relative_step,
                          jacobian=jacobian)
    if method.state_formulation.copies == 2:
        if dynamics.dimension != 2:
            raise TypeError('Extended GC compositions require planar guiding-centre dynamics.')
        recipe = (BM4 if isinstance(method, (BM4Implicit, BM4Midpoint)) else
                  ABBA2 if isinstance(method, ABBA2Midpoint) else method.recipe)
        parameters.update(coefficients=recipe.coefficients, coupling=getattr(method, 'coupling_frequency', None),
                          projection=getattr(method, 'projection_formulation', 'reduced_multiplier'))
    return MethodOptions(**parameters)


def advance(time: Any, before: Any, step: Any, dynamics: JaxDynamics,
            options: MethodOptions) -> tuple[Any, dict[str, Any], Any]:
    """Finish the shared physical/doubled state and optional passive momentum."""
    n = before.size // (options.copies * dynamics.dimension + (2 if options.track_energy else 0))
    physical = before[:dynamics.dimension * n]
    apply = extended_step if options.copies == 2 else classical_step
    after, increment, statistics, valid = apply(time, physical, step, dynamics, options)
    parts = [after] * options.copies
    if options.track_energy:
        parts += [jnp.full((n,), time + step), before[-n:] + increment]
    return jnp.concatenate(parts), statistics, valid


__all__: list[str] = []
