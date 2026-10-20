"""Optional environment preparation for executor-owned JAX integrations."""

from typing import Any

from contracts.compiled import CompiledStep
from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.result import DiagnosticValue
from execution._jax import prepare_dynamics, resolve_device
from integration.core import IntegrationMethod
from methods.base import NumericalMethod
from request import SimulationRequest


def prepare_jax_run(
    problem: InitialValueProblem, method: NumericalMethod,
    request: SimulationRequest, options: ExecutionOptions,
) -> tuple[IntegrationMethod[Any], CompiledStep | None, dict[str, DiagnosticValue]]:
    """Resolve the environment once, then bind the declared numerical capability."""
    mode = method.jax_mode
    if mode not in ("device_resident", "hybrid_scipy_controller"):
        raise NotImplementedError(f"{type(method).__name__} does not provide JAX execution.")
    device = resolve_device(options)
    dynamics = prepare_dynamics(problem.dynamics)
    from methods._jax_dispatch import prepare_step, validate_hybrid_method

    evaluated = None
    if mode == "hybrid_scipy_controller":
        validate_hybrid_method(method)
        from dynamics._jax_host import bind_host_dynamics
        evaluated = bind_host_dynamics(dynamics, device)
    run = method.new_run(problem, request, evaluated_dynamics=evaluated)
    if not isinstance(run, IntegrationMethod) or run._status != "ready":
        raise TypeError("Method preparation must return a fresh ready IntegrationMethod.")
    try:
        kernel = None
        if mode == "device_resident":
            if (run.step_observer is not None or run.progress
                    or getattr(run, "newton_observer", None) is not None):
                raise NotImplementedError(
                    "JAX fixed integration requires step_observer=None, "
                    "newton_observer=None and progress=False; Python callbacks "
                    "are supported by the SciPy execution path."
                )
            kernel = prepare_step(run, dynamics, device)
        details: dict[str, DiagnosticValue] = {
            "execution_mode": mode, "execution_device_kind": str(device.device_kind),
        }
        if mode == "hybrid_scipy_controller":
            details["adaptive_controller"] = "scipy"
        return run, kernel, details
    except BaseException:
        run._status = "finished"
        raise
