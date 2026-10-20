"""Prepare and coordinate one complete local integration."""

from contracts.compiled import CompiledStep
from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.result import DiagnosticValue
from execution._host import integrate_method
from request import SimulationRequest
from states.history import States
from methods.base import IntegrationMethod, NumericalMethod


class Execution:
    """Own run preparation and driver selection without changing global settings.

    Subclasses override ``run`` to perform the same complete job elsewhere.
    Methods own numerical steps, field evaluations and nonlinear solves;
    internal execution drivers own time control and sampling. This executor keeps no per-run state
    and does not configure threads, affinity, workers or JAX global settings.
    """

    def run(
        self,
        problem: InitialValueProblem,
        method: NumericalMethod,
        request: SimulationRequest,
        *, options: ExecutionOptions | None = None,
    ) -> States:
        """Return the complete physical history and integration diagnostics.

        Backend options belong to this job, independently of its executor.
        Methods supply fresh numerical resources; the executor starts the shared
        driver and supplies execution metadata without copying a finished history.
        """
        if options is not None and not isinstance(options, ExecutionOptions):
            raise TypeError("`options` must be an ExecutionOptions instance or None.")
        if not isinstance(method, NumericalMethod):
            raise TypeError("`method` must implement NumericalMethod preparation.")
        choice = ExecutionOptions() if options is None else options
        metadata: dict[str, DiagnosticValue] = {
            "execution_backend": choice.backend, "execution_device": choice.device,
            "execution_device_index": choice.device_index,
        }
        kernel: CompiledStep | None = None
        if choice.backend == "jax":
            # Keep the ordinary CPU path independent of the optional JAX package.
            from execution._jax_run import prepare_jax_run
            run, kernel, details = prepare_jax_run(problem, method, request, choice)
            metadata.update(details)
        else:
            run = method.new_run(problem, request)
        if not isinstance(run, IntegrationMethod) or run._status != "ready":
            raise TypeError("Method preparation must return a fresh ready IntegrationMethod.")
        try:
            if kernel is None:
                return integrate_method(run, metadata=metadata)
            from execution._jax_fixed import integrate_fixed
            return integrate_fixed(run, kernel, metadata=metadata)
        finally:
            # Include failures during controller or output preparation.
            run._status = "finished"


__all__ = ["Execution"]
