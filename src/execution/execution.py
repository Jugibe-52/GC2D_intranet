"""Local execution boundary for one complete integration."""

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import IntegrationData
from methods.base import NumericalMethod


class Execution:
    """Execute integrations locally without configuring the runtime.

    Subclasses override ``run`` to perform the same complete job elsewhere.
    Methods and integration retain ownership of numerical steps, field
    evaluations and nonlinear solves. This executor keeps no per-run state
    and does not configure threads, affinity, workers or JAX global settings.
    """

    def run(
        self,
        problem: InitialValueProblem,
        method: NumericalMethod,
        request: SimulationRequest,
        *, options: ExecutionOptions | None = None,
    ) -> IntegrationData:
        """Return the complete physical history and integration diagnostics.

        Backend options belong to this job, independently of its executor.
        Omission uses the existing NumPy/SciPy path. The caller constructs
        and validates the final Solution from the returned IntegrationData.
        """
        if options is not None and not isinstance(options, ExecutionOptions):
            raise TypeError("`options` must be an ExecutionOptions instance or None.")
        if options is not None and options.backend == "jax":
            return method.integrate(problem, request, execution=options)
        # Preserve support for external CPU methods with the two-argument API.
        return method.integrate(problem, request)


__all__ = ["Execution"]
