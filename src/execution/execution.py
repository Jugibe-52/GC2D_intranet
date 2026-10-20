"""Coordinate preparation, advancement and finalization of one complete job."""

from collections.abc import Mapping
from types import MappingProxyType

from contracts.compiled import CompiledStep
from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.result import DiagnosticValue
from execution._host import run_steps
from request import SimulationRequest
from states.history import States
from methods.base import IntegrationMethod, NumericalMethod


NEWTON_ALIASES: Mapping[str, str] = MappingProxyType({
	"newton_iterations": "nonlinear_iterations",
	"newton_residual_norms": "nonlinear_residual_norms",
	"newton_absolute_tolerance": "nonlinear_absolute_tolerance",
	"newton_relative_tolerance": "nonlinear_relative_tolerance",
	"newton_max_iterations": "nonlinear_max_iterations",
})



class Execution:
	"""Own one run lifecycle and result while numerical workers advance states.

	Subclasses override run to perform the complete job elsewhere. Numerical
	methods retain their algorithms and adaptive solvers. The executor keeps no
	per-run state and does not modify global JAX settings or worker configuration.
	"""

	def run(
		self, problem: InitialValueProblem, method: NumericalMethod,
		request: SimulationRequest, *, options: ExecutionOptions | None = None,
	) -> States:
		"""Prepare once, advance using the selected worker, then seal one result."""
		if options is not None and not isinstance(options, ExecutionOptions):
			raise TypeError("`options` must be an ExecutionOptions instance or None.")
		if not isinstance(method, NumericalMethod):
			raise TypeError("`method` must implement NumericalMethod preparation.")
		choice = ExecutionOptions() if options is None else options
		metadata: dict[str, DiagnosticValue] = {
			"execution_backend": choice.backend, "execution_device": choice.device,
			"execution_device_index": choice.device_index,
		}
		prepared = None
		device = None
		mode = "unsupported"
		if choice.backend == "jax":
			from execution._jax import prepare_environment
			mode = method.jax_mode
			prepared, device = prepare_environment(problem.dynamics, method, choice)
			metadata.update(execution_mode=mode, execution_device_kind=str(device.device_kind))
			if mode == "hybrid_scipy_controller":
				metadata["adaptive_controller"] = "scipy"
			run = method.new_run(problem, request,
				evaluated_dynamics=prepared if mode == "hybrid_scipy_controller" else None)
		else:
			run = method.new_run(problem, request)
		if not isinstance(run, IntegrationMethod) or run._status != "ready":
			raise TypeError("Method preparation must return a fresh ready IntegrationMethod.")
		try:
			run._status = "running"
			kernel: CompiledStep | None = None
			if mode == "device_resident":
				from methods._jax_dispatch import prepare_step
				if (run.step_observer is not None or run.progress
						or getattr(run, "newton_observer", None) is not None):
					raise NotImplementedError(
						"JAX fixed integration requires step_observer=None, "
						"newton_observer=None and progress=False; Python callbacks "
						"are supported by the SciPy execution path."
					)
				assert prepared is not None
				kernel = prepare_step(run, prepared, device)
			saved = States.prepare(request.output_times, problem.initial_state, layout=problem.layout)
			if kernel is None:
				details = run_steps(run, saved)
			else:
				from execution._jax_fixed import run_steps as run_compiled_steps
				details = run_compiled_steps(run, kernel, saved)
			return saved.finalize({**run.metadata, **metadata, **details}, aliases=NEWTON_ALIASES)
		finally:
			# Include kernel/controller preparation, recording and finalization failures.
			run._status = "finished"


__all__ = ["Execution"]
