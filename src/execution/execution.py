"""Coordinate preparation, advancement and finalization of one complete job."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

from contracts.methods.functions import VectorField

from contracts.dynamics.protocols import DynamicalSystem, HamiltonianSystem, GuidingCenterJacobianSystem

from contracts.methods.compiled import CompiledStep
from execution.options import ExecutionOptions
from problem import InitialValueProblem
from contracts.states.history import DiagnosticValue
from execution._host import run_steps, FixedStepController
from execution._plan import build_fixed_step_plan
from contracts.execution.plan import FixedStepPlan
from contracts.execution.controllers import StepController
from request import SimulationRequest
from states.history import States
from states.initial import InitialConditions
from methods.base import IntegrationMethod, FixedStepMethod
from contracts.methods.base import NumericalMethod


if TYPE_CHECKING:
	from diagnostics.methods.energy import EnergyObserver, PreparedEnergyObserver
	from jax import Device
	from execution._jax_fixed import PreparedIntegration


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
	methods retain their numerical algorithms. The executor keeps no
	per-run state and does not modify global JAX settings or worker configuration.
	"""

	def run(
		self, problem: InitialValueProblem, method: NumericalMethod,
		*, options: ExecutionOptions | None = None,
		observers: Sequence[EnergyObserver | Callable[[Any], None]] = (),
	) -> States:
		"""Prepare once, advance using the selected worker, then seal one result."""
		if not isinstance(problem, InitialValueProblem):
			raise TypeError("`problem` must be an InitialValueProblem.")
		request: SimulationRequest = problem.request
		if options is not None and not isinstance(options, ExecutionOptions):
			raise TypeError("`options` must be an ExecutionOptions instance or None.")
		if not isinstance(method, NumericalMethod):
			raise TypeError("`method` must implement NumericalMethod preparation.")
		execution_options: ExecutionOptions = ExecutionOptions() if options is None else options
		prepared_dynamics: DynamicalSystem
		device: Device | None
		mode: str
		metadata: dict[str, DiagnosticValue]
		prepared_dynamics, device, mode, metadata = _prepare_backend(problem.dynamics, method, execution_options)
		energy: PreparedEnergyObserver | None
		callbacks: list[Callable[[Any], None]]
		energy, callbacks = _prepare_observers(observers, prepared_dynamics,
			problem=problem)
		numerical_run: IntegrationMethod[Any]
		observation_metadata: dict[str, DiagnosticValue]
		numerical_run, observation_metadata = _prepare_numerical_run(
			problem.initial_conditions, method, prepared_dynamics,
			track_energy=energy is not None)
		metadata.update(observation_metadata)

		try:
			numerical_run._status = "running"
			prepared: PreparedIntegration | None = None
			controller: StepController[Any] | None = None
			plan: FixedStepPlan | None = None
			if mode != "device_resident":
				controller = numerical_run.controller()
			if mode == "device_resident" or type(controller) is FixedStepController:
				if not isinstance(numerical_run, FixedStepMethod):
					raise TypeError("Fixed-step planning requires a FixedStepMethod.")
				plan = build_fixed_step_plan(request, numerical_run.step)
				if isinstance(controller, FixedStepController):
					controller.plan = plan
			if mode == "device_resident":
				from execution._jax_step import prepare_step
				from execution._jax_fixed import prepare_integration
				assert device is not None and plan is not None
				kernel: CompiledStep = prepare_step(numerical_run, device, initial_conditions=problem.initial_conditions, callbacks=callbacks, progress=execution_options.progress)
				prepared = prepare_integration(numerical_run, kernel, plan, initial_state=problem.initial_state, energy=energy)
			saved: States = States.prepare(request.output_times, problem.initial_state, layout=problem.layout)
			if prepared is None:
				assert controller is not None
				step_diagnostics: dict[str, DiagnosticValue] = run_steps(numerical_run, saved, request,
					source=problem.dynamics, controller=controller, observers=callbacks, energy=energy,
					initial_state=problem.initial_state, progress=execution_options.progress)
			else:
				from execution._jax_fixed import run_steps as run_compiled_steps
				step_diagnostics = run_compiled_steps(prepared, saved)
			return saved.finalize({**numerical_run.metadata, **metadata, **step_diagnostics}, aliases=NEWTON_ALIASES)
		finally:
			# Include kernel/controller preparation, recording and finalization failures.
			numerical_run._status = "finished"


def _prepare_backend(
	dynamics: DynamicalSystem, method: NumericalMethod, options: ExecutionOptions,
) -> tuple[DynamicalSystem, Device | None, str, dict[str, DiagnosticValue]]:
	"""Resolve dynamics, device and metadata; load JAX only when selected."""
	metadata: dict[str, DiagnosticValue] = {
		"execution_backend": options.backend, "execution_device": options.device,
		"execution_device_index": options.device_index,
	}
	prepared_dynamics: DynamicalSystem = dynamics
	device: Device | None = None
	mode = "unsupported"
	if options.backend == "jax":
		from execution._jax import prepare_dynamics, resolve_device
		from execution._jax_step import execution_mode
		mode = execution_mode(method)
		device = resolve_device(options)
		prepared_dynamics = prepare_dynamics(dynamics)
		metadata.update(execution_mode=mode, execution_device_kind=str(device.device_kind))

	return prepared_dynamics, device, mode, metadata


def _prepare_observers(
	observers: Sequence[EnergyObserver | Callable[[Any], None]], dynamics: DynamicalSystem,
	*, problem: InitialValueProblem,
) -> tuple[PreparedEnergyObserver | None, list[Callable[[Any], None]]]:
	"""Bind endpoint energy observation and keep caller-owned callbacks in order."""
	energy: PreparedEnergyObserver | None = None
	callbacks: list[Callable[[Any], None]] = []
	if observers:
		from diagnostics.methods.energy import EnergyObserver
	for observer in observers:
		if isinstance(observer, EnergyObserver):
			if energy is not None:
				raise ValueError("Register at most one EnergyObserver per run.")
			if not isinstance(dynamics, HamiltonianSystem):
				raise TypeError("Energy observation requires HamiltonianSystem.")
			energy = observer.new_run(dynamics.hamiltonian, dynamics.hamiltonian_time_derivative,
				particle_count=problem.particle_count)
		elif callable(observer):
			callbacks.append(observer)
		else:
			raise TypeError("Observers must be EnergyObserver configurations or step callbacks.")
	return energy, callbacks


def _prepare_numerical_run(
	initial_conditions: InitialConditions,
	method: NumericalMethod, prepared_dynamics: DynamicalSystem, *, track_energy: bool,
) -> tuple[IntegrationMethod[Any], dict[str, DiagnosticValue]]:
	"""Bind explicit initial inputs and validate a fresh numerical run."""
	jacobian: VectorField | None = None
	if isinstance(prepared_dynamics, GuidingCenterJacobianSystem):
		jacobian = prepared_dynamics.particle_vector_field_jacobians
	numerical_run: IntegrationMethod[Any] = method.new_run(
		prepared_dynamics.vector_field, initial_conditions, jacobian=jacobian)
	metadata: dict[str, DiagnosticValue] = {"track_energy": track_energy}
	if track_energy:
		metadata.update(energy_observer="EnergyObserver", energy_derivative="analytic_partial_t_H")
	if not isinstance(numerical_run, IntegrationMethod) or numerical_run._status != "ready":
		raise TypeError("Method preparation must return a fresh ready IntegrationMethod.")
	return numerical_run, metadata


__all__ = ["Execution"]
