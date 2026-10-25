"""Coordinate preparation, advancement and finalization of one complete job."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

from contracts.methods.functions import VectorField, JacobianStructure

from contracts.dynamics.protocols import DynamicalSystem, HamiltonianSystem, GuidingCenterJacobianSystem

from contracts.methods.compiled import CompiledStep
from execution.options import ExecutionOptions
from problem import InitialValueProblem
from contracts.states.history import DiagnosticValue
from execution._host import run_steps
from request import SimulationRequest
from states.history import States
from methods.base import IntegrationMethod
from contracts.methods.base import NumericalMethod


if TYPE_CHECKING:
	from diagnostics.methods.energy import EnergyObserver, PreparedEnergyObserver
	from jax import Device


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
		observers: Sequence[EnergyObserver | Callable[[Any], None]] = (),
	) -> States:
		"""Prepare once, advance using the selected worker, then seal one result."""
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
			problem=problem, request=request)
		numerical_run: IntegrationMethod[Any]
		observation_metadata: dict[str, DiagnosticValue]
		numerical_run, observation_metadata = _prepare_numerical_run(problem, method, request,
			prepared_dynamics, track_energy=energy is not None, callbacks=callbacks)
		metadata.update(observation_metadata)

		try:
			numerical_run._status = "running"
			kernel: CompiledStep | None = _prepare_compiled_step(numerical_run, mode, device, callbacks)
			saved: States = States.prepare(request.output_times, problem.initial_state, layout=problem.layout)
			if kernel is None:
				step_diagnostics: dict[str, DiagnosticValue] = run_steps(numerical_run, saved,
					source=problem.dynamics, observers=callbacks, energy=energy)
			else:
				from execution._jax_fixed import run_steps as run_compiled_steps
				step_diagnostics = run_compiled_steps(numerical_run, kernel, saved, energy=energy)
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
		from execution._jax import prepare_environment
		mode = method.jax_mode
		prepared_dynamics, device = prepare_environment(dynamics, method, options)
		metadata.update(execution_mode=mode, execution_device_kind=str(device.device_kind))
		if mode == "hybrid_scipy_controller":
			metadata["adaptive_controller"] = "scipy"
	return prepared_dynamics, device, mode, metadata


def _prepare_observers(
	observers: Sequence[EnergyObserver | Callable[[Any], None]], dynamics: DynamicalSystem,
	*, problem: InitialValueProblem, request: SimulationRequest,
) -> tuple[PreparedEnergyObserver | None, list[Callable[[Any], None]]]:
	"""Bind a fresh energy accumulator and keep caller-owned callbacks in order."""
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
			energy = observer.new_run(dynamics.hamiltonian, dynamics.extended_momentum_derivative,
				particle_count=problem.particle_count, initial_time=request.t_span[0])
		elif callable(observer):
			callbacks.append(observer)
		else:
			raise TypeError("Observers must be EnergyObserver configurations or step callbacks.")
	return energy, callbacks


def _prepare_numerical_run(
	problem: InitialValueProblem, method: NumericalMethod, request: SimulationRequest,
	prepared_dynamics: DynamicalSystem, *, track_energy: bool,
	callbacks: Sequence[Callable[[Any], None]],
) -> tuple[IntegrationMethod[Any], dict[str, DiagnosticValue]]:
	"""Bind numerical inputs, describe observation settings and validate a fresh run."""
	jacobian: VectorField | None = None
	structure: JacobianStructure = "dense"
	# Adaptive defaults retain SciPy's own derivative approximation; fixed
	# algorithms receive the available independent-particle derivative once.
	if method.jax_mode != "hybrid_scipy_controller" and isinstance(prepared_dynamics, GuidingCenterJacobianSystem):
		jacobian, structure = prepared_dynamics.particle_vector_field_jacobians, "particle"
	numerical_run: IntegrationMethod[Any] = method.new_run(
		prepared_dynamics.vector_field, problem.initial_conditions, request,
		jacobian=jacobian, jacobian_structure=structure, observe=bool(callbacks),
		capture_quadrature=track_energy or any(getattr(callback, "requires_quadrature", False) for callback in callbacks),
	)
	metadata: dict[str, DiagnosticValue] = {
		"track_energy": track_energy,
		"momentum_state_dimension": problem.particle_count if track_energy else 0,
	}
	if track_energy:
		metadata.update(energy_observer="EnergyObserver", energy_accumulator_owner="observer")
	if method.jax_mode == "hybrid_scipy_controller":
		metadata["energy_quadrature_nodes"] = 8 if track_energy else 0
	if not isinstance(numerical_run, IntegrationMethod) or numerical_run._status != "ready":
		raise TypeError("Method preparation must return a fresh ready IntegrationMethod.")
	return numerical_run, metadata


def _prepare_compiled_step(
	numerical_run: IntegrationMethod[Any], mode: str, device: Device | None,
	callbacks: Sequence[Callable[[Any], None]],
) -> CompiledStep | None:
	"""Prepare a device-resident step or leave advancement to the host controller."""
	if mode != "device_resident":
		return None
	from methods._jax_dispatch import prepare_step
	if (callbacks or numerical_run.progress
			or getattr(getattr(numerical_run, "solver", None), "iteration_observer", None) is not None):
		raise NotImplementedError(
			"JAX fixed integration requires no step or iteration callbacks "
			"and progress=False; Python callbacks "
			"are supported by the SciPy execution path."
		)
	assert device is not None
	return prepare_step(numerical_run, device)


__all__ = ["Execution"]
