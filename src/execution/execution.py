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
		choice: ExecutionOptions = ExecutionOptions() if options is None else options
		metadata: dict[str, DiagnosticValue] = {
			"execution_backend": choice.backend, "execution_device": choice.device,
			"execution_device_index": choice.device_index,
		}
		prepared: DynamicalSystem = problem.dynamics
		device: Device | None = None
		mode = "unsupported"
		if choice.backend == "jax":
			from execution._jax import prepare_environment
			mode = method.jax_mode
			prepared, device = prepare_environment(problem.dynamics, method, choice)
			metadata.update(execution_mode=mode, execution_device_kind=str(device.device_kind))
			if mode == "hybrid_scipy_controller":
				metadata["adaptive_controller"] = "scipy"
		energy: PreparedEnergyObserver | None = None
		callbacks: list[Callable[[Any], None]] = []
		if observers:
			from diagnostics.methods.energy import EnergyObserver
		for observer in observers:
			if isinstance(observer, EnergyObserver):
				if energy is not None:
					raise ValueError("Register at most one EnergyObserver per run.")
				if not isinstance(prepared, HamiltonianSystem):
					raise TypeError("Energy observation requires HamiltonianSystem.")
				energy = observer.new_run(prepared.hamiltonian, prepared.extended_momentum_derivative,
					particle_count=problem.particle_count, initial_time=request.t_span[0])
			elif callable(observer):
				callbacks.append(observer)
			else:
				raise TypeError("Observers must be EnergyObserver configurations or step callbacks.")
		jacobian: VectorField | None = None
		structure: JacobianStructure = "dense"
		# Adaptive defaults retain SciPy's own derivative approximation; fixed
		# algorithms receive the available independent-particle derivative once.
		if method.jax_mode != "hybrid_scipy_controller" and isinstance(prepared, GuidingCenterJacobianSystem):
			jacobian, structure = prepared.particle_vector_field_jacobians, "particle"
		run: IntegrationMethod[Any] = method.new_run(prepared.vector_field, problem.initial_conditions, request,
			jacobian=jacobian, jacobian_structure=structure, observe=bool(callbacks),
			capture_quadrature=energy is not None or any(getattr(callback, "requires_quadrature", False) for callback in callbacks))
		metadata.update(track_energy=energy is not None,
			momentum_state_dimension=problem.particle_count if energy is not None else 0)
		if energy is not None:
			metadata.update(energy_observer="EnergyObserver", energy_accumulator_owner="observer")
		if method.jax_mode == "hybrid_scipy_controller":
			metadata["energy_quadrature_nodes"] = 8 if energy is not None else 0
		if not isinstance(run, IntegrationMethod) or run._status != "ready":
			raise TypeError("Method preparation must return a fresh ready IntegrationMethod.")
		try:
			run._status = "running"
			kernel: CompiledStep | None = None
			if mode == "device_resident":
				from methods._jax_dispatch import prepare_step
				if (callbacks or run.progress
						or getattr(getattr(run, "solver", None), "iteration_observer", None) is not None):
					raise NotImplementedError(
						"JAX fixed integration requires no step or iteration callbacks "
						"and progress=False; Python callbacks "
						"are supported by the SciPy execution path."
					)
				assert device is not None
				kernel = prepare_step(run, device)
			saved: States = States.prepare(request.output_times, problem.initial_state, layout=problem.layout)
			if kernel is None:
				# Record physical samples and step statistics in saved; return a dictionary
				# with output_interpolation_count and optional formulation diagnostics.
				details: dict[str, DiagnosticValue] = run_steps(run, saved, source=problem.dynamics, observers=callbacks, energy=energy)
			else:
				from execution._jax_fixed import run_steps as run_compiled_steps
				details = run_compiled_steps(run, kernel, saved, energy=energy)
			return saved.finalize({**run.metadata, **metadata, **details}, aliases=NEWTON_ALIASES)
		finally:
			# Include kernel/controller preparation, recording and finalization failures.
			run._status = "finished"


__all__ = ["Execution"]
