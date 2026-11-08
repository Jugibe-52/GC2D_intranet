"""Explicit stage-based reconstruction for existing generalized-energy studies.

This study helper is separate from EnergyObserver's endpoint observations.
It preserves historical kappa diagnostics without adding auxiliary integration
or energy functions to the generic executor or numerical methods.
"""
from collections.abc import Callable, Sequence
from typing import Any
import numpy as np
from contracts.methods.base import NumericalMethod
from contracts.methods.observation import IntegrationStep
from contracts.methods.step import StepResult
from contracts.states.history import DiagnosticValue
from contracts.dynamics.protocols import GuidingCenterJacobianSystem, HamiltonianSystem
from diagnostics.methods.stage_energy import StageEnergyObserver, PreparedStageEnergyObserver
from execution.options import ExecutionOptions
from methods.base import IntegrationMethod
from methods.implicit import ImplicitIntegrationMethod
from problem import InitialValueProblem
from simulation.runner import simulate
from states.history import States
from states.energy import energy_diagnostics


def simulate_generalized_energy(problem: InitialValueProblem, method: NumericalMethod,
	*, options: ExecutionOptions | None = None,
	observers: Sequence[Callable[[Any], None]] = ()) -> States:
	"""Reconstruct historical passive momentum from accepted numerical stages."""
	dynamics: Any = problem.dynamics
	if not isinstance(dynamics, HamiltonianSystem):
		raise TypeError("Generalized-energy studies require Hamiltonian dynamics.")
	observer: PreparedStageEnergyObserver = StageEnergyObserver().new_run(dynamics.hamiltonian,
		lambda t, z: -dynamics.hamiltonian_time_derivative(t, z),
		particle_count=problem.particle_count, initial_time=problem.request.t_span[0])
	times: list[float] = [problem.request.t_span[0]]
	momenta: list[np.ndarray] = [observer.momentum]
	states: list[np.ndarray] = [problem.initial_state]

	def accept(event: IntegrationStep) -> None:
		if event.quadrature is None:
			raise ValueError("Historical reconstruction requires numerical stage data.")
		observer.accept(event.step_index, event.start_time, event.time, event.quadrature)
		times.append(event.time)
		momenta.append(observer.momentum)
		states.append(np.array(event.state_after, copy=True))

	solution: States = simulate(problem, method, options=options, observers=(accept, *observers))
	run: IntegrationMethod[Any]
	if isinstance(method, ImplicitIntegrationMethod):
		run = method.new_run(dynamics.vector_field, problem.initial_conditions,
			jacobian=dynamics.particle_vector_field_jacobians if isinstance(dynamics, GuidingCenterJacobianSystem) else None)
	else:
		run = method.new_run(dynamics.vector_field, problem.initial_conditions)
	samples: list[np.ndarray] = []
	for time in solution.times:
		index: int = max(0, int(np.searchsorted(times, time, side="right")) - 1)
		if np.isclose(time, times[index], rtol=0., atol=64*np.finfo(float).eps*max(1., abs(time))):
			samples.append(momenta[index])
		else:
			step: StepResult[Any] = run.advance(times[index], states[index], float(time - times[index]), observe=False)
			if step.quadrature is None:
				raise ValueError("Historical sampling requires numerical stage data.")
			samples.append(momenta[index] + observer._increment(step.quadrature))
	momentum: np.ndarray = np.stack(samples, axis=1)
	diagnostics: dict[str, DiagnosticValue] = energy_diagnostics(solution.times, solution.states.T,
		{"extended_momentum": momentum}, energy=np.asarray(dynamics.hamiltonian(solution.times, solution.states.T)))
	diagnostics.update(track_energy=True, energy_observer="StageEnergyObserver",
		momentum_state_dimension=problem.particle_count)
	return solution.with_diagnostics(diagnostics)
