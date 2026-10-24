"""Contracts for the interoperable dynamics/formulation/method architecture."""

from __future__ import annotations

from dataclasses import replace

from diagnostics.states.geometry import calculate_area
from states import generators as state_generators
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics

import unittest
from types import SimpleNamespace
from execution import Execution

import numpy as np

from dynamics import (
	FullCyclotronDynamics,
	GuidingCenterDynamics,
)

from potential import Potential
from simulation import (
	FCSplitMaps,
	GCDoubledMaps,
	InitialValueProblem,
	RK4,
	SimulationRequest,

	States,
	StateLayout,
	simulate,
)


def deterministic_potential() -> Potential:
	"""Return the small field used by architecture regression tests."""
	return Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=3,
	)


class _RotationDynamics:
	"""Minimal non-Hamiltonian dynamics used to prove RK4 interoperability."""

	state_dimension = 2

	def vector_field(self, _t: float, state: np.ndarray) -> np.ndarray:
		x, y = state
		return np.asarray([y, -x])


class _TimePolynomialDynamics:
	"""Non-autonomous field used to check the RK4 stage-time convention."""

	state_dimension = 2

	def vector_field(self, t: float, _state: np.ndarray) -> np.ndarray:
		return np.asarray([t**4, 0.0])


class _FixedOutputExecution(Execution):
	"""External executor returning a predetermined history for boundary validation."""

	def __init__(self, states: np.ndarray, times: np.ndarray | None = None) -> None:
		self.states = states
		self.times = times

	def run(
		self,
		_problem: InitialValueProblem,
		method,
		request: SimulationRequest,
		*, options=None, observers=(),
	) -> States:
		return States(layout=_problem.layout, initial_state=_problem.initial_state,
			t=request.output_times if self.times is None else self.times,
			states=self.states,
			diagnostics={},
		)


class _ExternalParticleLayout:
	"""Minimal planar layout proving structural third-party composition."""

	state_dimension = 2
	component_names = ('x', 'y')
	position_names = ('x', 'y')
	velocity_names = ()

	def validate_packed_state_layout(self, state: np.ndarray) -> np.ndarray:
		value = np.asarray(state)
		if value.ndim == 0 or value.shape[0] == 0 or value.shape[0] % 2:
			raise ValueError("A particle state must contain matching x and y blocks.")
		return value

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		value = self.validate_packed_state_layout(state)
		return tuple(np.split(value, 2, axis=0))

	def particle_count(self, state: np.ndarray) -> int:
		return int(self.validate_packed_state_layout(state).shape[0] // 2)

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return the external layout's x and y particle blocks."""
		x, y = self.split(state)
		return x, y


class _ExternalParticleConfiguration:
	"""Initial-state provider with no inheritance from project base classes."""

	layout = _ExternalParticleLayout()

	@property
	def initial_state(self) -> np.ndarray:
		return np.asarray([1.0, 3.0])


class ExtensibleArchitectureTests(unittest.TestCase):
	"""Verify independent and interoperable numerical composition."""

	def test_physical_dynamics_expose_required_capabilities(self) -> None:
		potential = deterministic_potential()
		gc_configuration = np.array(np.asarray([1.0, 1.2]), dtype=float, copy=True)
		fc_configuration = np.array(np.asarray([1.0, 1.2, 0.4, -0.3]), dtype=float, copy=True)

		for dynamics, state in (
			(
				GuidingCenterDynamics(potential, rho=0.05),
				gc_configuration,
			),
			(
				FullCyclotronDynamics(potential, rho=0.2, eta=0.1),
				fc_configuration,
			),
		):
			assert state is not None
			with self.subTest(dynamics=type(dynamics).__name__):
				self.assertEqual(dynamics.vector_field(0.3, state).shape, state.shape)
				self.assertEqual(dynamics.hamiltonian(0.3, state).shape, (1,))
				self.assertEqual(
					dynamics.extended_momentum_derivative(0.3, state).shape,
					(1,),
				)

	def test_external_configuration_composes_an_independent_layout(self) -> None:
		"""Keep initial-state ownership separate from component interpretation."""
		source = _ExternalParticleConfiguration()
		self.assertIsInstance(source.initial_state, np.ndarray)
		self.assertIsInstance(GuidingCenterDynamics.coordinates, StateLayout)
		solution = States(t=np.asarray([0.0, 1.0]), states=np.asarray([[1.0, 2.0], [3.0, 4.0]]), initial_state=source.initial_state, layout=source.layout)
		self.assertEqual(solution.components()[0].shape, (1, 2))
		self.assertEqual(solution.positions()[0].shape, (1, 2))


	def test_rk4_has_fourth_order_convergence_without_a_formulation(self) -> None:
		source = np.array(np.asarray([1.0, 0.0]), dtype=float, copy=True)
		problem = InitialValueProblem(_RotationDynamics(), source, layout=GuidingCenterDynamics.coordinates)
		exact = np.asarray([np.cos(1.0), -np.sin(1.0)])
		errors = []
		for max_step in (0.1, 0.05):
			solution = simulate(
				problem,
				RK4(step=max_step),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					sample_count=2,
				),
			)
			errors.append(float(np.linalg.norm(solution.states[:, -1] - exact)))
		self.assertGreater(errors[0] / errors[1], 14.0)
		self.assertLess(errors[0] / errors[1], 18.0)

	def test_rk4_uses_non_autonomous_stage_times(self) -> None:
		source = np.array(np.asarray([0.0, 0.0]), dtype=float, copy=True)
		solution = simulate(
			InitialValueProblem(_TimePolynomialDynamics(), source, layout=GuidingCenterDynamics.coordinates),
			RK4(step=0.1),
			SimulationRequest.uniform(
				t_span=(0.0, 1.0),
				sample_count=2,
			),
		)
		self.assertLess(abs(float(solution.states[0, -1]) - 0.2), 1e-6)
		self.assertEqual(float(solution.states[1, -1]), 0.0)

	def test_rk4_step_observer_receives_only_main_grid_steps(self) -> None:
		source = np.array(np.asarray([1.0, 0.0]), dtype=float, copy=True)
		problem = InitialValueProblem(_RotationDynamics(), source, layout=GuidingCenterDynamics.coordinates)
		events = []
		solution = simulate(
			problem,
			RK4(step=0.01),
			SimulationRequest.uniform(
				t_span=(0.0, 0.02),
				sample_count=5,
			),
			observers=(events.append,),
		)

		self.assertEqual(solution.n_steps, 2)
		self.assertEqual(len(events), 2)
		self.assertEqual([event.step_index for event in events], [0, 1])
		for event in events:
			self.assertEqual(event.method_name, "RK4")
			np.testing.assert_allclose(
				event.map_state(event.state_before),
				event.state_after,
			)

	def test_one_rk4_instance_runs_gc_and_fc(self) -> None:
		potential = deterministic_potential()
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.04),
			sample_count=5,
		)
		method = RK4()
		cases = (
			(
				GuidingCenterDynamics(potential, rho=0.05),
				np.array(np.asarray([1.0, 1.2]), dtype=float, copy=True),
				2,
			),
			(
				FullCyclotronDynamics(potential, rho=0.2, eta=0.1),
				np.array(np.asarray([1.0, 1.2, 0.4, -0.3]), dtype=float, copy=True),
				4,
			),
		)
		for dynamics, source, physical_size in cases:
			with self.subTest(dynamics=type(dynamics).__name__):
				solution = simulate(
					InitialValueProblem(dynamics, source),
					replace(method, step=0.01),
					request,
				)
				self.assertEqual(solution.states.shape, (physical_size, 5))
				np.testing.assert_array_equal(
					solution.states[:, 0],
					source,
				)
				self.assertTrue(np.all(np.isfinite(solution.states)))

	def test_incompatible_combinations_fail_before_stepping(self) -> None:
		potential = deterministic_potential()
		gc_source = np.array(np.asarray([1.0, 1.2]), dtype=float, copy=True)
		fc_source = np.array(np.asarray([1.0, 1.2, 0.4, -0.3]), dtype=float, copy=True)
		gc_dynamics = GuidingCenterDynamics(potential, rho=0.05)
		fc_dynamics = FullCyclotronDynamics(potential, rho=0.2, eta=0.1)

		with self.assertRaises(TypeError):
			InitialValueProblem(gc_dynamics, fc_source, layout=FullCyclotronDynamics.coordinates)
		with self.assertRaises(TypeError):
			GCDoubledMaps(fc_dynamics.vector_field, fc_source.size, coordinates=fc_dynamics.coordinates)
		with self.assertRaises(TypeError):
			FCSplitMaps(fc_dynamics.cyclotron_flow, fc_dynamics.electric_kick, gc_source.size, coordinates=gc_dynamics.coordinates)

		problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=0.2), gc_source, layout=GuidingCenterDynamics.coordinates)
		self.assertEqual(problem.dynamics.rho, 0.2)
		fc_problem = InitialValueProblem(FullCyclotronDynamics(potential, rho=0.2, eta=-0.1), fc_source, layout=FullCyclotronDynamics.coordinates)
		self.assertEqual(fc_problem.dynamics.eta, -0.1)

	def test_area_is_source_of_the_temporal_solution(self) -> None:
		potential = deterministic_potential()
		area = state_generators.circle(
			center=(np.pi, np.pi),
			radius=0.2,
			points=16,

		)
		solution = simulate(
			InitialValueProblem(GuidingCenterDynamics(potential, rho=0.05), area, layout=GuidingCenterDynamics.coordinates),
			RK4(step=0.01),
			SimulationRequest.uniform(
				t_span=(0.0, 0.02),
				sample_count=3,
			),
		)

		self.assertIsInstance(area, np.ndarray)
		self.assertIsInstance(GuidingCenterDynamics.coordinates, StateLayout)
		assert area is not None
		self.assertEqual(area.ndim, 1)
		self.assertEqual(solution.states.ndim, 2)
		np.testing.assert_array_equal(solution.initial_state, area)
		np.testing.assert_array_equal(solution.initial_state, area)
		self.assertIs(solution.values, solution.states)
		self.assertEqual(calculate_area(solution.states, layout=GuidingCenterDynamics.coordinates).shape, (3,))
		self.assertEqual(solution.positions()[0].shape, (16, 3))



	def test_runner_rejects_invalid_method_output(self) -> None:
		source = np.array(np.asarray([1.0, 0.0]), dtype=float, copy=True)
		problem = InitialValueProblem(_RotationDynamics(), source, layout=GuidingCenterDynamics.coordinates)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			sample_count=2,
		)
		invalid_histories = (
			np.asarray(1.0),
			np.asarray([1.0, 0.0]),
			np.zeros((2, 0)),
			np.zeros((2, 3)),
			np.zeros((3, 2)),
			np.zeros((4, 2)),
			np.asarray([[1.0, np.nan], [0.0, 0.0]]),
			np.asarray([[2.0, 1.0], [0.0, 0.0]]),
		)
		for states in invalid_histories:
			with self.subTest(states=states):
				with self.assertRaises(ValueError):
					simulate(
						problem,
						RK4(step=0.1),
						request,
						execution=_FixedOutputExecution(states),
					)

	def test_runner_rejects_output_times_that_do_not_match_the_request(self) -> None:
		problem = InitialValueProblem(_RotationDynamics(), np.array(np.asarray([1.0, 0.0]), dtype=float, copy=True), layout=GuidingCenterDynamics.coordinates)
		request = SimulationRequest.uniform(t_span=(0.0, 0.1),  sample_count=2)
		states = np.asarray([[1.0, 0.9], [0.0, -0.1]])
		for times in (
			np.asarray([0.0, 0.2]),
			np.asarray([0.1, 0.0]),
			np.asarray([0.0, np.nan]),
			np.asarray([[0.0, 0.1]]),
		):
			with self.subTest(times=times):
				with self.assertRaises(ValueError):
					simulate(problem, RK4(step=0.1), request, execution=_FixedOutputExecution(states, times))

	def test_gc_and_fc_split_maps_preserve_physical_motion_and_reverse_signed_steps(self) -> None:
		potential = deterministic_potential()
		cases = (
			(
				GCDoubledMaps,
				GuidingCenterDynamics(potential, rho=0.05),
				np.array(np.asarray([1.0, 1.1, 1.2, 1.3]), dtype=float, copy=True),
			),
			(
				FCSplitMaps,
				FullCyclotronDynamics(potential, rho=0.2, eta=0.1),
				np.array(np.asarray([1.0, 1.1, 1.2, 1.3, 0.4, 0.3, -0.3, -0.2]), dtype=float, copy=True),
			),
		)
		for formulation, dynamics, source in cases:
			maps = (GCDoubledMaps(dynamics.vector_field, source.size, coordinates=dynamics.coordinates)
				if formulation is GCDoubledMaps else FCSplitMaps(dynamics.cyclotron_flow,
					dynamics.electric_kick, source.size, coordinates=dynamics.coordinates))
			for forward, reverse in (("direct_map", "adjoint_map"), ("adjoint_map", "direct_map")):
				with self.subTest(formulation=formulation.__name__, forward=forward):
					initial = np.tile(source, 2) if formulation is GCDoubledMaps else source.copy()
					advanced = getattr(maps, forward)(0.01, 0.23, initial)
					restored = getattr(maps, reverse)(-0.01, 0.23, advanced)
					self.assertEqual(advanced.shape, initial.shape)
					self.assertFalse(hasattr(maps, 'dynamics'))
					np.testing.assert_allclose(restored, initial, rtol=0, atol=2e-14)

	def test_request_revalidates_normalized_endpoints(self) -> None:
		with self.assertRaises(ValueError):
			SimulationRequest(
				t_span=(0.0, 1.0),
				output_times=np.asarray([0.0, 1.0 + 1e-15, 1.0 + 2e-15]),
			)

	def test_solution_is_immutable_and_preserves_read_only_aliases(self) -> None:
		source = np.array(np.asarray([1.0, 0.0]), dtype=float, copy=True)
		times = np.asarray([0.0, 0.1])
		states = np.asarray([[1.0, 0.9], [0.0, -0.1]])
		momentum = np.asarray([[0.0, 0.01]])
		solution = States(t=times, states=states, initial_state=source, diagnostics={
				"step_count": 1,
				"extended_momentum": momentum,
				"energy_error": 0.02,
			}, layout=GuidingCenterDynamics.coordinates)

		self.assertIs(solution.values, solution.states)
		np.testing.assert_array_equal(solution.initial_state, source)
		self.assertEqual(solution.n_steps, 1)
		np.testing.assert_array_equal(solution.k, momentum)
		self.assertEqual(solution.err, 0.02)
		times[0] = -1.0
		states[0, 0] = 5.0
		momentum[0, 0] = 7.0
		self.assertEqual(solution.t[0], 0.0)
		self.assertEqual(solution.states[0, 0], 1.0)
		self.assertEqual(solution.k[0, 0], 0.0)  # type: ignore[index]
		with self.assertRaises(ValueError):
			solution.t[0] = -1.0
		with self.assertRaises(ValueError):
			solution.states[0, 0] = 2.0
		with self.assertRaises(ValueError):
			solution.k[0, 0] = 2.0  # type: ignore[index]
		with self.assertRaises(TypeError):
			solution.diagnostics["step_count"] = 2  # type: ignore[index]


if __name__ == "__main__":
	unittest.main()
