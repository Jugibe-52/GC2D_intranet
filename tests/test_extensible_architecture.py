"""Contracts for the interoperable dynamics/formulation/method architecture."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from dynamics import (
	FullCyclotronDynamics,
	GuidingCenterDynamics,
)
from initial_conditions import (
	Area,
	FCInitialConfiguration,
	GCInitialConfiguration,
)
from potential import Potential
from simulation import (
	BM4Composition,
	FCSplitFormulation,
	GCExtendedFormulation,
	InitialConfiguration,
	InitialValueProblem,
	RK4,
	SimulationRequest,
	SimulationRunner,
	Solution,
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


class _FixedOutputMethod:
	"""Minimal third-party-like method returning a predetermined history."""

	def __init__(self, states: np.ndarray) -> None:
		self.states = states

	def integrate(
		self,
		_problem: InitialValueProblem,
		request: SimulationRequest,
	) -> SimpleNamespace:
		return SimpleNamespace(
			t=request.output_times,
			states=self.states,
			diagnostics={},
		)


class ExtensibleArchitectureTests(unittest.TestCase):
	"""Verify independent and interoperable numerical composition."""

	def test_physical_dynamics_expose_required_capabilities(self) -> None:
		potential = deterministic_potential()
		gc_configuration = GCInitialConfiguration(
			np.asarray([1.0, 1.2]),
		)
		fc_configuration = FCInitialConfiguration(
			np.asarray([1.0, 1.2, 0.4, -0.3]),
		)

		for dynamics, state in (
			(
				GuidingCenterDynamics(potential, rho=0.05),
				gc_configuration.initial_state,
			),
			(
				FullCyclotronDynamics(potential, rho=0.2, eta=0.1),
				fc_configuration.initial_state,
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

	def test_bm4_formulations_interoperate_through_one_runner(self) -> None:
		potential = deterministic_potential()
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.04),
			max_step=0.01,
			sample_count=7,
		)
		cases = (
			(
				GCInitialConfiguration(np.asarray([1.0, 1.2])),
				GuidingCenterDynamics(potential, rho=0.05),
				GCExtendedFormulation(coupling_frequency=2.5),
				2,
			),
			(
				FCInitialConfiguration(
					np.asarray([1.0, 1.2, 0.4, -0.3]),
				),
				FullCyclotronDynamics(potential, rho=0.2, eta=0.1),
				FCSplitFormulation(),
				4,
			),
		)

		for source, dynamics, formulation, physical_size in cases:
			with self.subTest(source=type(source).__name__):
				solution = simulate(
					InitialValueProblem(dynamics, source),
					BM4Composition(formulation, track_energy=True),
					request,
				)
				self.assertEqual(solution.states.shape, (physical_size, 7))
				self.assertEqual(np.asarray(solution.k).shape, (1, 7))
				self.assertEqual(solution.n_steps, 4)
				self.assertIs(solution.source, source)

	def test_bm4_matches_pre_refactor_golden_values(self) -> None:
		potential = deterministic_potential()
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.04),
			max_step=0.01,
			sample_count=5,
		)
		gc = simulate(
			InitialValueProblem(
				GuidingCenterDynamics(potential, rho=0.05),
				GCInitialConfiguration(np.asarray([1.0, 1.2])),
			),
			BM4Composition(GCExtendedFormulation(), track_energy=True),
			request,
		)
		fc = simulate(
			InitialValueProblem(
				FullCyclotronDynamics(potential, rho=0.2, eta=0.1),
				FCInitialConfiguration(
					np.asarray([1.0, 1.2, 0.4, -0.3]),
				),
			),
			BM4Composition(FCSplitFormulation(), track_energy=True),
			request,
		)

		np.testing.assert_allclose(
			gc.states[:, -1],
			[0.9988091473170178, 1.2012404342104765],
			atol=1e-12,
			rtol=0.0,
		)
		np.testing.assert_allclose(
			np.asarray(gc.k)[:, -1],
			[0.00053432194466351],
			atol=1e-12,
			rtol=0.0,
		)
		np.testing.assert_allclose(
			fc.states[:, -1],
			[
				1.0145676429052044,
				1.1863758215414457,
				0.3256681908352309,
				-0.3788239173127920,
			],
			atol=1e-12,
			rtol=0.0,
		)
		np.testing.assert_allclose(
			np.asarray(fc.k)[:, -1],
			[0.00267734528550445],
			atol=1e-12,
			rtol=0.0,
		)

	def test_rk4_has_fourth_order_convergence_without_a_formulation(self) -> None:
		source = GCInitialConfiguration(np.asarray([1.0, 0.0]))
		problem = InitialValueProblem(_RotationDynamics(), source)
		exact = np.asarray([np.cos(1.0), -np.sin(1.0)])
		errors = []
		for max_step in (0.1, 0.05):
			solution = simulate(
				problem,
				RK4(),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=max_step,
					sample_count=2,
				),
			)
			errors.append(float(np.linalg.norm(solution.states[:, -1] - exact)))
		self.assertGreater(errors[0] / errors[1], 14.0)
		self.assertLess(errors[0] / errors[1], 18.0)

	def test_rk4_uses_non_autonomous_stage_times(self) -> None:
		source = GCInitialConfiguration(np.asarray([0.0, 0.0]))
		solution = simulate(
			InitialValueProblem(_TimePolynomialDynamics(), source),
			RK4(),
			SimulationRequest.uniform(
				t_span=(0.0, 1.0),
				max_step=0.1,
				sample_count=2,
			),
		)
		self.assertLess(abs(float(solution.states[0, -1]) - 0.2), 1e-6)
		self.assertEqual(float(solution.states[1, -1]), 0.0)

	def test_rk4_step_observer_receives_only_main_grid_steps(self) -> None:
		source = GCInitialConfiguration(np.asarray([1.0, 0.0]))
		problem = InitialValueProblem(_RotationDynamics(), source)
		events = []
		solution = simulate(
			problem,
			RK4(step_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.02),
				max_step=0.01,
				sample_count=5,
			),
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
			max_step=0.01,
			sample_count=5,
		)
		method = RK4()
		cases = (
			(
				GuidingCenterDynamics(potential, rho=0.05),
				GCInitialConfiguration(np.asarray([1.0, 1.2])),
				2,
			),
			(
				FullCyclotronDynamics(potential, rho=0.2, eta=0.1),
				FCInitialConfiguration(
					np.asarray([1.0, 1.2, 0.4, -0.3]),
				),
				4,
			),
		)
		for dynamics, source, physical_size in cases:
			with self.subTest(dynamics=type(dynamics).__name__):
				solution = SimulationRunner().simulate(
					InitialValueProblem(dynamics, source),
					method,
					request,
				)
				self.assertEqual(solution.states.shape, (physical_size, 5))
				np.testing.assert_array_equal(
					solution.states[:, 0],
					source.initial_state,
				)
				self.assertTrue(np.all(np.isfinite(solution.states)))

	def test_incompatible_combinations_fail_before_stepping(self) -> None:
		potential = deterministic_potential()
		gc_source = GCInitialConfiguration(np.asarray([1.0, 1.2]))
		fc_source = FCInitialConfiguration(
			np.asarray([1.0, 1.2, 0.4, -0.3]),
		)
		gc_dynamics = GuidingCenterDynamics(potential, rho=0.05)
		fc_dynamics = FullCyclotronDynamics(potential, rho=0.2, eta=0.1)

		with self.assertRaises(TypeError):
			InitialValueProblem(gc_dynamics, fc_source)
		with self.assertRaises(TypeError):
			GCExtendedFormulation().prepare(
				InitialValueProblem(fc_dynamics, fc_source),
				track_energy=False,
			)
		with self.assertRaises(TypeError):
			FCSplitFormulation().prepare(
				InitialValueProblem(gc_dynamics, gc_source),
				track_energy=False,
			)
		problem = InitialValueProblem(
			GuidingCenterDynamics(potential, rho=0.2),
			gc_source,
		)
		self.assertEqual(problem.dynamics.rho, 0.2)
		fc_problem = InitialValueProblem(
			FullCyclotronDynamics(potential, rho=0.2, eta=-0.1),
			fc_source,
		)
		self.assertEqual(fc_problem.dynamics.eta, -0.1)

	def test_area_is_source_of_the_temporal_solution(self) -> None:
		potential = deterministic_potential()
		area = Area.circle(
			center=(np.pi, np.pi),
			radius=0.2,
			points=16,
			rho=0.05,
		)
		solution = simulate(
			InitialValueProblem(
				GuidingCenterDynamics(potential, rho=0.05),
				area,
			),
			RK4(),
			SimulationRequest.uniform(
				t_span=(0.0, 0.02),
				max_step=0.01,
				sample_count=3,
			),
		)

		self.assertIsInstance(area, InitialConfiguration)
		assert area.initial_state is not None
		self.assertEqual(area.initial_state.ndim, 1)
		self.assertEqual(solution.states.ndim, 2)
		self.assertIs(solution.source, area)
		self.assertIs(solution.trajectory, area)
		self.assertIs(solution.y, solution.states)
		self.assertEqual(area.calculate_area(solution.states).shape, (3,))
		self.assertEqual(solution.positions()[0].shape, (16, 3))

	def test_bm4_output_density_does_not_change_common_samples(self) -> None:
		potential = deterministic_potential()
		source = GCInitialConfiguration(np.asarray([1.0, 1.2]))
		problem = InitialValueProblem(
			GuidingCenterDynamics(potential, rho=0.05),
			source,
		)
		method = BM4Composition(GCExtendedFormulation())
		sparse = simulate(
			problem,
			method,
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=3,
			),
		)
		dense = simulate(
			problem,
			method,
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=7,
			),
		)
		self.assertEqual(sparse.n_steps, 3)
		self.assertEqual(dense.n_steps, 3)
		np.testing.assert_array_equal(sparse.states[:, 1], dense.states[:, 3])
		np.testing.assert_array_equal(sparse.states[:, -1], dense.states[:, -1])

	def test_energy_diagnostics_do_not_change_physical_states(self) -> None:
		potential = deterministic_potential()
		source = GCInitialConfiguration(np.asarray([1.0, 1.2]))
		problem = InitialValueProblem(
			GuidingCenterDynamics(potential, rho=0.05),
			source,
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.04),
			max_step=0.01,
			sample_count=5,
		)
		plain = simulate(
			problem,
			BM4Composition(GCExtendedFormulation()),
			request,
		)
		tracked = simulate(
			problem,
			BM4Composition(GCExtendedFormulation(), track_energy=True),
			request,
		)

		np.testing.assert_array_equal(plain.states, tracked.states)
		assert tracked.k is not None
		energy = problem.dynamics.hamiltonian(tracked.t, tracked.states) + tracked.k
		expected_error = float(
			np.max(np.abs(energy - np.asarray(energy)[:, :1]))
		)
		self.assertEqual(tracked.err, expected_error)

	def test_runner_rejects_invalid_method_output(self) -> None:
		source = GCInitialConfiguration(np.asarray([1.0, 0.0]))
		problem = InitialValueProblem(_RotationDynamics(), source)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.1,
			sample_count=2,
		)
		invalid_histories = (
			np.zeros((4, 2)),
			np.asarray([[1.0, np.nan], [0.0, 0.0]]),
			np.asarray([[2.0, 1.0], [0.0, 0.0]]),
		)
		for states in invalid_histories:
			with self.subTest(states=states):
				with self.assertRaises(ValueError):
					simulate(
						problem,
						_FixedOutputMethod(states),
						request,
					)

	def test_request_revalidates_normalized_endpoints(self) -> None:
		with self.assertRaises(ValueError):
			SimulationRequest(
				t_span=(0.0, 1.0),
				max_step=0.1,
				output_times=np.asarray([0.0, 1.0 + 1e-15, 1.0 + 2e-15]),
			)

	def test_solution_is_immutable_and_preserves_read_only_aliases(self) -> None:
		source = GCInitialConfiguration(np.asarray([1.0, 0.0]))
		times = np.asarray([0.0, 0.1])
		states = np.asarray([[1.0, 0.9], [0.0, -0.1]])
		momentum = np.asarray([[0.0, 0.01]])
		solution = Solution(
			t=times,
			states=states,
			source=source,
			diagnostics={
				"step_count": 1,
				"extended_momentum": momentum,
				"energy_error": 0.02,
			},
		)

		self.assertIs(solution.y, solution.states)
		self.assertIs(solution.trajectory, source)
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
