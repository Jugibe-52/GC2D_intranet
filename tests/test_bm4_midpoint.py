"""Numerical accuracy and runtime contracts for arithmetic-projected BM4."""

import unittest

import numpy as np
from scipy.integrate import solve_ivp

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import BM4Midpoint, InitialValueProblem, SimulationRequest, simulate
from simulation.formulations import GCExtendedFormulation


class _Rotation:
	"""Autonomous planar oscillator with an exact flow, also valid in batches."""

	state_dimension = 2

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		p = state.size // 2
		return np.concatenate((-state[p:], state[:p]))


def _gc_problem(state: np.ndarray | None = None) -> InitialValueProblem:
	"""Return a reproducible smooth time-dependent GC problem."""
	dynamics = GuidingCenterDynamics(Potential.random(
		A=0.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5,
	), rho=0.05)
	return InitialValueProblem(dynamics, GCInitialConfiguration(
		np.asarray([1.0, 1.2]) if state is None else state,
	))


class BM4MidpointTests(unittest.TestCase):
	"""Check projection placement, order, energy, vectorization and observers."""

	def test_one_step_matches_independent_twelve_stage_replay(self) -> None:
		problem = _gc_problem()
		prepared = GCExtendedFormulation(0.7).prepare(problem, track_energy=False)
		# Independent coefficient/order traversal catches changes to the shared
		# cycle, time convention, and accidental projection between stages.
		half = [0.0792036964311957, 0.1303114101821663, 0.2228614958676077,
			-0.3667132690474257, 0.3246481886897062, 0.1096884778767498]
		value = np.tile(problem.initial_state, 2)
		time = 0.3
		for index, coefficient in enumerate(half + half[::-1]):
			duration = coefficient * 0.2
			if index % 2:
				value = prepared.direct_map(duration, time + duration, value)
			else:
				value = prepared.adjoint_map(duration, time, value)
			time += duration
		solution = simulate(problem, BM4Midpoint(coupling_frequency=0.7),
			SimulationRequest.uniform(t_span=(0.3, 0.5), max_step=0.2, sample_count=2))
		np.testing.assert_allclose(solution.states[:, -1], (value[:2] + value[2:]) / 2, atol=1e-15)
		np.testing.assert_allclose(solution.diagnostics["copy_separation_norms"],
			[np.linalg.norm(value[:2] - value[2:], ord=np.inf)])
		self.assertEqual(solution.diagnostics["vector_field_evaluations_per_step"], 24)
		self.assertEqual(solution.diagnostics["nonlinear_unknown_dimension"], 0)

	def test_fourth_order_against_exact_rotation(self) -> None:
		problem = InitialValueProblem(_Rotation(), GCInitialConfiguration(np.asarray([1., 0.])))
		errors = []
		for step in (0.2, 0.1):
			solution = simulate(problem, BM4Midpoint(), SimulationRequest.uniform(
				t_span=(0., 1.), max_step=step, sample_count=2))
			errors.append(np.linalg.norm(solution.states[:, -1] - [np.cos(1.), np.sin(1.)]))
		self.assertGreater(errors[0] / errors[1], 14.)
		self.assertLess(errors[0] / errors[1], 18.)

	def test_nonautonomous_order_for_both_extensions(self) -> None:
		problem = _gc_problem()
		reference = solve_ivp(problem.dynamics.vector_field, (0., 0.4), problem.initial_state,
			method="DOP853", atol=1e-14, rtol=1e-13, max_step=0.005).y[:, -1]
		for extension in ("physical", "fully_extended"):
			with self.subTest(extension=extension):
				errors = []
				for step in (0.1, 0.05):
					solution = simulate(problem, BM4Midpoint(state_extension=extension),
						SimulationRequest.uniform(t_span=(0., 0.4), max_step=step, sample_count=2))
					errors.append(np.linalg.norm(solution.states[:, -1] - reference))
				self.assertGreater(errors[0] / errors[1], 13.)
				self.assertLess(errors[0] / errors[1], 19.)

	def test_energy_tracking_and_particle_batching(self) -> None:
		problem = _gc_problem(np.asarray([1., 1.1, 1.2, 0.9]))
		request = SimulationRequest.uniform(t_span=(0.2, 0.4), max_step=0.05, sample_count=5)
		plain = simulate(problem, BM4Midpoint(), request)
		tracked = simulate(problem, BM4Midpoint(track_energy=True), request)
		np.testing.assert_array_equal(plain.states, tracked.states)
		for particle in range(2):
			indices = [particle, particle + 2]
			scalar = simulate(InitialValueProblem(problem.dynamics,
				GCInitialConfiguration(problem.initial_state[indices])), BM4Midpoint(track_energy=True), request)
			np.testing.assert_allclose(tracked.states[indices], scalar.states, atol=2e-15)
			np.testing.assert_allclose(tracked.diagnostics["extended_momentum"][particle],
				scalar.diagnostics["extended_momentum"][0], atol=2e-15)
		# Independent augmented ODE checks the factor-of-two momentum convention.
		initial = np.concatenate((problem.initial_state, np.zeros(2)))
		def rhs(t: float, state: np.ndarray) -> np.ndarray:
			return np.concatenate((problem.dynamics.vector_field(t, state[:4]),
				problem.dynamics.extended_momentum_derivative(t, state[:4])))
		reference = solve_ivp(rhs, request.t_span, initial, method="DOP853", rtol=1e-12, atol=1e-14)
		np.testing.assert_allclose(tracked.diagnostics["extended_momentum"][:, -1], reference.y[4:, -1], atol=1e-7)

	def test_shadow_samples_and_retained_observer_maps(self) -> None:
		problem = _gc_problem()
		for extension, tracking in (("physical", False), ("physical", True), ("fully_extended", True)):
			with self.subTest(extension=extension, tracking=tracking):
				events = []
				method = BM4Midpoint(extension, step_observer=events.append, track_energy=tracking)
				dense = simulate(problem, method, SimulationRequest.uniform(
					t_span=(0.3, 0.5), max_step=0.1, sample_count=8))
				self.assertEqual(len(events), 2)
				self.assertEqual(len(dense.diagnostics["copy_separation_norms"]), 2)
				for event in events:
					np.testing.assert_array_equal(event.map_state(event.state_before), event.state_after)
					self.assertEqual(event.state_before.size, 4 if extension == "fully_extended" else 2)
				sparse = simulate(problem, BM4Midpoint(extension, track_energy=tracking),
					SimulationRequest.uniform(t_span=(0.3, 0.5), max_step=0.1, sample_count=2))
				np.testing.assert_array_equal(dense.states[:, -1], sparse.states[:, -1])
				if extension == "fully_extended":
					np.testing.assert_array_equal(dense.diagnostics["extended_time"], dense.t)
					self.assertEqual(dense.diagnostics["extended_momentum_normalization"], "direct_k")

	def test_invalid_configuration_and_energy_capability(self) -> None:
		for frequency in (-1., np.nan, np.inf):
			with self.assertRaises(ValueError):
				BM4Midpoint(coupling_frequency=frequency)
		with self.assertRaises(ValueError):
			BM4Midpoint(state_extension="invalid")
		with self.assertRaises(TypeError):
			BM4Midpoint(nonlinear_solver="newton")
		self.assertTrue(BM4Midpoint("fully_extended").track_energy)
		request = SimulationRequest.uniform(t_span=(0., 0.1), max_step=0.1, sample_count=2)
		with self.assertRaises(TypeError):
			simulate(InitialValueProblem(_Rotation(), GCInitialConfiguration(np.asarray([1., 0.]))),
				BM4Midpoint(track_energy=True), request)
		with self.assertRaises(ValueError):
			simulate(_gc_problem(np.asarray([1., 1.1, 1.2, 0.9])), BM4Midpoint("fully_extended"), request)


if __name__ == "__main__":
	unittest.main()
