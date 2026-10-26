"""Preserve input policies and numerical work when validation is shared."""

from __future__ import annotations

from dynamics.gc import GuidingCenterDynamics

from dataclasses import dataclass, field, replace
from typing import Any
import unittest

import numpy as np

from problem import InitialValueProblem

from request import SimulationRequest

from solvers.nonlinear import _solve_broyden, _solve_newton, NewtonSolver
from methods.adaptive.scipy import DOP853, Radau
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.rk4 import RK4
from methods.classical.sdirk import SDIRK4
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from methods.hbvm.order4 import HBVM42
from states.history import States


class _CountingDynamics:
	"""Harmonic field without analytic Jacobian capabilities."""

	state_dimension = 2

	def __init__(self) -> None:
		self.evaluations = 0

	def vector_field(self, time: float, state: np.ndarray) -> np.ndarray:
		self.evaluations += 1
		return np.asarray((state[1], -state[0]))


@dataclass(slots=True)
class _ArrayMetadataRK4(RK4):
	"""Expose a caller-owned metadata array through ordinary run preparation."""

	probe: np.ndarray = field(default_factory=lambda: np.asarray((1.0, 2.0)))

	def initialize(self, request: SimulationRequest) -> None:
		RK4.initialize(self, request)
		self.metadata = {"probe": self.probe}


class MethodValidationContractTests(unittest.TestCase):
	"""Verify behavior that generic numeric validators must not silently change."""

	def test_algebraic_and_adaptive_tolerances_have_explicit_owners(self) -> None:
		for value in (True, np.bool_(True)):
			with self.assertRaises(ValueError):
				NewtonSolver(absolute_tolerance=value)
			for constructor in (DOP853, Radau):
				method = constructor(absolute_tolerance=value)
				self.assertEqual(method.absolute_tolerance, 1.)
				self.assertIs(type(method.absolute_tolerance), float)

	def test_removed_coupling_and_optional_first_step_controls(self) -> None:
		for constructor in (BM4Midpoint, BM4Implicit):
			for value in (0., .4, False, True, np.bool_(False), np.bool_(True)):
				with self.assertRaises(TypeError):
					constructor(coupling_frequency=value)
		for constructor in (DOP853, Radau):
			self.assertIsNone(constructor(first_step=None).first_step)
			self.assertEqual(constructor(first_step=True).first_step, 1.0)
			with self.assertRaisesRegex(ValueError, "first_step must be finite and positive"):
				constructor(first_step=False)

	def test_iteration_limits_are_validated_once_by_the_solver(self) -> None:
		solver = NewtonSolver(max_iterations=np.int64(3))
		self.assertIs(type(solver.max_iterations), int)
		for value in (True, np.bool_(True), 3.0, 0):
			with self.assertRaises(ValueError):
				NewtonSolver(max_iterations=value)
		for constructor in (GaussLegendre4, SDIRK4, HBVM42, ABBA2Implicit, BM4Implicit):
			self.assertIs(constructor(solver=solver).solver, solver)

	def test_invalid_controls_are_reported_by_their_owner(self) -> None:
		with self.assertRaisesRegex(ValueError, 'jacobian_relative_step'):
			NewtonSolver(jacobian_relative_step=0., max_iterations=0)
		for constructor in (DOP853, Radau):
			with self.assertRaisesRegex(ValueError, 'relative_tolerance must be finite and positive'):
				constructor(relative_tolerance=0., absolute_tolerance=0.)

	def test_request_snaps_owned_endpoints_and_rechecks_increasing_times(self) -> None:
		times = np.asarray((np.nextafter(0.0, -np.inf), 0.5, np.nextafter(1.0, np.inf)))
		original = times.copy()
		request = SimulationRequest((0.0, 1.0),  times)
		np.testing.assert_array_equal(request.output_times, (0.0, 0.5, 1.0))
		np.testing.assert_array_equal(times, original)
		self.assertFalse(request.output_times.flags.writeable)
		self.assertFalse(np.shares_memory(request.output_times, times))
		with self.assertRaisesRegex(ValueError, "must remain strictly increasing"):
			SimulationRequest((0.0, 1.0),  np.asarray((-np.finfo(float).eps, 0.0, 1.0)))
		with self.assertRaisesRegex(ValueError, "`t_span` must contain two finite, increasing times"):
			SimulationRequest((1.0, 0.0),  np.asarray((0.0, 1.0)))

	def test_solution_owns_readonly_history_and_diagnostics(self) -> None:
		times = np.asarray((0.0, 1.0))
		states = np.asarray(((0, 1), (1, 2)), dtype=np.int16)
		metric = np.asarray((1, 2), dtype=np.int64)
		solution = States(t=times, states=states.T, initial_state=np.array(states[:, 0], dtype=float, copy=True), diagnostics={"metric": metric}, layout=GuidingCenterDynamics.coordinates)
		times[0], states[0, 0], metric[0] = -1.0, -1, -1
		self.assertEqual(solution.t[0], 0.0)
		self.assertEqual(solution.states[0, 0], 0)
		self.assertEqual(solution.states.dtype, np.dtype(np.int16))
		stored = np.asarray(solution.diagnostics["metric"])
		self.assertEqual(stored[0], 1)
		for array in (solution.t, solution.states.T, stored):
			self.assertFalse(array.flags.writeable)
		with self.assertRaises(TypeError):
			solution.diagnostics["other"] = 1  # type: ignore[index]

	def test_run_owns_initial_state_and_metadata_arrays(self) -> None:
		problem = InitialValueProblem(_CountingDynamics(), np.array(np.asarray((1.0, 0.0)), dtype=float, copy=True), layout=GuidingCenterDynamics.coordinates)
		request = SimulationRequest.uniform(t_span=(0.0, 0.1), sample_count=2)
		method = _ArrayMetadataRK4()
		run = replace(method, step=0.1).new_run(problem.dynamics.vector_field, problem.initial_conditions, request)
		self.assertFalse(run.initial_state.flags.writeable)
		stored = np.asarray(run.metadata["probe"])
		self.assertFalse(stored.flags.writeable)
		self.assertFalse(np.shares_memory(stored, method.probe))
		method.probe[0] = 7.0
		self.assertEqual(stored[0], 1.0)
		self.assertEqual(run._status, "ready")

	def test_cached_residual_counts_once_without_evaluating_it_again(self) -> None:
		for solver in ("Newton", "Broyden"):
			for initial_residual in (0.0, -1.0):
				with self.subTest(solver=solver, residual=initial_residual):
					calls: list[np.ndarray] = []

					def residual(unknown: np.ndarray) -> tuple[np.ndarray, str]:
						calls.append(unknown.copy())
						return unknown - 1.0, "evaluated"

					initial = np.zeros(1)
					cached = (np.asarray((initial_residual,)), "cached")
					if solver == "Newton":
						result = _solve_newton(
							residual, initial, lambda unknown, value, payload: unknown - value,
							tolerance=1e-12, max_iterations=2, context="test", initial_evaluation=cached,
						)
					else:
						result = _solve_broyden(
							residual, initial, np.eye(1), tolerance=1e-12,
							max_iterations=2, context="test", initial_evaluation=cached,
						)
					expected_calls = int(initial_residual != 0.0)
					self.assertEqual(len(calls), expected_calls)
					self.assertEqual(result.residual_evaluations, 1 + expected_calls)
					self.assertEqual(result.iterations, expected_calls)
					np.testing.assert_array_equal(initial, (0.0,))

	def test_hbvm_capability_error_stays_after_initial_field_work(self) -> None:
		dynamics = _CountingDynamics()
		problem = InitialValueProblem(dynamics, np.array(np.asarray((1.0, 0.0)), dtype=float, copy=True), layout=GuidingCenterDynamics.coordinates)
		request = SimulationRequest.uniform(t_span=(0.0, 0.1), sample_count=2)
		run = HBVM42(step=.1, solver=NewtonSolver(jacobian_method="analytic", backtracking=True)).new_run(problem.dynamics.vector_field, problem.initial_conditions, request)
		self.assertEqual(dynamics.evaluations, 0)
		with self.assertRaisesRegex(TypeError, "Analytic differentiation requires a Jacobian function"):
			run.advance(0.0, run.initial_state, 0.1)
		self.assertEqual(dynamics.evaluations, 9)


if __name__ == "__main__":
	unittest.main()
