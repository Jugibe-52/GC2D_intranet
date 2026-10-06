"""Preserve input policies and numerical work when validation is shared."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import unittest

import numpy as np

from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from initial_conditions import GCInitialConfiguration
from methods._nonlinear import _solve_broyden, _solve_newton
from methods.adaptive.scipy import DOP853, Radau
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.rk4 import RK4
from methods.classical.sdirk import SDIRK4
from methods.extended.abba import ABBA2Implicit, ABBA4Implicit, ABBA6Implicit
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from methods.hbvm.order4 import HBVM42
from solution import Solution


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

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		RK4.initialize(self, problem, request)
		self.metadata = {"probe": self.probe}


class MethodValidationContractTests(unittest.TestCase):
	"""Verify behavior that generic numeric validators must not silently change."""

	def test_boolean_tolerance_policy_remains_method_specific(self) -> None:
		permissive: tuple[tuple[type[Any], str], ...] = (
			(GaussLegendre4, "newton_absolute_tolerance"),
			(SDIRK4, "newton_absolute_tolerance"),
			(HBVM42, "absolute_tolerance"),
			(DOP853, "absolute_tolerance"),
			(Radau, "absolute_tolerance"),
		)
		for constructor, name in permissive:
			for value in (True, np.bool_(True)):
				with self.subTest(method=constructor.__name__, value=type(value)):
					method = constructor(**{name: value})
					self.assertEqual(getattr(method, name), 1.0)
					self.assertIs(type(getattr(method, name)), float)
		for constructor in (ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Implicit):
			for value in (True, np.bool_(True)):
				with self.subTest(method=constructor.__name__, value=type(value)):
					with self.assertRaisesRegex(ValueError, "`newton_absolute_tolerance` must be positive and finite"):
						constructor(newton_absolute_tolerance=value)

	def test_boolean_coupling_and_optional_first_step_policies_remain_distinct(self) -> None:
		for value in (False, True, np.bool_(False), np.bool_(True)):
			self.assertEqual(BM4Midpoint(coupling_frequency=value).coupling_frequency, float(value))
			with self.assertRaisesRegex(ValueError, "`coupling_frequency` must be non-negative and finite"):
				BM4Implicit(coupling_frequency=value)
		for constructor in (DOP853, Radau):
			self.assertIsNone(constructor(first_step=None).first_step)
			self.assertEqual(constructor(first_step=True).first_step, 1.0)
			with self.assertRaisesRegex(ValueError, "first_step must be finite and positive"):
				constructor(first_step=False)

	def test_iteration_limits_keep_integer_normalization_and_reject_booleans(self) -> None:
		constructors: tuple[tuple[type[Any], str], ...] = (
			(GaussLegendre4, "newton_max_iterations"), (SDIRK4, "newton_max_iterations"),
			(HBVM42, "max_iterations"), (ABBA2Implicit, "newton_max_iterations"),
			(BM4Implicit, "newton_max_iterations"),
		)
		for constructor, name in constructors:
			with self.subTest(method=constructor.__name__):
				self.assertIs(type(getattr(constructor(**{name: np.int64(3)}), name)), int)
				for value in (True, np.bool_(True), 3.0, 0):
					with self.assertRaisesRegex(ValueError, f"`{name}` must be a positive integer"):
						constructor(**{name: value})

	def test_invalid_controls_keep_the_original_first_error(self) -> None:
		for constructor in (GaussLegendre4, SDIRK4):
			with self.assertRaisesRegex(ValueError, "`newton_jacobian_relative_step` must be positive and finite"):
				constructor(newton_jacobian_relative_step=0.0, newton_max_iterations=0)
		with self.assertRaisesRegex(ValueError, "`max_iterations` must be a positive integer"):
			HBVM42(max_iterations=0, jacobian_relative_step=0.0)
		for constructor in (DOP853, Radau):
			with self.assertRaisesRegex(ValueError, "relative_tolerance must be finite and positive"):
				constructor(relative_tolerance=0.0, absolute_tolerance=0.0)

	def test_request_snaps_owned_endpoints_and_rechecks_increasing_times(self) -> None:
		times = np.asarray((np.nextafter(0.0, -np.inf), 0.5, np.nextafter(1.0, np.inf)))
		original = times.copy()
		request = SimulationRequest((0.0, 1.0), 0.1, times)
		np.testing.assert_array_equal(request.output_times, (0.0, 0.5, 1.0))
		np.testing.assert_array_equal(times, original)
		self.assertFalse(request.output_times.flags.writeable)
		self.assertFalse(np.shares_memory(request.output_times, times))
		with self.assertRaisesRegex(ValueError, "must remain strictly increasing"):
			SimulationRequest((0.0, 1.0), 0.1, np.asarray((-np.finfo(float).eps, 0.0, 1.0)))
		with self.assertRaisesRegex(ValueError, "`t_span` must contain two finite, increasing times"):
			SimulationRequest((1.0, 0.0), False, np.asarray((0.0, 1.0)))

	def test_solution_owns_readonly_history_and_diagnostics(self) -> None:
		times = np.asarray((0.0, 1.0))
		states = np.asarray(((0, 1), (1, 2)), dtype=np.int16)
		metric = np.asarray((1, 2), dtype=np.int64)
		solution = Solution(
			t=times, states=states, source=GCInitialConfiguration(states[:, 0]),
			diagnostics={"metric": metric},
		)
		times[0], states[0, 0], metric[0] = -1.0, -1, -1
		self.assertEqual(solution.t[0], 0.0)
		self.assertEqual(solution.states[0, 0], 0)
		self.assertEqual(solution.states.dtype, np.dtype(np.int16))
		stored = np.asarray(solution.diagnostics["metric"])
		self.assertEqual(stored[0], 1)
		for array in (solution.t, solution.states, stored):
			self.assertFalse(array.flags.writeable)
		with self.assertRaises(TypeError):
			solution.diagnostics["other"] = 1  # type: ignore[index]

	def test_run_owns_initial_state_and_metadata_arrays(self) -> None:
		problem = InitialValueProblem(_CountingDynamics(), GCInitialConfiguration(np.asarray((1.0, 0.0))))
		request = SimulationRequest.uniform(t_span=(0.0, 0.1), sample_count=2)
		method = _ArrayMetadataRK4()
		run = method.new_run(problem, request)
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
		problem = InitialValueProblem(dynamics, GCInitialConfiguration(np.asarray((1.0, 0.0))))
		request = SimulationRequest.uniform(t_span=(0.0, 0.1), sample_count=2)
		run = HBVM42(jacobian_method="analytic").new_run(problem, request)
		self.assertEqual(dynamics.evaluations, 0)
		with self.assertRaisesRegex(TypeError, "Analytic HBVM Jacobians require GuidingCenterJacobianSystem dynamics"):
			run.advance(0.0, run.initial_state, 0.1)
		self.assertEqual(dynamics.evaluations, 9)


if __name__ == "__main__":
	unittest.main()
