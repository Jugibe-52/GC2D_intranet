"""Broyden contracts shared by all four implicit projection methods."""

from __future__ import annotations

import unittest

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA_PROJECTION_FORMULATIONS,
	BM4Implicit1,
	BM4Implicit2,
	ABBA2Implicit,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)


def _problem(*, interpolation_order: int = 3) -> InitialValueProblem:
	"""Build a compact two-particle problem with deterministic field data."""
	potential = Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=interpolation_order,
	)
	configuration = GCInitialConfiguration.from_components(
		x=np.asarray([1.0, 1.4]),
		y=np.asarray([1.2, 1.6]),
	)
	return InitialValueProblem(
		GuidingCenterDynamics(potential, rho=0.05),
		configuration,
	)


class ImplicitBroydenTests(unittest.TestCase):
	"""Verify convergence, root equivalence, and residual-only ABBA operation."""

	def test_all_methods_select_broyden_and_match_newton_roots(self) -> None:
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.05,
			sample_count=3,
		)
		common = {
			"newton_absolute_tolerance": 1e-14,
			"newton_relative_tolerance": 1e-13,
			"newton_max_iterations": 30,
		}
		methods = [
			ABBA2Implicit(projection_formulation=formulation)
			for formulation in ABBA_PROJECTION_FORMULATIONS
		]
		methods.extend((BM4Implicit1(), BM4Implicit2()))
		for method in methods:
			method_kwargs = (
				{"projection_formulation": method.projection_formulation}
				if isinstance(method, ABBA2Implicit)
				else {}
			)
			with self.subTest(
				method=type(method).__name__,
				formulation=method_kwargs.get("projection_formulation"),
			):
				events = []
				newton = simulate(
					_problem(),
					type(method)(
						nonlinear_solver="newton",
						**method_kwargs,
						**common,
					),
					request,
				)
				broyden = simulate(
					_problem(),
					type(method)(
						nonlinear_solver="broyden",
						step_observer=events.append,
						**method_kwargs,
						**common,
					),
					request,
				)
				np.testing.assert_allclose(
					broyden.states,
					newton.states,
					rtol=0.0,
					atol=2e-13,
				)
				self.assertEqual(
					broyden.diagnostics["nonlinear_solver"], "broyden"
				)
				iterations = np.asarray(
					broyden.diagnostics["nonlinear_iterations"]
				)
				residual_evaluations = np.asarray(
					broyden.diagnostics["residual_evaluations"]
				)
				np.testing.assert_array_equal(
					residual_evaluations,
					iterations + 1,
				)
				self.assertEqual(iterations.shape, (2,))
				self.assertEqual(len(events), 2)
				self.assertTrue(
					all(
						event.nonlinear_solver == "broyden"
						and event.residual_evaluations == event.newton_iterations + 1
						for event in events
					)
				)

	def test_broyden_abba_does_not_require_spatial_hessians(self) -> None:
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.05),
			max_step=0.05,
			sample_count=2,
		)
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			with self.subTest(formulation=formulation):
				solution = simulate(
					_problem(interpolation_order=2),
					ABBA2Implicit(
						projection_formulation=formulation,
						nonlinear_solver="broyden",
						newton_max_iterations=30,
					),
					request,
				)
				self.assertEqual(solution.states.shape, (4, 2))
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			with self.subTest(newton_formulation=formulation):
				with self.assertRaisesRegex(ValueError, "interpolation_order"):
					simulate(
						_problem(interpolation_order=2),
						ABBA2Implicit(projection_formulation=formulation),
						request,
					)

	def test_unknown_solver_fails_during_method_configuration(self) -> None:
		with self.assertRaisesRegex(ValueError, "nonlinear_solver"):
			ABBA2Implicit(nonlinear_solver="unknown")  # type: ignore[arg-type]
		with self.assertRaisesRegex(ValueError, "nonlinear_solver"):
			BM4Implicit2(nonlinear_solver="unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
	unittest.main()
