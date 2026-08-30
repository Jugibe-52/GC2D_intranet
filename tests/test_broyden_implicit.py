"""Broyden contracts shared by the four canonical implicit ABBA methods."""

from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	ABBA6Implicit,
	ABBA_PROJECTION_FORMULATIONS,
	ABBA_STATE_EXTENSIONS,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)


_IMPLICIT_METHODS = (
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	ABBA6Implicit,
)


def _problem(
	*,
	interpolation_order: int = 3,
	particle_count: int = 1,
) -> InitialValueProblem:
	"""Build a compact deterministic GC problem."""
	potential = Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=interpolation_order,
	)
	configuration = GCInitialConfiguration.from_components(
		x=np.linspace(1.0, 1.4, particle_count),
		y=np.linspace(1.2, 1.6, particle_count),
	)
	return InitialValueProblem(
		GuidingCenterDynamics(potential, rho=0.05),
		configuration,
	)


def _request() -> SimulationRequest:
	"""Return one short accepted step."""
	return SimulationRequest.uniform(
		t_span=(0.0, 0.02),
		max_step=0.02,
		sample_count=2,
	)


class ImplicitBroydenTests(unittest.TestCase):
	"""Verify convergence, root equivalence, and residual-only operation."""

	def test_all_canonical_methods_select_broyden_and_match_newton_roots(
		self,
	) -> None:
		common = {
			"state_extension": "physical",
			"newton_absolute_tolerance": 1e-13,
			"newton_relative_tolerance": 1e-13,
			"newton_max_iterations": 40,
		}
		for method_type, formulation in product(
			_IMPLICIT_METHODS,
			ABBA_PROJECTION_FORMULATIONS,
		):
			with self.subTest(
				method=method_type.__name__,
				formulation=formulation,
			):
				events = []
				newton = simulate(
					_problem(particle_count=2),
					method_type(
						projection_formulation=formulation,
						nonlinear_solver="newton",
						**common,
					),
					_request(),
				)
				broyden = simulate(
					_problem(particle_count=2),
					method_type(
						projection_formulation=formulation,
						nonlinear_solver="broyden",
						step_observer=events.append,
						**common,
					),
					_request(),
				)
				np.testing.assert_allclose(
					broyden.states,
					newton.states,
					rtol=0.0,
					atol=5e-10,
				)
				self.assertEqual(
					broyden.diagnostics["nonlinear_solver"],
					"broyden",
				)
				iterations = np.asarray(
					broyden.diagnostics["nonlinear_iterations"]
				)
				residual_evaluations = np.asarray(
					broyden.diagnostics["residual_evaluations"]
				)
				solves_per_step = int(
					broyden.diagnostics["nonlinear_solves_per_step"]
				)
				np.testing.assert_array_equal(
					residual_evaluations,
					iterations + solves_per_step,
				)
				self.assertEqual(iterations.shape, (1,))
				self.assertEqual(len(events), 1)
				self.assertEqual(events[0].nonlinear_solver, "broyden")
				self.assertEqual(
					events[0].residual_evaluations,
					events[0].newton_iterations + solves_per_step,
				)

	def test_broyden_is_residual_only_for_all_methods_and_extensions(self) -> None:
		problem = _problem(interpolation_order=2)
		for method_type, formulation, extension in product(
			_IMPLICIT_METHODS,
			ABBA_PROJECTION_FORMULATIONS,
			ABBA_STATE_EXTENSIONS,
		):
			with self.subTest(
				method=method_type.__name__,
				formulation=formulation,
				extension=extension,
			):
				solution = simulate(
					problem,
					method_type(
						projection_formulation=formulation,
						nonlinear_solver="broyden",
						state_extension=extension,
						newton_max_iterations=50,
					),
					_request(),
				)
				self.assertEqual(solution.states.shape, (2, 2))

	def test_newton_abba2_requires_analytic_derivatives_for_every_extension(
		self,
	) -> None:
		problem = _problem(interpolation_order=2)
		for formulation, extension in product(
			ABBA_PROJECTION_FORMULATIONS,
			ABBA_STATE_EXTENSIONS,
		):
			with self.subTest(
				formulation=formulation,
				extension=extension,
			):
				with self.assertRaisesRegex(ValueError, "interpolation_order"):
					simulate(
						problem,
						ABBA2Implicit(
							projection_formulation=formulation,
							state_extension=extension,
						),
						_request(),
					)

	def test_fully_extended_broyden_observation_needs_no_analytic_hessian(
		self,
	) -> None:
		problem = _problem(interpolation_order=2)
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			events = []
			with self.subTest(formulation=formulation):
				solution = simulate(
					problem,
					ABBA2Implicit(
						projection_formulation=formulation,
						nonlinear_solver="broyden",
						state_extension="fully_extended",
						step_observer=events.append,
					),
					_request(),
				)
				self.assertEqual(len(events), 1)
				self.assertEqual(events[0].jacobian.shape, (4, 4))
				self.assertTrue(np.all(np.isfinite(events[0].jacobian)))
				self.assertEqual(
					solution.diagnostics["projection_jacobian"],
					"centered_difference_observer_fallback",
				)

	def test_unknown_solver_fails_for_all_four_methods(self) -> None:
		for method_type in _IMPLICIT_METHODS:
			with self.subTest(method=method_type.__name__):
				with self.assertRaisesRegex(ValueError, "nonlinear_solver"):
					method_type(nonlinear_solver="unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
	unittest.main()
