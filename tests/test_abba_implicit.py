"""Contracts for the reduced and simultaneous implicit ABBA formulations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnostics.symplecticity import (
	central_difference_jacobian,
)
from initial_conditions import TrajectoryGC
from simulation import (
	ABBA_PROJECTION_FORMULATIONS,
	ABBA_STATE_EXTENSIONS,
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	ABBA6Implicit,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods.abba._projection_common import (
	_differentiate_stages,
	_evaluate_displaced_stages,
)
from simulation.methods.abba._projection_reduced import (
	_solve_reduced_multiplier_step,
)
from simulation.methods.abba._projection_simultaneous import (
	_simultaneous_newton_jacobian,
	_simultaneous_residual_blocks,
	_solve_simultaneous_state_multiplier_step,
)
from studies import (
	ImplicitABBAObserverConfig,
	ImplicitABBASymplecticityComparison,
	ImplicitABBASymplecticityConfig,
	RandomPotentialConfig,
	centered_square,
	pi_area_steps,
	run_implicit_abba_symplecticity_study,
)

from tests.test_abba import gc_dynamics


class ImplicitABBAFormulationTests(unittest.TestCase):
	"""Verify equation (21) and its equivalence to the reduced solve."""

	def test_all_four_public_implicit_methods_expose_the_three_axes(self) -> None:
		for method_type in (
			ABBA2Implicit,
			ABBA4Implicit,
			ABBA4ImplicitSingleProjection,
			ABBA6Implicit,
		):
			for formulation in ABBA_PROJECTION_FORMULATIONS:
				for extension in ABBA_STATE_EXTENSIONS:
					with self.subTest(
						method=method_type.__name__,
						formulation=formulation,
						extension=extension,
					):
						method = method_type(
							projection_formulation=formulation,
							nonlinear_solver="broyden",
							state_extension=extension,
						)
						self.assertEqual(method.projection_formulation, formulation)
						self.assertEqual(method.nonlinear_solver, "broyden")
						self.assertEqual(method.state_extension, extension)
			with self.assertRaisesRegex(ValueError, "projection_formulation"):
				method_type(projection_formulation="unknown")  # type: ignore[arg-type]

	def test_simultaneous_jacobian_matches_centered_differences(self) -> None:
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.2])
		time = 0.2
		step = 0.05
		candidate = np.asarray([1.01, 1.19, 0.99, 1.21, 1e-3, -2e-3])

		def residual(unknown: np.ndarray) -> np.ndarray:
			first_output = unknown[:2]
			second_output = unknown[2:4]
			multiplier = unknown[4:]
			stages = _evaluate_displaced_stages(
				dynamics,
				time,
				state,
				step,
				multiplier,
			)
			return _simultaneous_residual_blocks(
				stages,
				multiplier,
				first_output,
				second_output,
				dynamics.state_dimension,
			)[0]

		stages = _evaluate_displaced_stages(
			dynamics,
			time,
			state,
			step,
			candidate[4:],
		)
		evaluation = _differentiate_stages(
			dynamics,
			time,
			state,
			step,
			stages,
		)
		analytic = _simultaneous_newton_jacobian(evaluation)[0]
		numerical = central_difference_jacobian(
			residual,
			candidate,
			relative_step=1e-5,
		)
		np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-8)

	def test_both_roots_are_equivalent_and_reversible(self) -> None:
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.2])
		time = 0.3
		step = 0.07
		solver = {
			"absolute_tolerance": 1e-14,
			"relative_tolerance": 1e-14,
			"max_iterations": 12,
		}
		first = _solve_reduced_multiplier_step(dynamics, time, state, step, **solver)
		second = _solve_simultaneous_state_multiplier_step(
			dynamics,
			time,
			state,
			step,
			**solver,
		)
		np.testing.assert_allclose(second.state, first.state, rtol=0.0, atol=2e-15)
		np.testing.assert_allclose(
			second.multiplier,
			first.multiplier,
			rtol=0.0,
			atol=2e-15,
		)

		backward = _solve_simultaneous_state_multiplier_step(
			dynamics,
			time + step,
			second.state,
			-step,
			**solver,
		)
		np.testing.assert_allclose(backward.state, state, rtol=0.0, atol=2e-14)

	def test_public_method_reports_and_matches_both_projection_formulations(
		self,
	) -> None:
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.4, 1.2, 1.6])
		problem = InitialValueProblem(
			dynamics,
			TrajectoryGC(state, rho=0.05),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.2),
			max_step=0.05,
			sample_count=5,
		)
		solutions = {
			formulation: simulate(
				problem,
				ABBA2Implicit(projection_formulation=formulation),
				request,
			)
			for formulation in ABBA_PROJECTION_FORMULATIONS
		}
		for formulation, solution in solutions.items():
			self.assertEqual(
				solution.diagnostics["projection_formulation"],
				formulation,
			)
			self.assertEqual(solution.diagnostics["state_extension"], "physical")
			self.assertEqual(solution.diagnostics["newton_iterations"].shape, (4,))
		np.testing.assert_allclose(
			solutions["simultaneous_state_multiplier"].states,
			solutions["reduced_multiplier"].states,
			rtol=0.0,
			atol=5e-15,
		)


class ImplicitABBASymplecticityStudyTests(unittest.TestCase):
	"""Verify the reusable paired study used by the development notebook."""

	def test_short_comparison_records_both_formulations(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		).build()
		area = centered_square(
			potential,
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		config = ImplicitABBASymplecticityConfig(
			steps=pi_area_steps(400, 800),
			t_span=(0.0, np.pi / 100),
			save_interval=np.pi / 100,
			chunk_size=2,
			observers=(
				ImplicitABBAObserverConfig(
					label="simultaneous",
					formulation="simultaneous_state_multiplier",
					jacobian_method="finite_difference",
				),
				ImplicitABBAObserverConfig(
					label="reduced",
					formulation="reduced_multiplier",
					jacobian_method="finite_difference",
				),
			),
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			comparison = run_implicit_abba_symplecticity_study(
				potential,
				area,
				notebook_path=(
					root
					/ "notebooks"
					/ "developements"
					/ "implicit_abba.ipynb"
				),
				config=config,
				project_root=root,
			)

		self.assertEqual(tuple(comparison.results), ("simultaneous", "reduced"))
		self.assertEqual(
			{
				label: result.method_name
				for label, result in comparison.results.items()
			},
			{
				"simultaneous": "ABBA2Implicit[simultaneous_state_multiplier]",
				"reduced": "ABBA2Implicit[reduced_multiplier]",
			},
		)
		with self.assertRaisesRegex(TypeError, "does not match formulation"):
			ImplicitABBASymplecticityComparison(
				observers=config.observers,
				results={
					"simultaneous": comparison.results["reduced"],
					"reduced": comparison.results["simultaneous"],
				},
			)
		difference = comparison.maximum_state_differences()[config.steps[0].label]
		self.assertLess(difference, 5e-15)
		for result in comparison.results.values():
			summary = result.summaries()[0]
			self.assertLess(summary.max_local_defect, 1e-9)
			self.assertIsNotNone(summary.max_newton_residual_norm)


if __name__ == "__main__":
	unittest.main()
