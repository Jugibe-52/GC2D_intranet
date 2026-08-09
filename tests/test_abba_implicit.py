"""Contracts for the reduced and simultaneous implicit ABBA formulations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnostics.symplecticity import (
	central_difference_jacobian,
	gc_physical_symplectic_form,
)
from initial_conditions import TrajectoryGC
from simulation import (
	ImplicitABBA1,
	ImplicitABBA2,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods.abba import (
	_differentiate_stages,
	_evaluate_stages,
	_simultaneous_newton_jacobian,
	_simultaneous_residual_blocks,
	_solve_projected_step,
	_solve_simultaneous_projected_step,
)
from studies import (
	ImplicitABBASymplecticityConfig,
	RandomPotentialConfig,
	centered_square,
	pi_area_steps,
	run_implicit_abba_symplecticity_study,
)

from tests.test_abba import gc_dynamics


class ImplicitABBAFormulationTests(unittest.TestCase):
	"""Verify equation (21) and its equivalence to the reduced solve."""

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
			stages = _evaluate_stages(
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

		stages = _evaluate_stages(
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

	def test_both_roots_are_equivalent_reversible_and_symplectic(self) -> None:
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.2])
		time = 0.3
		step = 0.07
		solver = {
			"absolute_tolerance": 1e-14,
			"relative_tolerance": 1e-14,
			"max_iterations": 12,
		}
		first = _solve_projected_step(dynamics, time, state, step, **solver)
		second = _solve_simultaneous_projected_step(
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

		backward = _solve_simultaneous_projected_step(
			dynamics,
			time + step,
			second.state,
			-step,
			**solver,
		)
		np.testing.assert_allclose(backward.state, state, rtol=0.0, atol=2e-14)
		assert second.ideal_state_jacobian is not None
		form = gc_physical_symplectic_form(1)
		defect = (
			second.ideal_state_jacobian.T
			@ form
			@ second.ideal_state_jacobian
			- form
		)
		self.assertLess(float(np.linalg.norm(defect, ord="fro")), 1e-12)

	def test_public_methods_report_distinct_solver_formulations(self) -> None:
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
		first = simulate(problem, ImplicitABBA1(), request)
		second = simulate(problem, ImplicitABBA2(), request)
		np.testing.assert_allclose(second.states, first.states, rtol=0.0, atol=5e-15)
		self.assertEqual(
			first.diagnostics["projection_solver_formulation"],
			"implicit_1_reduced_equation_11",
		)
		self.assertEqual(
			second.diagnostics["projection_solver_formulation"],
			"implicit_2_simultaneous_equation_21",
		)
		self.assertEqual(
			second.diagnostics["newton_iterations"].shape,
			(4,),
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

		self.assertEqual(tuple(comparison.results), ("implicit_1", "implicit_2"))
		difference = comparison.maximum_state_differences()[config.steps[0].label]
		self.assertLess(difference, 5e-15)
		for result in comparison.results.values():
			summary = result.summaries()[0]
			self.assertLess(summary.max_local_defect, 1e-9)
			self.assertIsNotNone(summary.max_newton_residual_norm)


if __name__ == "__main__":
	unittest.main()
