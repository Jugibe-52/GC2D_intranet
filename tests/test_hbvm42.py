"""Contracts for HBVM(4,2), its study composition, and plots."""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import simulation
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import HBVM42, InitialValueProblem, SimulationRequest, simulate
from simulation.methods.hbvm.order4 import (
	_HBVM42_NODES,
	_HBVM42_RUNGE_KUTTA_MATRIX,
	_HBVM42_WEIGHTS,
)
from studies import (
	HBVM42BM4ComparisonConfig,
	HBVM42EvaluationConfig,
	QuarticOscillatorDynamics,
	quartic_oscillator_configuration,
	run_hbvm42_bm4_comparison,
	run_hbvm42_evaluation,
)
from visualization import (
	plot_hbvm42_bm4_comparison,
	plot_hbvm42_energy_errors,
	plot_hbvm42_evaluation,
)


class _QuarticWithoutJacobian:
	"""Small Hamiltonian system that forces the finite-difference Newton path."""

	state_dimension = 2

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		del t
		return np.asarray((state[1], -state[0] ** 3))

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		del t
		return 0.5 * state[1:2] ** 2 + 0.25 * state[0:1] ** 4


class HBVM42MethodTests(unittest.TestCase):
	"""Verify coefficients, order, energy, diagnostics, and solver selection."""

	def setUp(self) -> None:
		self.configuration = quartic_oscillator_configuration(
			position=1.0,
			momentum=0.0,
		)
		self.dynamics = QuarticOscillatorDynamics(quartic_strength=1.0)
		self.problem = InitialValueProblem(self.dynamics, self.configuration)

	def test_public_export_and_rank_two_runge_kutta_coefficients(self) -> None:
		self.assertIs(simulation.HBVM42, HBVM42)
		self.assertEqual(_HBVM42_NODES.shape, (4,))
		self.assertEqual(_HBVM42_WEIGHTS.shape, (4,))
		self.assertEqual(_HBVM42_RUNGE_KUTTA_MATRIX.shape, (4, 4))
		self.assertEqual(np.linalg.matrix_rank(_HBVM42_RUNGE_KUTTA_MATRIX), 2)
		np.testing.assert_allclose(np.sum(_HBVM42_WEIGHTS), 1.0, atol=2e-15)
		np.testing.assert_allclose(
			_HBVM42_RUNGE_KUTTA_MATRIX @ np.ones(4),
			_HBVM42_NODES,
			rtol=0.0,
			atol=2e-15,
		)

	def test_fourth_order_accuracy_and_quartic_energy_preservation(self) -> None:
		reference = solve_ivp(
			self.dynamics.vector_field,
			(0.0, 2.0),
			self.problem.initial_state,
			method="DOP853",
			t_eval=np.asarray((2.0,)),
			rtol=1e-13,
			atol=1e-15,
			max_step=0.002,
		).y[:, -1]
		errors: list[float] = []
		for step in (0.2, 0.1):
			solution = simulate(
				self.problem,
				HBVM42(
					absolute_tolerance=1e-14,
					relative_tolerance=1e-13,
					jacobian_method="analytic",
					track_energy=True,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 2.0),
					max_step=step,
					sample_count=11,
				),
			)
			errors.append(float(np.linalg.norm(solution.states[:, -1] - reference)))
			self.assertLess(float(solution.diagnostics["energy_error"]), 2e-13)
			self.assertEqual(
				np.asarray(solution.diagnostics["nonlinear_iterations"]).shape,
				(solution.n_steps,),
			)
		self.assertGreater(errors[0] / errors[1], 14.0)
		self.assertLess(errors[0] / errors[1], 18.0)

	def test_auto_jacobian_falls_back_to_centered_finite_differences(self) -> None:
		problem = InitialValueProblem(_QuarticWithoutJacobian(), self.configuration)
		solution = simulate(
			problem,
			HBVM42(jacobian_method="auto", track_energy=True),
			SimulationRequest.uniform(
				t_span=(0.0, 0.2),
				max_step=0.1,
				sample_count=3,
			),
		)
		field_counts = np.asarray(
			solution.diagnostics["vector_field_evaluations_per_step"]
		)
		self.assertEqual(field_counts.shape, (2,))
		self.assertTrue(np.all(field_counts > 17))
		with self.assertRaisesRegex(TypeError, "Analytic HBVM Jacobians"):
			simulate(
				problem,
				HBVM42(jacobian_method="analytic"),
				SimulationRequest.uniform(
					t_span=(0.0, 0.1),
					max_step=0.1,
					sample_count=2,
				),
			)

	def test_exact_and_finite_difference_jacobians_agree_for_gc_dynamics(self) -> None:
		"""Exercise HBVM on the repository's nonautonomous guiding-center model."""
		problem = InitialValueProblem(
			GuidingCenterDynamics(
				Potential.random(
					A=0.08,
					M=3,
					nx=16,
					ny=16,
					seed=27,
					interpolation_order=5,
				),
				rho=0.05,
			),
			GCInitialConfiguration.from_components(
				x=np.asarray((1.0,)),
				y=np.asarray((1.2,)),
			),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.2),
			max_step=0.05,
			sample_count=5,
		)
		analytic = simulate(problem, HBVM42(jacobian_method="analytic"), request)
		finite_difference = simulate(
			problem,
			HBVM42(jacobian_method="finite_difference"),
			request,
		)
		np.testing.assert_allclose(
			analytic.states,
			finite_difference.states,
			rtol=0.0,
			atol=2e-13,
		)


class HBVM42StudyTests(unittest.TestCase):
	"""Keep compact study runs aligned and visualization-ready."""

	def test_individual_and_comparison_studies_produce_complete_rows(self) -> None:
		dynamics = QuarticOscillatorDynamics(quartic_strength=1.0)
		configuration = quartic_oscillator_configuration()
		evaluation = run_hbvm42_evaluation(
			dynamics,
			configuration,
			config=HBVM42EvaluationConfig(
				steps=(0.4, 0.2),
				t_span=(0.0, 0.8),
				save_interval=0.4,
				reference_maximum_step=0.01,
				runtime_warmups=0,
				runtime_repeats=1,
			),
		)
		self.assertEqual(len(evaluation.summaries()), 2)
		self.assertEqual(len(evaluation.convergence_orders()), 1)
		self.assertGreater(
			evaluation.convergence_orders()[0].trajectory_rms_order,
			3.5,
		)
		self.assertGreater(evaluation.summaries()[0].local_symplecticity_defect, 0.0)

		comparison = run_hbvm42_bm4_comparison(
			dynamics,
			configuration,
			config=HBVM42BM4ComparisonConfig(
				steps=(0.4, 0.2),
				t_span=(0.0, 0.8),
				reference_maximum_step=0.01,
				runtime_warmups=0,
				runtime_repeats=1,
			),
		)
		self.assertEqual(len(comparison.summaries()), 4)
		self.assertEqual(
			{row.method for row in comparison.summaries()},
			{"HBVM(4,2)", "BM4"},
		)

		figures = (
			plot_hbvm42_evaluation(
				evaluation.summaries(),
				evaluation.convergence_orders(),
			)[0],
			plot_hbvm42_energy_errors(evaluation.solutions)[0],
			plot_hbvm42_bm4_comparison(comparison.summaries())[0],
		)
		for figure in figures:
			self.assertGreater(len(figure.axes), 0)
			plt.close(figure)


if __name__ == "__main__":
	unittest.main()
