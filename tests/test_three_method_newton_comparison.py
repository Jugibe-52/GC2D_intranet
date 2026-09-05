"""Contracts for the three-method, three-trajectory Newton comparison."""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from studies import (
	THREE_METHOD_NEWTON_METHODS,
	ThreeMethodNewtonComparisonConfig,
	RandomPotentialConfig,
	latin_hypercube_gc_configuration,
	run_three_method_newton_comparison,
)
from visualization import (
	animate_implicit_method_trajectories,
	plot_accuracy_runtime_tradeoff,
	plot_accuracy_summary,
	plot_energy_accuracy_over_time,
	plot_implicit_method_iterations,
	plot_trajectory_accuracy_over_time,
)


class ThreeMethodNewtonComparisonTests(unittest.TestCase):
	"""Verify aligned accuracy, timings, Newton work, and notebook plots."""

	def test_three_trajectory_result_and_visualization_contracts(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.2,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		).build()
		initial_configuration = latin_hypercube_gc_configuration(
			potential,
			particle_count=3,
			seed=20260831,
			domain_margin_fraction=0.35,
		)
		config = ThreeMethodNewtonComparisonConfig(
			rho=0.3,
			t_span=(0.0, 0.2),
			integration_step=0.1,
			save_interval=0.1,
			absolute_tolerance=1e-13,
			relative_tolerance=1e-12,
			max_iterations=20,
			reference_relative_tolerance=1e-12,
			reference_absolute_tolerance=1e-14,
			reference_maximum_step=0.02,
			audit_relative_tolerance=1e-12,
			audit_absolute_tolerance=1e-14,
			audit_maximum_step=0.01,
			timing_warmups=0,
			timing_repeats=1,
			distance_convention="euclidean",
		)
		result = run_three_method_newton_comparison(
			potential,
			initial_configuration,
			config=config,
		)

		expected_methods = (
			"ABBA4ImplicitSingleProjection",
			"GaussLegendre4",
			"BM4Implicit1",
		)
		self.assertEqual(THREE_METHOD_NEWTON_METHODS, expected_methods)
		self.assertNotIn("HBVM42", THREE_METHOD_NEWTON_METHODS)
		self.assertEqual(tuple(result.solutions), expected_methods)
		self.assertEqual(tuple(result.accuracy), expected_methods)
		self.assertEqual(tuple(result.energy_accuracy), expected_methods)
		self.assertEqual(tuple(result.runtime_samples), expected_methods)
		initial_state = result.initial_configuration.initial_state
		assert initial_state is not None
		self.assertEqual(
			result.initial_configuration.layout.particle_count(initial_state),
			3,
		)
		self.assertEqual(result.reference.times.size, 3)
		self.assertEqual(result.reference_energies.shape, (3, 3))
		self.assertEqual(result.audit_energies.shape, (3, 3))
		self.assertFalse(result.reference_energies.flags.writeable)
		self.assertFalse(result.audit_energies.flags.writeable)
		self.assertGreater(result.reference_energy_scale, 0.0)
		self.assertGreaterEqual(
			result.energy_reference_time_integrated_rms_floor,
			0.0,
		)
		self.assertGreater(result.total_method_runtime_seconds, 0.0)
		self.assertGreater(result.total_study_runtime_seconds, 0.0)
		for method_name in THREE_METHOD_NEWTON_METHODS:
			solution = result.solutions[method_name]
			self.assertEqual(solution.states.shape, (6, 3))
			self.assertEqual(solution.diagnostics["step_count"], 2)
			self.assertEqual(
				np.asarray(solution.diagnostics["nonlinear_iterations"]).shape,
				(2,),
			)
			self.assertEqual(result.runtime_samples[method_name].shape, (1,))
			self.assertFalse(result.runtime_samples[method_name].flags.writeable)
			delta = solution.states - result.reference.states
			expected_distances = np.hypot(delta[:3], delta[3:])
			np.testing.assert_allclose(
				result.accuracy[method_name].distances,
				expected_distances,
				rtol=0.0,
				atol=0.0,
			)
			expected_energies = np.asarray(
				result.dynamics.hamiltonian(result.reference.times, solution.states),
				dtype=float,
			)
			expected_energy_errors = expected_energies - result.reference_energies
			energy_series = result.energy_accuracy[method_name]
			np.testing.assert_allclose(
				energy_series.errors,
				expected_energy_errors,
				rtol=0.0,
				atol=0.0,
			)
			self.assertEqual(energy_series.errors.shape, (3, 3))
			self.assertFalse(energy_series.errors.flags.writeable)
			np.testing.assert_array_equal(energy_series.errors[:, 0], 0.0)
			self.assertTrue(np.all(energy_series.rms_error >= 0.0))
			self.assertTrue(
				np.all(np.diff(energy_series.running_maximum_absolute_error) >= 0.0)
			)

		summaries = result.summaries()
		self.assertEqual(len(summaries), 3)
		self.assertEqual(
			tuple(row.method_name for row in summaries),
			expected_methods,
		)
		self.assertTrue(all(row.nonlinear_solver == "Newton" for row in summaries))
		self.assertTrue(all(row.trajectory_count == 3 for row in summaries))
		self.assertTrue(all(row.step_count == 2 for row in summaries))
		self.assertTrue(all(row.runtime_seconds > 0.0 for row in summaries))
		self.assertTrue(all(row.total_newton_iterations >= 0 for row in summaries))
		self.assertTrue(all(row.maximum_residual_to_tolerance <= 1.0 for row in summaries))
		self.assertTrue(all(row.time_integrated_rms_energy_error >= 0.0 for row in summaries))
		self.assertTrue(
			all(row.relative_time_integrated_rms_energy_error >= 0.0 for row in summaries)
		)
		self.assertTrue(all(row.energy_reference_floor_ratio >= 0.0 for row in summaries))

		accuracy_figure, accuracy_axis = plot_accuracy_summary(summaries)
		tradeoff_figure, tradeoff_axis = plot_accuracy_runtime_tradeoff(summaries)
		time_figure, time_axes = plot_trajectory_accuracy_over_time(
			result.reference.times,
			result.accuracy,
			reference_floor=result.reference.time_integrated_rms_floor,
		)
		energy_figure, energy_axes = plot_energy_accuracy_over_time(
			result.reference.times,
			result.energy_accuracy,
			reference_energy_errors=result.reference_energy_errors,
		)
		work_figure, work_axes = plot_implicit_method_iterations(result.solutions)
		self.assertGreater(len(accuracy_axis.patches), 0)
		self.assertEqual(len(tradeoff_axis.collections), 3)
		self.assertEqual(time_axes.shape, (2,))
		self.assertEqual(energy_axes.shape, (2,))
		energy_figure.canvas.draw()
		self.assertEqual(work_axes.shape, (3,))
		animation = animate_implicit_method_trajectories(
			result.effective_potential,
			result.solutions,
			frames=2,
			interval=10,
			repeat=False,
		)
		self.assertEqual(len(animation._func(1)), 7)
		self.assertIn("Three implicit methods", animation._fig.axes[0].get_title())
		animation._draw_was_started = True
		for figure in (
			accuracy_figure,
			tradeoff_figure,
			time_figure,
			energy_figure,
			work_figure,
		):
			plt.close(figure)

if __name__ == "__main__":
	unittest.main()
