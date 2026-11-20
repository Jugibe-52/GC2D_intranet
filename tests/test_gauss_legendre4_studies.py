"""Fast contracts for Gauss4 evaluation and BM4 comparison studies."""

from __future__ import annotations

from diagnostics.studies.comparison import time_integrated_rms_floor

from dynamics.gc import GuidingCenterDynamics

from dataclasses import replace
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np


from potential import Grid, Potential
from studies import (
	GAUSS_BM4_METHODS,
	GaussBM4ComparisonConfig,
	GaussLegendre4EvaluationConfig,
	run_gauss_bm4_comparison,
	run_gauss_legendre4_evaluation,
)
from visualization import (
	plot_gauss_bm4_accuracy_runtime,
	plot_gauss_legendre4_energy,
	plot_gauss_legendre4_evaluation,
	plot_gauss_legendre4_observed_order,
	plot_gauss_legendre4_symplecticity,
)


def _problem_data() -> tuple[Potential, np.ndarray]:
	"""Return the compact deterministic field and one GC initial point."""
	return (
		Potential.random(
			A=0.08,
			M=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
			spatial_extension="periodic",
		),
		np.array(np.asarray((1.0, 1.2)), dtype=float, copy=True),
	)


class GaussLegendre4StudyTests(unittest.TestCase):
	"""Verify result alignment, diagnostics, timings, and plotting contracts."""

	def test_individual_evaluation_returns_all_requested_diagnostics(self) -> None:
		potential, initial = _problem_data()
		config = GaussLegendre4EvaluationConfig(
			integration_steps=(0.02, 0.01),
			t_span=(0.0, 0.04),
			save_interval=0.02,
			rho=0.05,
			timing_warmups=0,
			timing_repeats=1,
			symplecticity_audit_stride=1,
			reference_maximum_step=0.005,
			audit_maximum_step=0.0025,
		)
		result = run_gauss_legendre4_evaluation(
			potential,
			initial,
			config=config,
		)
		self.assertEqual(tuple(result.solutions), config.integration_steps)
		self.assertEqual(len(result.summaries()), 2)
		self.assertEqual(len(result.observed_orders()), 1)
		for row in result.summaries():
			self.assertGreater(row.runtime_seconds, 0.0)
			self.assertLess(row.maximum_local_symplecticity_defect, 1e-12)
			self.assertGreaterEqual(row.maximum_relative_generalized_energy_error, 0.0)
			self.assertGreaterEqual(
				row.newton_audit_time_integrated_rms_difference,
				0.0,
			)
		for step in config.integration_steps:
			self.assertEqual(
				result.energy_times[step].size,
				int(round((config.t_span[1] - config.t_span[0]) / step)) + 1,
			)
		figures = (
			plot_gauss_legendre4_evaluation(
				result.summaries(),
				designed_order=config.designed_order,
				reference_floor=time_integrated_rms_floor(result.reference),
			)[0],
			plot_gauss_legendre4_observed_order(
				result.observed_orders(),
				designed_order=config.designed_order,
				reduction_threshold=config.order_reduction_threshold,
			)[0],
			plot_gauss_legendre4_symplecticity(result.symplecticity)[0],
			plot_gauss_legendre4_energy(
				result.energy_times,
				result.generalized_energies,
			)[0],
		)
		for figure in figures:
			plt.close(figure)

	def test_gauss_bm4_comparison_is_aligned_and_timed(self) -> None:
		potential, initial = _problem_data()
		config = GaussBM4ComparisonConfig(
			integration_steps=(0.02, 0.01),
			t_span=(0.0, 0.04),
			save_interval=0.02,
			rho=0.05,
			timing_warmups=0,
			timing_repeats=1,
			reference_maximum_step=0.005,
			audit_maximum_step=0.0025,
		)
		result = run_gauss_bm4_comparison(
			potential,
			initial,
			config=config,
		)
		self.assertEqual(tuple(result.solutions), GAUSS_BM4_METHODS)
		self.assertEqual(len(result.summaries()), 4)
		self.assertEqual(len(result.equal_step_ratios()), 2)
		self.assertEqual(len(result.equal_accuracy_ratios()), 3)
		for row in result.equal_accuracy_ratios():
			self.assertGreater(row.gauss_runtime_seconds, 0.0)
			self.assertGreater(row.bm4_runtime_seconds, 0.0)
		for method_name in GAUSS_BM4_METHODS:
			for step in config.integration_steps:
				self.assertEqual(result.runtime_samples[method_name][step].shape, (1,))
				self.assertTrue(
					np.array_equal(
						result.solutions[method_name][step].t,
						result.reference.times,
					)
				)
		figure, _ = plot_gauss_bm4_accuracy_runtime(
			result.summaries(),
			result.observed_orders(),
			designed_order=config.designed_order,
			reference_floor=time_integrated_rms_floor(result.reference),
		)
		plt.close(figure)

	def test_reflected_stationary_corner_has_undefined_error_ratio(self) -> None:
		grid = Grid.periodic(8, 8, period=1.0)
		axis = np.linspace(0.0, 1.0, 9)
		profile = 3 * axis**2 - 2 * axis**3
		closed = profile[:, None] + profile[None, :]
		potential = Potential(grid, mean=closed[:-1, :-1], reflected_samples=closed[None])
		self.assertEqual(potential.spatial_extension, "reflected")
		config = GaussBM4ComparisonConfig(
			integration_steps=(0.02, 0.01), t_span=(0.0, 0.04), save_interval=0.02,
			rho=0.0, timing_warmups=0, timing_repeats=1,
			reference_maximum_step=0.005, audit_maximum_step=0.0025,
		)
		result = run_gauss_bm4_comparison(potential, np.zeros(2), config=config)
		for method_solutions in result.solutions.values():
			for solution in method_solutions.values():
				np.testing.assert_array_equal(solution.states, np.zeros_like(solution.states))
		for row in result.summaries():
			self.assertEqual(row.time_integrated_rms_distance, 0.0)
		for ratio in result.equal_step_ratios():
			self.assertTrue(np.isnan(ratio.gauss_to_bm4_error_ratio))
			self.assertTrue(np.isfinite(ratio.bm4_to_gauss_runtime_ratio))
			self.assertGreater(ratio.bm4_to_gauss_runtime_ratio, 0.0)
		for order in result.observed_orders():
			self.assertFalse(order.resolved_above_reference_floor)
			self.assertTrue(np.isnan(order.time_integrated_rms_order))
		self.assertEqual(result.equal_accuracy_ratios(), ())

		# Distinguish an undefined 0/0 from a positive error over an exact result,
		# while preserving ordinary finite ratios when the denominator is positive.
		for gauss_error, bm4_error, expected in ((1e-6, 0.0, np.inf), (0.0, 1e-6, 0.0), (1e-6, 2e-6, 0.5)):
			rows = tuple(replace(row, time_integrated_rms_distance=(
				gauss_error if row.method_name == "GaussLegendre4" else bm4_error
			)) for row in result.summaries())
			with patch.object(type(result), "summaries", return_value=rows):
				for ratio in result.equal_step_ratios():
					self.assertEqual(ratio.gauss_to_bm4_error_ratio, expected)


if __name__ == "__main__":
	unittest.main()
