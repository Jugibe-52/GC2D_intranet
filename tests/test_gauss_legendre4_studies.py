"""Fast contracts for Gauss4 evaluation and BM4 comparison studies."""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from initial_conditions import GCInitialConfiguration
from potential import Potential
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


def _problem_data() -> tuple[Potential, GCInitialConfiguration]:
	"""Return the compact deterministic field and one GC initial point."""
	return (
		Potential.random(
			A=0.08,
			M=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		),
		GCInitialConfiguration(np.asarray((1.0, 1.2))),
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
				reference_floor=result.reference.time_integrated_rms_floor,
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
			reference_floor=result.reference.time_integrated_rms_floor,
		)
		plt.close(figure)


if __name__ == "__main__":
	unittest.main()
