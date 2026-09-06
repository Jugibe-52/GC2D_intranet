"""Contracts for the four-method SDIRK long-time comparison."""

from __future__ import annotations

import unittest

import numpy as np

from studies import (
	FOUR_METHOD_SDIRK_METHODS,
	FourMethodSDIRKComparisonConfig,
	RandomPotentialConfig,
	latin_hypercube_gc_configuration,
	run_four_method_sdirk_comparison,
)


class FourMethodSDIRKComparisonTests(unittest.TestCase):
	"""Verify aligned solutions, summaries, and SDIRK diagnostics."""

	def test_short_four_method_study(self) -> None:
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
			seed=20260905,
			domain_margin_fraction=0.35,
		)
		config = FourMethodSDIRKComparisonConfig(
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
		result = run_four_method_sdirk_comparison(
			potential,
			initial_configuration,
			config=config,
		)

		self.assertEqual(
			FOUR_METHOD_SDIRK_METHODS,
			(
				"ABBA4ImplicitSingleProjection",
				"GaussLegendre4",
				"BM4Implicit",
				"SDIRK4",
			),
		)
		self.assertEqual(tuple(result.solutions), FOUR_METHOD_SDIRK_METHODS)
		self.assertEqual(tuple(result.accuracy), FOUR_METHOD_SDIRK_METHODS)
		self.assertEqual(tuple(result.energy_accuracy), FOUR_METHOD_SDIRK_METHODS)
		self.assertEqual(tuple(result.runtime_samples), FOUR_METHOD_SDIRK_METHODS)
		self.assertEqual(result.reference.times.size, 3)
		self.assertEqual(result.reference_energies.shape, (3, 3))
		self.assertEqual(result.audit_energies.shape, (3, 3))
		self.assertGreater(result.total_method_runtime_seconds, 0.0)
		self.assertGreater(result.total_study_runtime_seconds, 0.0)

		for method_name in FOUR_METHOD_SDIRK_METHODS:
			solution = result.solutions[method_name]
			self.assertEqual(solution.states.shape, (6, 3))
			self.assertEqual(solution.diagnostics["step_count"], 2)
			self.assertEqual(
				np.asarray(solution.diagnostics["nonlinear_iterations"]).shape,
				(2,),
			)
			self.assertEqual(result.runtime_samples[method_name].shape, (1,))
			self.assertFalse(result.runtime_samples[method_name].flags.writeable)

		sdirk_diagnostics = result.solutions["SDIRK4"].diagnostics
		self.assertEqual(sdirk_diagnostics["stage_count"], 5)
		self.assertFalse(sdirk_diagnostics["symmetric"])
		self.assertFalse(sdirk_diagnostics["symplectic"])
		self.assertEqual(
			np.asarray(sdirk_diagnostics["stage_nonlinear_iterations"]).shape,
			(2, 5),
		)

		summaries = result.summaries()
		self.assertEqual(len(summaries), 4)
		self.assertEqual(
			tuple(row.method_name for row in summaries),
			FOUR_METHOD_SDIRK_METHODS,
		)
		self.assertTrue(all(row.nonlinear_solver == "Newton" for row in summaries))
		self.assertTrue(all(row.trajectory_count == 3 for row in summaries))
		self.assertTrue(all(row.step_count == 2 for row in summaries))
		self.assertTrue(all(row.runtime_seconds > 0.0 for row in summaries))
		self.assertTrue(all(row.maximum_residual_to_tolerance <= 1.0 for row in summaries))


if __name__ == "__main__":
	unittest.main()
