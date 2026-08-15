"""Symplecticity analysis for stage-projected BM4 composition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies import (
	AreaStep,
	ProjectedBM4SymplecticityConfig,
	ProjectedBM4SymplecticityResult,
	ProjectedBM4SymplecticitySummary,
	RandomPotentialConfig,
	centered_square,
	run_projected_bm4_symplecticity_study,
)


class ProjectedBM4SymplecticityStudyTests(unittest.TestCase):
	"""Verify complete-step local and accumulated physical diagnostics."""

	def test_short_study_returns_local_and_flow_defects(self) -> None:
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
		config = ProjectedBM4SymplecticityConfig(
			steps=(AreaStep("h=0.01", 0.01), AreaStep("h=0.005", 0.005)),
			t_span=(0.0, 0.02),
			save_interval=0.01,
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_projected_bm4_symplecticity_study(
				potential,
				area,
				notebook_path=(
					root
					/ "notebooks"
					/ "developements"
					/ "projected_bm4_symplecticity.ipynb"
				),
				config=config,
				project_root=root,
			)

		self.assertIs(type(result), ProjectedBM4SymplecticityResult)
		self.assertEqual(tuple(result.solutions), ("h=0.01", "h=0.005"))
		for step in config.steps:
			records = result.records[step.label]
			self.assertEqual(len(records), 3)
			np.testing.assert_allclose(
				[record.time for record in records],
				result.solutions[step.label].t,
			)
			self.assertTrue(
				all(np.isfinite(record.local_relative_defect) for record in records)
			)
			self.assertTrue(
				all(np.isfinite(record.relative_defect) for record in records)
			)
			if np.isclose(step.value, config.save_interval):
				self.assertAlmostEqual(
					records[1].local_relative_defect,
					records[1].relative_defect,
					places=13,
				)
			self.assertLess(
				max(record.relative_copy_separation for record in records),
				1e-13,
			)

		summaries = result.summaries()
		self.assertEqual(len(summaries), 2)
		self.assertTrue(
			all(type(row) is ProjectedBM4SymplecticitySummary for row in summaries)
		)
		self.assertEqual(len(result.convergence_orders()), 1)
		diagnostic_figure, diagnostic_axes = result.plot_diagnostics()
		convergence_figure, convergence_axis = result.plot_convergence()
		self.assertEqual(diagnostic_axes.shape, (2, 2))
		self.assertEqual(len(convergence_axis.lines), 3)
		diagnostic_figure.canvas.draw()
		convergence_figure.canvas.draw()
		plt.close(diagnostic_figure)
		plt.close(convergence_figure)


if __name__ == "__main__":
	unittest.main()
