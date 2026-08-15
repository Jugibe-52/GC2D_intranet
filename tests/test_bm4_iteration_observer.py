"""Nonlinear-iteration diagnostics for projected BM4 methods."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import ImplicitBM4IterationObserver
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	BM4Implicit1,
	BM4Implicit2,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from studies import (
	BM4ImplicitIterationStudyConfig,
	run_bm4_implicit_iteration_study,
)
from visualization import (
	plot_implicit_bm4_iteration_comparison,
	plot_implicit_bm4_iteration_diagnostics,
)


def _potential() -> Potential:
	"""Build a compact deterministic potential for projected-BM4 tests."""
	return Potential.random(A=0.08, M=3, nx=16, ny=16, seed=27)


def _configuration() -> GCInitialConfiguration:
	"""Return one planar guiding-centre particle."""
	return GCInitialConfiguration.from_components(
		x=np.asarray([1.0]),
		y=np.asarray([1.2]),
	)


class ImplicitBM4IterationObserverTests(unittest.TestCase):
	"""Verify projected-BM4 iteration capture, study assembly, and plots."""

	def test_observer_matches_both_bm4_solution_diagnostics(self) -> None:
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.04),
			max_step=0.02,
			sample_count=3,
		)
		for method_type in (BM4Implicit1, BM4Implicit2):
			with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
				root = Path(temporary)
				with ImplicitBM4IterationObserver(
					notebook_path=(
						root / "notebooks" / "developements" / "bm4.ipynb"
					),
					project_root=root,
					chunk_size=2,
				) as observer:
					solution = simulate(
						InitialValueProblem(
							GuidingCenterDynamics(_potential(), rho=0.05),
							_configuration(),
						),
						method_type(step_observer=observer),
						request,
					)
				iterations = np.asarray(
					[record.newton_iterations for record in observer.records]
				)
				np.testing.assert_array_equal(
					iterations,
					solution.diagnostics["newton_iterations"],
				)
				self.assertEqual(len(observer.records), solution.n_steps)
				self.assertEqual(len(observer.output_blocks), 1)
				self.assertTrue(
					all(
						record.method_name == method_type.__name__
						for record in observer.records
					)
				)

	def test_study_and_bm4_iteration_plots(self) -> None:
		config = BM4ImplicitIterationStudyConfig(
			formulation="implicit_2",
			rho=0.05,
			t_span=(0.0, 0.04),
			max_step=0.02,
			sample_count=3,
			observer_chunk_size=2,
		)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_bm4_implicit_iteration_study(
				_potential(),
				_configuration(),
				notebook_path=(
					root / "notebooks" / "developements" / "study.ipynb"
				),
				config=config,
				project_root=root,
			)
			self.assertEqual(result.iteration_counts.shape, (2,))
			self.assertFalse(result.iteration_counts.flags.writeable)
			self.assertEqual(sum(result.iteration_frequencies().values()), 2)
			figure, axes = plot_implicit_bm4_iteration_diagnostics(result.records)
			self.assertEqual(axes.shape, (2, 2))
			figure.canvas.draw()
			plt.close(figure)
			figure, axes = plot_implicit_bm4_iteration_comparison(
				{"implicit_2": result.records}
			)
			self.assertEqual(axes.shape, (2,))
			figure.canvas.draw()
			plt.close(figure)


if __name__ == "__main__":
	unittest.main()
