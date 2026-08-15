"""Nonlinear-iteration diagnostics for implicit ABBA methods."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import ImplicitABBAIterationObserver
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ImplicitABBA1,
	ImplicitABBA2,
	InitialValueProblem,
	IntegrationStep,
	SimulationRequest,
	simulate,
)
from studies import (
	ImplicitABBAIterationStudyConfig,
	run_implicit_abba_iteration_study,
)
from visualization import (
	plot_implicit_abba_iteration_comparison,
	plot_implicit_abba_iteration_diagnostics,
)


def _potential() -> Potential:
	"""Build a deterministic Hessian-capable potential for short studies."""
	return Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)


def _configuration() -> GCInitialConfiguration:
	"""Return one planar guiding-centre particle."""
	return GCInitialConfiguration.from_components(
		x=np.asarray([1.0]),
		y=np.asarray([1.2]),
	)


class ImplicitABBAIterationObserverTests(unittest.TestCase):
	"""Verify per-step solver metrics, persistence, and study presentation."""

	def test_observer_matches_solution_diagnostics_for_both_formulations(
		self,
	) -> None:
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.04),
			max_step=0.02,
			sample_count=3,
		)
		for method_type in (ImplicitABBA1, ImplicitABBA2):
			with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
				root = Path(temporary)
				with ImplicitABBAIterationObserver(
					notebook_path=(
						root
						/ "notebooks"
						/ "developements"
						/ "iterations.ipynb"
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

				self.assertEqual(len(observer.records), solution.n_steps)
				iterations = np.asarray(
					[record.newton_iterations for record in observer.records]
				)
				residuals = np.asarray(
					[record.newton_residual_norm for record in observer.records]
				)
				np.testing.assert_array_equal(
					iterations,
					solution.diagnostics["newton_iterations"],
				)
				np.testing.assert_allclose(
					residuals,
					solution.diagnostics["newton_residual_norms"],
				)
				self.assertTrue(
					all(
						record.residual_to_tolerance_ratio <= 1.0
						for record in observer.records
					)
				)
				self.assertEqual(len(observer.output_blocks), 1)
				block = observer.output_blocks[0]
				self.assertIn("_iterations_", block.arrays_path.name)
				with np.load(block.arrays_path) as arrays:
					np.testing.assert_array_equal(
						arrays["newton_iterations"], iterations
					)
				with block.metadata_path.open(encoding="utf-8") as stream:
					metadata = json.load(stream)
				self.assertEqual(metadata["residual_norm"], "infinity")

	def test_study_exposes_frequencies_and_plot(self) -> None:
		config = ImplicitABBAIterationStudyConfig(
			formulation="implicit_1",
			rho=0.05,
			t_span=(0.0, 0.04),
			max_step=0.02,
			sample_count=3,
			observer_chunk_size=2,
		)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_implicit_abba_iteration_study(
				_potential(),
				_configuration(),
				notebook_path=(
					root / "notebooks" / "developements" / "study.ipynb"
				),
				config=config,
				project_root=root,
			)
			self.assertEqual(result.iteration_counts.shape, (2,))
			self.assertEqual(result.end_times.shape, (2,))
			self.assertFalse(result.iteration_counts.flags.writeable)
			self.assertEqual(sum(result.iteration_frequencies().values()), 2)
			figure, axes = plot_implicit_abba_iteration_diagnostics(result.records)
			self.assertEqual(axes.shape, (2, 2))
			self.assertEqual(
				axes[0, 0].get_title(),
				"Nonlinear iterations at each accepted step",
			)
			figure.canvas.draw()
			plt.close(figure)
			figure, axes = plot_implicit_abba_iteration_comparison(
				{"implicit_1": result.records}
			)
			self.assertEqual(axes.shape, (2,))
			self.assertEqual(
				axes[0].get_title(),
				"Nonlinear iteration count by formulation",
			)
			figure.canvas.draw()
			plt.close(figure)

	def test_observer_rejects_a_generic_step(self) -> None:
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			observer = ImplicitABBAIterationObserver(
				notebook_path=(
					root / "notebooks" / "developements" / "study.ipynb"
				),
				project_root=root,
			)
			with self.assertRaisesRegex(TypeError, "ImplicitABBAIntegrationStep"):
				observer(
					IntegrationStep(
						dynamics_name="GuidingCenterDynamics",
						method_name="generic",
						step_index=0,
						time=0.1,
						duration=0.1,
						state_before=np.zeros(2),
						state_after=np.zeros(2),
						map_state=lambda state: state,
					)
				)
			observer.close()


if __name__ == "__main__":
	unittest.main()
