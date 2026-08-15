"""Jacobian-only diagnostics for complete implicit ABBA steps."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.abba_jacobian import (
	ImplicitABBAJacobianObserver,
	analyze_particle_jacobian,
	line_angle,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ImplicitABBA1,
	ImplicitABBA2,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from studies import (
	ImplicitABBAJacobianStudyConfig,
	run_implicit_abba_jacobian_study,
)
from visualization import (
	plot_implicit_abba_jacobian_directions,
	plot_implicit_abba_jacobian_matrices,
	plot_implicit_abba_jacobian_polar_snapshots,
	plot_implicit_abba_jacobian_spectrum,
	plot_implicit_abba_particle_step_series,
)


def _potential() -> Potential:
	"""Build a compact Hessian-capable potential for observer tests."""
	return Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)


class ParticleJacobianAnalysisTests(unittest.TestCase):
	"""Verify robust real, complex, and repeated spectral classifications."""

	def test_hyperbolic_analysis_has_two_real_eigenlines(self) -> None:
		analysis = analyze_particle_jacobian(np.diag([2.0, 0.5]))
		self.assertEqual(analysis.spectral_class, "hyperbolic")
		self.assertTrue(analysis.eigendirections_defined)
		np.testing.assert_allclose(np.abs(analysis.eigenvalues), [0.5, 2.0])
		self.assertTrue(np.all(np.isfinite(analysis.eigenvector_line_angles)))

	def test_elliptic_analysis_preserves_complex_vectors_without_real_angles(
		self,
	) -> None:
		quarter_turn = np.asarray([[0.0, -1.0], [1.0, 0.0]])
		analysis = analyze_particle_jacobian(quarter_turn)
		self.assertEqual(analysis.spectral_class, "elliptic")
		self.assertFalse(analysis.eigendirections_defined)
		self.assertTrue(np.all(np.isnan(analysis.eigenvector_line_angles)))
		np.testing.assert_allclose(np.abs(analysis.eigenvalues), 1.0)
		self.assertTrue(np.any(np.abs(analysis.eigenvectors.imag) > 0.0))
		self.assertFalse(analysis.singular_directions_defined)

	def test_repeated_spectrum_is_parabolic_and_line_angles_ignore_sign(self) -> None:
		analysis = analyze_particle_jacobian(np.asarray([[1.0, 1.0], [0.0, 1.0]]))
		self.assertEqual(analysis.spectral_class, "parabolic")
		self.assertFalse(analysis.eigendirections_defined)
		self.assertAlmostEqual(
			line_angle(np.asarray([1.0, 2.0])),
			line_angle(np.asarray([-1.0, -2.0])),
		)


class ImplicitABBAJacobianObserverTests(unittest.TestCase):
	"""Verify ABBA step integration, persistence, and presentation helpers."""

	def test_observer_analyzes_both_implicit_formulations_without_symplecticity(
		self,
	) -> None:
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0, 1.4]),
			y=np.asarray([1.2, 1.6]),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.04),
			max_step=0.02,
			sample_count=3,
		)
		for method_type in (ImplicitABBA1, ImplicitABBA2):
			with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
				root = Path(temporary)
				with ImplicitABBAJacobianObserver(
					notebook_path=(
						root
						/ "notebooks"
						/ "developements"
						/ "abba_jacobian.ipynb"
					),
					project_root=root,
					chunk_size=2,
					verbose=False,
				) as observer:
					solution = simulate(
						InitialValueProblem(
							GuidingCenterDynamics(_potential(), rho=0.05),
							configuration,
						),
						method_type(step_observer=observer),
						request,
					)
				self.assertEqual(len(observer.samples), solution.n_steps)
				self.assertEqual(len(observer.records), 2 * solution.n_steps)
				self.assertEqual(observer.samples[0].jacobian.shape, (4, 4))
				self.assertEqual(
					observer.samples[0].particle_jacobians.shape,
					(2, 2, 2),
				)
				self.assertEqual(len(observer.output_blocks), 1)
				block = observer.output_blocks[0]
				with np.load(block.arrays_path) as arrays:
					self.assertEqual(arrays["particle_jacobians"].shape, (2, 2, 2, 2))
					self.assertEqual(arrays["eigenvalues"].shape, (2, 2, 2))
					self.assertEqual(arrays["eigenvectors"].shape, (2, 2, 2, 2))
				with block.metadata_path.open(encoding="utf-8") as stream:
					metadata = json.load(stream)
				self.assertNotIn("symplectic", json.dumps(metadata).lower())

	def test_study_and_all_jacobian_plots_run_for_one_particle(self) -> None:
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0]),
			y=np.asarray([1.2]),
		)
		config = ImplicitABBAJacobianStudyConfig(
			rho=0.05,
			t_span=(0.0, 0.04),
			max_step=0.02,
			sample_count=3,
			observer_chunk_size=2,
		)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_implicit_abba_jacobian_study(
				_potential(),
				configuration,
				notebook_path=(
					root / "notebooks" / "developements" / "study.ipynb"
				),
				config=config,
				project_root=root,
			)
			self.assertEqual(sum(result.spectral_class_counts().values()), 2)
			step_series = result.particle_step_series(particle_index=0)
			self.assertEqual(step_series.start_times.shape, (2,))
			self.assertEqual(step_series.end_times.shape, (2,))
			self.assertEqual(step_series.states_before.shape, (2, 2))
			self.assertEqual(step_series.states_after.shape, (2, 2))
			self.assertEqual(
				step_series.effective_electric_fields_after.shape,
				(2, 2),
			)
			self.assertEqual(step_series.state_increments.shape, (2, 2))
			self.assertEqual(step_series.state_increment_norms.shape, (2,))
			self.assertEqual(step_series.state_increment_angles.shape, (2,))
			self.assertFalse(step_series.state_increments.flags.writeable)
			self.assertFalse(step_series.state_increment_angles.flags.writeable)
			for sample_index, sample in enumerate(result.samples):
				expected_before = sample.state_before[[0, 1]]
				expected_after = sample.state_after[[0, 1]]
				np.testing.assert_allclose(
					step_series.states_before[:, sample_index],
					expected_before,
				)
				np.testing.assert_allclose(
					step_series.states_after[:, sample_index],
					expected_after,
				)
				np.testing.assert_allclose(
					step_series.state_increments[:, sample_index],
					expected_after - expected_before,
				)
				expected_increment = expected_after - expected_before
				self.assertAlmostEqual(
					step_series.state_increment_angles[sample_index],
					np.arctan2(expected_increment[1], expected_increment[0]),
				)
			field_x, field_y = result.dynamics.effective_potential.electric_field(
				step_series.end_times,
				step_series.states_after[0],
				step_series.states_after[1],
			)
			np.testing.assert_allclose(
				step_series.effective_electric_fields_after,
				np.vstack((field_x, field_y)),
			)
			for sample_index, time in enumerate(step_series.end_times):
				np.testing.assert_allclose(
					result.dynamics.vector_field(
						time,
						step_series.states_after[:, sample_index],
					),
					[
						step_series.effective_electric_fields_after[1, sample_index],
						-step_series.effective_electric_fields_after[0, sample_index],
					],
				)
			with self.assertRaises(TypeError):
				result.particle_step_series(particle_index=True)
			figure, axes = plot_implicit_abba_particle_step_series(step_series)
			self.assertEqual(axes.shape, (3,))
			self.assertEqual(
				axes[0].get_title(),
				"Effective electric field at step endpoints",
			)
			self.assertEqual(
				axes[2].get_title(),
				r"Oriented direction of $\Delta z_n$",
			)
			figure.canvas.draw()
			plt.close(figure)
			plotters = (
				plot_implicit_abba_jacobian_matrices,
				plot_implicit_abba_jacobian_spectrum,
				plot_implicit_abba_jacobian_directions,
				plot_implicit_abba_jacobian_polar_snapshots,
			)
			for plotter in plotters:
				figure, axes = plotter(result.samples)
				self.assertIsNotNone(figure)
				if plotter is plot_implicit_abba_jacobian_matrices:
					self.assertEqual(axes.shape, (2, 3))
					self.assertEqual(axes[0, 1].get_title(), "Jacobian trace")
					self.assertEqual(axes[0, 2].get_title(), "Jacobian determinant")
				plt.close(figure)


if __name__ == "__main__":
	unittest.main()
