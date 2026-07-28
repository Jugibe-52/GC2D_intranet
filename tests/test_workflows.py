"""Fast contracts for reusable experiment initialization and composition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from classes import Area, TrajectoryGC
from workflows import (
	AreaComparisonConfig,
	GeneralizedEnergyConfig,
	RandomPotentialConfig,
	RK4SymplecticityConfig,
	centered_circle,
	centered_gc_trajectory,
	centered_square,
	domain_center,
	pi_area_steps,
	run_area_comparison,
	run_generalized_energy_comparison,
	run_rk4_symplecticity_study,
)


def small_potential_config() -> RandomPotentialConfig:
	"""Return a deterministic field small enough for workflow contract tests."""
	return RandomPotentialConfig(
		amplitude=0.08,
		max_wave_number=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=3,
	)


class InitializationWorkflowTests(unittest.TestCase):
	"""Verify shared potential and centered-condition construction."""

	def test_random_potential_configuration_is_reproducible(self) -> None:
		config = small_potential_config()
		first = config.build()
		second = config.build()

		np.testing.assert_allclose(first.evaluate(0.2), second.evaluate(0.2))
		self.assertEqual(config.metadata()["max_wave_number"], 3)

	def test_centered_initial_conditions_use_the_periodic_cell_center(self) -> None:
		potential = small_potential_config().build()
		expected_center = domain_center(potential)

		trajectory = centered_gc_trajectory(potential, rho=0.3)
		self.assertIsInstance(trajectory, TrajectoryGC)
		state = trajectory.initial_state
		assert state is not None
		x, y = trajectory.positions(state)
		np.testing.assert_allclose((x[0], y[0]), expected_center)

		circle = centered_circle(
			potential,
			radius=0.5,
			points=8,
			rho=0.3,
		)
		square = centered_square(
			potential,
			side=1.0,
			points_per_side=2,
			rho=0.3,
		)
		self.assertIsInstance(circle, Area)
		self.assertIsInstance(square, Area)
		self.assertEqual(circle.particle_count(circle.initial_state), 8)
		self.assertAlmostEqual(float(square.calculate_area()), 1.0)


class AreaComparisonWorkflowTests(unittest.TestCase):
	"""Verify synchronization and collected projected-area results."""

	def test_short_area_comparison_returns_aligned_diagnostics(self) -> None:
		potential = small_potential_config().build()
		area = centered_square(
			potential,
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		config = AreaComparisonConfig(
			steps=pi_area_steps(400, 800),
			t_span=(0.0, np.pi / 100),
			save_interval=np.pi / 100,
			coupling_frequency=0.0,
			chunk_size=2,
			progress=False,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_area_comparison(
				potential,
				area,
				notebook_path=root / "notebooks" / "developements" / "area.ipynb",
				config=config,
				project_root=root,
				metadata={"purpose": "test"},
			)

			self.assertEqual(tuple(result.solutions), tuple(step.label for step in config.steps))
			self.assertEqual(config.output_sample_count, 2)
			for step in config.steps:
				label = step.label
				self.assertEqual(result.solutions[label].states.shape[1], 2)
				self.assertEqual(result.diagnostic_times[label].shape, (2,))
				self.assertTrue(result.output_directories[label].is_dir())
			self.assertEqual(len(result.summaries()), 2)

	def test_sampling_ratios_must_be_integral(self) -> None:
		with self.assertRaises(ValueError):
			AreaComparisonConfig(
				steps=pi_area_steps(40, 80),
				t_span=(0.0, np.pi),
				save_interval=0.1,
			)

	def test_stage_projected_area_comparison_has_no_copy_separation(self) -> None:
		potential = small_potential_config().build()
		area = centered_square(
			potential,
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		config = AreaComparisonConfig(
			steps=pi_area_steps(400, 800),
			t_span=(0.0, np.pi / 100),
			save_interval=np.pi / 100,
			method_kind="stage_projected_bm4",
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_area_comparison(
				potential,
				area,
				notebook_path=root / "notebooks" / "experiments" / "area.ipynb",
				config=config,
				project_root=root,
			)

		for separation in result.relative_copy_separations.values():
			np.testing.assert_allclose(separation, 0.0)

	def test_stage_projected_method_rejects_coupling(self) -> None:
		with self.assertRaises(ValueError):
			AreaComparisonConfig(
				steps=pi_area_steps(40, 80),
				t_span=(0.0, np.pi),
				save_interval=np.pi / 8,
				coupling_frequency=1.0,
				method_kind="stage_projected_bm4",
			)


class EnergyWorkflowTests(unittest.TestCase):
	"""Verify canonical diagnostics and plotting for energy comparisons."""

	def test_short_generalized_energy_comparison(self) -> None:
		potential = small_potential_config().build()
		trajectory = centered_gc_trajectory(potential, rho=0.05)
		config = GeneralizedEnergyConfig(
			steps=(0.01, 0.005),
			t_span=(0.0, 0.02),
			output_sample_count=3,
			progress=False,
		)

		result = run_generalized_energy_comparison(
			potential,
			trajectory,
			config=config,
		)

		self.assertEqual(tuple(result.solutions), config.steps)
		for step in config.steps:
			self.assertEqual(result.relative_errors[step].shape, (3,))
			self.assertIn("energy_error", result.solutions[step].diagnostics)
		np.testing.assert_allclose(
			[result.relative_errors[step][0] for step in config.steps],
			0.0,
		)
		figure, axes = result.plot()
		self.assertEqual(len(axes.lines), len(config.steps) + 1)
		plt.close(figure)


class RK4SymplecticityWorkflowTests(unittest.TestCase):
	"""Verify synchronized physical GC symplecticity analysis for RK4."""

	def test_short_rk4_study_returns_aligned_diagnostics(self) -> None:
		potential = small_potential_config().build()
		area = centered_square(
			potential,
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		config = RK4SymplecticityConfig(
			steps=pi_area_steps(400, 800),
			t_span=(0.0, np.pi / 100),
			save_interval=np.pi / 100,
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_rk4_symplecticity_study(
				potential,
				area,
				notebook_path=(
					root
					/ "notebooks"
					/ "experiments"
					/ "symplecticity"
					/ "rk4.ipynb"
				),
				config=config,
				project_root=root,
			)

		self.assertEqual(tuple(result.solutions), tuple(
			step.label for step in config.steps
		))
		for step in config.steps:
			self.assertEqual(len(result.records[step.label]), 2)
			np.testing.assert_allclose(
				[record.time for record in result.records[step.label]],
				result.solutions[step.label].t,
			)
		self.assertEqual(len(result.summaries()), 2)
		self.assertEqual(len(result.convergence_orders()), 1)
		diagnostic_figure, diagnostic_axes = result.plot_diagnostics()
		convergence_figure, convergence_axes = result.plot_convergence()
		self.assertEqual(diagnostic_axes.shape, (2, 2))
		self.assertEqual(len(convergence_axes.lines), 2)
		animation = result.animate(frames=2, interval=10)
		self.assertGreater(len(animation._func(1)), 0)
		animation._draw_was_started = True
		plt.close(diagnostic_figure)
		plt.close(convergence_figure)


if __name__ == "__main__":
	unittest.main()
