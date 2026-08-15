"""Contracts for the averaged five-trajectory midpoint-BM4 study."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from studies import (
	AreaStep,
	MidpointBM4SymplecticityConfig,
	RandomPotentialConfig,
	random_gc_configuration,
	run_midpoint_bm4_symplecticity_study,
)


class MidpointBM4SymplecticityStudyTests(unittest.TestCase):
	"""Verify three-step comparison and arithmetic trajectory aggregation."""

	def test_five_random_trajectories_are_aligned_for_three_steps(self) -> None:
		potential_config = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		)
		potential = potential_config.build()
		configuration = random_gc_configuration(
			potential,
			particle_count=5,
			seed=31415,
		)
		config = MidpointBM4SymplecticityConfig(
			steps=(
				AreaStep(label="h=0.1", value=0.1),
				AreaStep(label="h=0.05", value=0.05),
				AreaStep(label="h=0.025", value=0.025),
			),
			t_span=(0.0, 0.2),
			save_interval=0.1,
			rho=0.05,
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_midpoint_bm4_symplecticity_study(
				potential,
				configuration,
				notebook_path=(
					root
					/ "notebooks"
					/ "developements"
					/ "bm4_midpoint_symplecticity.ipynb"
				),
				config=config,
				project_root=root,
				metadata={
					**potential_config.metadata(),
					"initial_condition_seed": 31415,
				},
			)
			first_metadata = sorted(
				result.output_directories["h=0.1"].glob("*_metadata_*.json")
			)[0]
			payload = json.loads(first_metadata.read_text(encoding="utf-8"))

		self.assertEqual(len(config.steps), 3)
		self.assertEqual(tuple(result.solutions), tuple(step.label for step in config.steps))
		self.assertEqual(payload["metadata"]["trajectory_count"], 5)
		self.assertEqual(
			payload["metadata"]["jacobian_method"],
			"explicit_uncoupled_stage_factorization",
		)
		self.assertEqual(
			payload["metadata"]["symplecticity_reduction"],
			"arithmetic_mean_of_per_trajectory_relative_defects",
		)
		self.assertEqual(payload["metadata"]["projection_kind"], "arithmetic_mean")
		self.assertEqual(
			payload["metadata"]["projection_scope"],
			"complete_bm4_cycle",
		)
		self.assertEqual(payload["metadata"]["projections_per_step"], 1)
		self.assertEqual(payload["metadata"]["t_span"], [0.0, 0.2])
		self.assertEqual(payload["metadata"]["save_interval"], 0.1)

		for expected_step_count, step in zip((2, 4, 8), config.steps, strict=True):
			label = step.label
			solution = result.solutions[label]
			records = result.records[label]
			self.assertEqual(solution.states.shape, (10, 3))
			self.assertEqual(solution.n_steps, expected_step_count)
			np.testing.assert_array_equal(
				solution.states[:, 0],
				configuration.initial_state,
			)
			np.testing.assert_array_equal(
				solution.t,
				result.solutions[config.steps[0].label].t,
			)
			self.assertEqual(len(records), 3)
			np.testing.assert_allclose(
				[record.time for record in records],
				solution.t,
				rtol=0.0,
				atol=1e-14,
			)
			self.assertEqual(records[0].mean_accumulated_relative_defect, 0.0)
			self.assertTrue(
				all(
					np.isfinite(record.mean_accumulated_relative_defect)
					for record in records
				)
			)

		summaries = result.summaries()
		self.assertEqual(len(summaries), 3)
		self.assertTrue(all(row.trajectory_count == 5 for row in summaries))
		self.assertEqual(len(result.convergence_orders()), 2)
		error_figure, error_axes = result.plot_symplecticity()
		trajectory_figure, trajectory_axis = result.plot_trajectories()
		self.assertEqual(error_axes.shape, (2,))
		self.assertTrue(all(len(axis.lines) == 3 for axis in error_axes))
		self.assertEqual(len(trajectory_axis.lines), 5)
		self.assertTrue(
			all(line.get_linestyle() == "None" for line in trajectory_axis.lines)
		)
		self.assertTrue(
			all(line.get_marker() == "." for line in trajectory_axis.lines)
		)
		plt.close(error_figure)
		plt.close(trajectory_figure)

	def test_sampling_interval_must_be_divisible_by_each_step(self) -> None:
		with self.assertRaises(ValueError):
			MidpointBM4SymplecticityConfig(
				steps=(AreaStep(label="h=0.03", value=0.03),),
				t_span=(0.0, 0.2),
				save_interval=0.1,
			)

	def test_steps_must_be_ordered_from_coarsest_to_finest(self) -> None:
		with self.assertRaises(ValueError):
			MidpointBM4SymplecticityConfig(
				steps=(
					AreaStep(label="h=0.05", value=0.05),
					AreaStep(label="h=0.1", value=0.1),
				),
				t_span=(0.0, 0.2),
				save_interval=0.1,
			)


if __name__ == "__main__":
	unittest.main()
