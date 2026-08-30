"""Exact-Jacobian contracts for independent-trajectory symplecticity studies."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import (
	bm4_implicit_1_step_particle_jacobians,
	central_difference_jacobian,
	abba2_implicit_step_particle_jacobians,
	abba2_midpoint_step_particle_jacobians,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	BM4Implicit1,
	ABBA2Implicit,
	InitialValueProblem,
	IntegrationStep,
	ABBA2Midpoint,
	SimulationRequest,
	simulate,
)
from studies import (
	AreaStep,
	RandomPotentialConfig,
	TrajectorySymplecticityConfig,
	random_gc_configuration,
	run_bm4_implicit_1_trajectory_symplecticity_study,
	run_abba2_reduced_multiplier_trajectory_symplecticity_study,
	run_abba2_midpoint_trajectory_symplecticity_study,
)


def _problem() -> InitialValueProblem:
	"""Return a compact two-particle problem with exact potential Hessians."""
	potential = Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)
	dynamics = GuidingCenterDynamics(potential, rho=0.05)
	configuration = GCInitialConfiguration.from_components(
		x=np.asarray([1.0, 1.4]),
		y=np.asarray([1.2, 1.6]),
	)
	return InitialValueProblem(dynamics, configuration)


def _dense(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent blocks into component-major packed layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


class ExactTrajectoryJacobianTests(unittest.TestCase):
	"""Audit all three analytic complete-step Jacobian constructions."""

	def test_exact_jacobians_match_centered_difference_audits(self) -> None:
		problem = _problem()
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.1,
			sample_count=2,
		)
		cases = (
			(ABBA2Midpoint, abba2_midpoint_step_particle_jacobians),
			(ABBA2Implicit, abba2_implicit_step_particle_jacobians),
			(BM4Implicit1, bm4_implicit_1_step_particle_jacobians),
		)
		for method_type, calculator in cases:
			with self.subTest(method=method_type.__name__):
				events: list[IntegrationStep] = []
				kwargs: dict[str, object] = {"step_observer": events.append}
				if method_type in (ABBA2Implicit, BM4Implicit1):
					kwargs.update(
						newton_absolute_tolerance=1e-14,
						newton_relative_tolerance=1e-14,
					)
				simulate(problem, method_type(**kwargs), request)
				self.assertEqual(len(events), 1)
				event = events[0]
				exact = _dense(calculator(event))
				numerical = central_difference_jacobian(
					event.map_state,
					event.state_before,
					relative_step=2e-5,
				)
				relative_error = float(
					np.linalg.norm(exact - numerical, ord="fro")
					/ np.linalg.norm(numerical, ord="fro")
				)
				self.assertLess(relative_error, 2e-8)
				if method_type is BM4Implicit1:
					self.assertEqual(len(event.base_stages), 12)

	def test_three_method_studies_share_five_paths_and_three_steps(self) -> None:
		potential_config = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		)
		potential = potential_config.build()
		configuration = random_gc_configuration(
			potential,
			particle_count=5,
			seed=31415,
		)
		config = TrajectorySymplecticityConfig(
			steps=(
				AreaStep("h=0.1", 0.1),
				AreaStep("h=0.05", 0.05),
				AreaStep("h=0.025", 0.025),
			),
			t_span=(0.0, 0.2),
			save_interval=0.1,
			rho=0.05,
			chunk_size=8,
			newton_absolute_tolerance=1e-14,
			newton_relative_tolerance=1e-14,
		)
		runners = (
			run_abba2_midpoint_trajectory_symplecticity_study,
			run_abba2_reduced_multiplier_trajectory_symplecticity_study,
			run_bm4_implicit_1_trajectory_symplecticity_study,
		)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			for runner in runners:
				with self.subTest(runner=runner.__name__):
					result = runner(
						potential,
						configuration,
						notebook_path=(
							root
							/ "notebooks"
							/ "developements"
							/ "symplecticity"
							/ f"{runner.__name__}.ipynb"
						),
						project_root=root,
						config=config,
					)
					self.assertEqual(len(result.steps), 3)
					for expected_steps, step in zip(
						(2, 4, 8),
						result.steps,
						strict=True,
					):
						solution = result.solutions[step.label]
						records = result.records[step.label]
						self.assertEqual(solution.n_steps, expected_steps)
						self.assertEqual(solution.states.shape, (10, 3))
						self.assertEqual(len(records), 3)
						self.assertEqual(records[0].particle_count, 5)
						np.testing.assert_array_equal(
							solution.states[:, 0],
							configuration.initial_state,
						)
					output = result.output_directories[result.steps[0].label]
					npz_path = sorted(output.glob("*_jacobians_*.npz"))[0]
					csv_path = sorted(output.glob("*_summary_*.csv"))[0]
					with np.load(npz_path) as arrays:
						defects = arrays["accumulated_relative_defects"]
						self.assertEqual(defects.shape, (3, 5))
					with csv_path.open(newline="", encoding="utf-8") as stream:
						rows = list(csv.DictReader(stream))
					np.testing.assert_array_equal(
						[float(row["mean_accumulated_relative_defect"]) for row in rows],
						defects.mean(axis=1),
					)
					error_figure, error_axes = result.plot_symplecticity()
					path_figure, path_axis = result.plot_trajectories()
					self.assertEqual(error_axes.shape, (2,))
					self.assertEqual(len(path_axis.lines), 5)
					self.assertTrue(
						all(line.get_linestyle() == "None" for line in path_axis.lines)
					)
					plt.close(error_figure)
					plt.close(path_figure)


if __name__ == "__main__":
	unittest.main()
