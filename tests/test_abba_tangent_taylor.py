"""Proposed tangent-Taylor methods and their trajectory comparisons."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import (
	abba4_implicit_1_step_particle_jacobians,
	implicit_function_step_jacobian,
)
from initial_conditions import GCInitialConfiguration
from simulation import (
	ABBA4Implicit1,
	ABBA4Implicit1TangentTaylor,
	ImplicitABBA1,
	ImplicitABBA1TangentTaylor,
	SimulationRequest,
	simulate,
)
from studies import (
	ABBATangentTaylorComparisonConfig,
	RandomPotentialConfig,
	run_abba4_implicit1_tangent_taylor_comparison,
	run_implicit_abba1_tangent_taylor_comparison,
)
from tests.test_abba4_implicit import _rotation_problem
from visualization import (
	animate_tangent_taylor_particle_evolution,
	plot_tangent_taylor_component_comparison,
	plot_tangent_taylor_trajectory_comparison,
)


def _dense(blocks: np.ndarray) -> np.ndarray:
	"""Expand particle-major planar blocks into component-major layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


class ABBATangentTaylorMethodTests(unittest.TestCase):
	"""Verify that both methods apply the stated complete-map tangent formula."""

	def test_one_step_matches_formula_for_both_base_maps(self) -> None:
		problem = _rotation_problem()
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.1,
			sample_count=2,
		)
		common = {
			"newton_absolute_tolerance": 1e-14,
			"newton_relative_tolerance": 1e-14,
		}
		for base_type, tangent_type in (
			(ImplicitABBA1, ImplicitABBA1TangentTaylor),
			(ABBA4Implicit1, ABBA4Implicit1TangentTaylor),
		):
			events = []
			simulate(
				problem,
				base_type(step_observer=events.append, **common),
				request,
			)
			step = events[0]
			jacobian = (
				_dense(abba4_implicit_1_step_particle_jacobians(step))
				if base_type is ABBA4Implicit1
				else implicit_function_step_jacobian(step)
			)
			state = problem.initial_state
			velocity = problem.dynamics.vector_field(0.0, state)
			expected = state + 0.1 * velocity + 0.5 * 0.1**2 * (jacobian @ velocity)
			solution = simulate(problem, tangent_type(**common), request)
			np.testing.assert_allclose(
				solution.states[:, -1],
				expected,
				rtol=0.0,
				atol=2e-15,
			)
			expected_substeps = 3 if base_type is ABBA4Implicit1 else 1
			self.assertEqual(
				solution.diagnostics["base_substeps_per_step"],
				expected_substeps,
			)

	def test_both_comparison_studies_and_plots(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		).build()
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0]),
			y=np.asarray([1.2]),
		)
		config = ABBATangentTaylorComparisonConfig(
			rho=0.05,
			t_span=(0.0, 0.04),
			max_step=0.02,
			sample_count=3,
			newton_absolute_tolerance=1e-14,
			newton_relative_tolerance=1e-14,
		)
		for runner in (
			run_implicit_abba1_tangent_taylor_comparison,
			run_abba4_implicit1_tangent_taylor_comparison,
		):
			result = runner(potential, configuration, config=config)
			self.assertEqual(result.particle_distances.shape, (1, 3))
			self.assertEqual(result.rms_distance.shape, (3,))
			self.assertEqual(result.max_distance.shape, (3,))
			self.assertEqual(result.particle_distances[0, 0], 0.0)
			self.assertTrue(np.all(result.particle_distances >= 0.0))
			figure, axes = plot_tangent_taylor_trajectory_comparison(result)
			self.assertEqual(axes.shape, (2,))
			figure.canvas.draw()
			plt.close(figure)
			animation = animate_tangent_taylor_particle_evolution(
				result,
				frames=3,
				interval=20,
				repeat=False,
			)
			artists = animation._func(1)
			self.assertEqual(len(artists), 6)
			animation._draw_was_started = True
			plt.close(animation._fig)
			figure, axes = plot_tangent_taylor_component_comparison(result)
			self.assertEqual(axes.shape, (2,))
			figure.canvas.draw()
			plt.close(figure)


if __name__ == "__main__":
	unittest.main()
