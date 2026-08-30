"""Contracts for fourth-order reduced implicit ABBA and its studies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import (
	abba4_implicit_1_step_particle_jacobians,
	central_difference_jacobian,
)
from initial_conditions import GCInitialConfiguration
from simulation import (
	ABBA4Implicit1,
	ImplicitABBA4IntegrationStep,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods.abba.order4_implicit_1 import (
	_ABBA4_COEFFICIENTS,
	_solve_abba4_step,
)
from studies import (
	ABBA4Implicit1AccuracyConfig,
	AreaStep,
	HighPrecisionReferenceConfig,
	RandomPotentialConfig,
	TrajectorySymplecticityConfig,
	random_gc_configuration,
	run_abba4_implicit_1_accuracy_study,
	run_abba4_implicit_1_trajectory_symplecticity_study,
	run_high_precision_reference_trajectory,
)
from visualization import (
	plot_single_method_accuracy_refinement,
	plot_trajectory_accuracy_over_time,
)


class _LinearRotationDynamics:
	"""Canonical oscillator with a constant exact particle Jacobian."""

	state_dimension = 2

	def vector_field(self, _t: float, state: np.ndarray) -> np.ndarray:
		"""Apply ``(x, y)' = (-y, x)`` in component-major layout."""
		particle_count = state.size // 2
		return np.concatenate((-state[particle_count:], state[:particle_count]))

	def particle_vector_field_jacobians(
		self,
		_t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return the constant canonical rotation generator per particle."""
		particle_count = state.size // 2
		return np.broadcast_to(
			np.asarray([[0.0, -1.0], [1.0, 0.0]]),
			(particle_count, 2, 2),
		).copy()


class _PolynomialTimeDynamics:
	"""Non-autonomous canonical field ``(x, y)' = (t**4, 0)``."""

	state_dimension = 2

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Return the time polynomial in component-major layout."""
		particle_count = state.size // 2
		return np.concatenate(
			(
				np.full(particle_count, float(t) ** 4),
				np.zeros(particle_count),
			)
		)

	def particle_vector_field_jacobians(
		self,
		_t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return the zero spatial Jacobian for every particle."""
		return np.zeros((state.size // 2, 2, 2), dtype=float)


def _rotation_problem() -> InitialValueProblem:
	"""Return one oscillator trajectory starting from ``(1, 0)``."""
	configuration = GCInitialConfiguration.from_components(
		x=np.asarray([1.0]),
		y=np.asarray([0.0]),
	)
	return InitialValueProblem(_LinearRotationDynamics(), configuration)


def _dense(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent planar blocks into component-major packed layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


class ABBA4Implicit1MethodTests(unittest.TestCase):
	"""Verify signed composition, designed order, reversibility, and tangent."""

	def test_observation_contains_three_continuous_signed_substeps(self) -> None:
		self.assertLess(_ABBA4_COEFFICIENTS[1], 0.0)
		self.assertAlmostEqual(float(np.sum(_ABBA4_COEFFICIENTS)), 1.0)
		self.assertAlmostEqual(float(np.sum(_ABBA4_COEFFICIENTS**3)), 0.0)
		events = []
		solution = simulate(
			_rotation_problem(),
			ABBA4Implicit1(
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
				step_observer=events.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.2),
				max_step=0.1,
				sample_count=5,
			),
		)
		self.assertEqual(len(events), 2)
		step = events[0]
		self.assertIsInstance(step, ImplicitABBA4IntegrationStep)
		self.assertEqual(len(step.substeps), 3)
		self.assertLess(step.substeps[1].duration, 0.0)
		np.testing.assert_allclose(
			[substep.duration for substep in step.substeps],
			_ABBA4_COEFFICIENTS * step.duration,
			rtol=0.0,
			atol=1e-16,
		)
		expected_starts = step.start_time + np.concatenate(
			(
				np.asarray([0.0]),
				np.cumsum(_ABBA4_COEFFICIENTS[:-1]) * step.duration,
			)
		)
		np.testing.assert_allclose(
			[substep.start_time for substep in step.substeps],
			expected_starts,
			rtol=0.0,
			atol=2e-16,
		)
		np.testing.assert_array_equal(step.substeps[0].state_before, step.state_before)
		for first, second in zip(step.substeps, step.substeps[1:]):
			np.testing.assert_array_equal(first.state_after, second.state_before)
			self.assertFalse(np.shares_memory(first.multiplier, second.multiplier))
		np.testing.assert_array_equal(step.substeps[-1].state_after, step.state_after)
		np.testing.assert_allclose(
			step.map_state(step.state_before),
			step.state_after,
			rtol=0.0,
			atol=2e-15,
		)
		self.assertEqual(solution.diagnostics["nonlinear_solves_per_step"], 3)
		self.assertEqual(
			solution.diagnostics["substep_nonlinear_iterations"].shape,
			(2, 3),
		)
		np.testing.assert_array_equal(
			solution.diagnostics["nonlinear_iterations"],
			np.sum(
				solution.diagnostics["substep_nonlinear_iterations"],
				axis=1,
			),
		)

	def test_method_is_fourth_order_and_reversible(self) -> None:
		problem = _rotation_problem()
		exact = np.asarray([np.cos(0.8), np.sin(0.8)])
		errors = []
		for step in (0.2, 0.1, 0.05):
			solution = simulate(
				problem,
				ABBA4Implicit1(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 0.8),
					max_step=step,
					sample_count=2,
				),
			)
			errors.append(float(np.linalg.norm(solution.states[:, -1] - exact)))
		for gain in (errors[0] / errors[1], errors[1] / errors[2]):
			self.assertGreater(gain, 15.0)
			self.assertLess(gain, 17.0)

		state = np.asarray([1.0, 0.2])
		solver = {
			"absolute_tolerance": 1e-14,
			"relative_tolerance": 1e-14,
			"max_iterations": 12,
			"nonlinear_solver": "newton",
		}
		forward = _solve_abba4_step(
			_LinearRotationDynamics(),
			0.3,
			state,
			0.2,
			**solver,
		)
		backward = _solve_abba4_step(
			_LinearRotationDynamics(),
			0.5,
			forward.state,
			-0.2,
			**solver,
		)
		np.testing.assert_allclose(backward.state, state, rtol=0.0, atol=2e-15)

	def test_fourth_order_composition_uses_signed_non_autonomous_times(self) -> None:
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([0.0]),
			y=np.asarray([0.0]),
		)
		problem = InitialValueProblem(_PolynomialTimeDynamics(), configuration)
		errors = []
		for step in (0.2, 0.1, 0.05, 0.025):
			solution = simulate(
				problem,
				ABBA4Implicit1(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=step,
					sample_count=2,
				),
			)
			errors.append(abs(float(solution.states[0, -1]) - 0.2))
		for gain in np.asarray(errors[:-1]) / np.asarray(errors[1:]):
			self.assertGreater(float(gain), 15.9)
			self.assertLess(float(gain), 16.1)

	def test_exact_composed_jacobian_matches_centered_difference(self) -> None:
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0, 1.4]),
			y=np.asarray([0.2, 0.6]),
		)
		problem = InitialValueProblem(_LinearRotationDynamics(), configuration)
		events = []
		simulate(
			problem,
			ABBA4Implicit1(
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
				step_observer=events.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.1),
				max_step=0.1,
				sample_count=2,
			),
		)
		exact = _dense(abba4_implicit_1_step_particle_jacobians(events[0]))
		numerical = central_difference_jacobian(
			events[0].map_state,
			events[0].state_before,
			relative_step=2e-5,
		)
		relative_error = float(
			np.linalg.norm(exact - numerical, ord="fro")
			/ np.linalg.norm(numerical, ord="fro")
		)
		self.assertLess(relative_error, 2e-8)
		for block in abba4_implicit_1_step_particle_jacobians(events[0]):
			self.assertAlmostEqual(float(np.linalg.det(block)), 1.0, places=13)
		malformed = replace(
			events[0],
			composition_coefficients=np.asarray([1.0, -1.0, 1.0]),
		)
		with self.assertRaisesRegex(ValueError, "coefficients"):
			abba4_implicit_1_step_particle_jacobians(malformed)

	def test_broyden_path_matches_the_newton_root(self) -> None:
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.4),
			max_step=0.1,
			sample_count=5,
		)
		newton = simulate(_rotation_problem(), ABBA4Implicit1(), request)
		broyden = simulate(
			_rotation_problem(),
			ABBA4Implicit1(nonlinear_solver="broyden"),
			request,
		)
		self.assertEqual(broyden.diagnostics["nonlinear_solver"], "broyden")
		np.testing.assert_allclose(
			broyden.states,
			newton.states,
			rtol=0.0,
			atol=2e-12,
		)


class ABBA4Implicit1StudyTests(unittest.TestCase):
	"""Exercise focused reference-accuracy and symplecticity runners."""

	def test_short_accuracy_and_symplecticity_studies(self) -> None:
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
			particle_count=2,
			seed=41,
		)
		initial_metadata = {
			"particle_count": 2,
			"seed": 41,
			"sampling": "uniform_full_periodic_cell",
		}
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			(root / "pyproject.toml").write_text("[project]\nname='test'\n")
			reference_result = run_high_precision_reference_trajectory(
				potential,
				configuration,
				notebook_path=(
					root / "notebooks/developements/accuracy/reference.ipynb"
				),
				config=HighPrecisionReferenceConfig(
					t_span=(0.0, 0.2),
					save_interval=0.025,
					rho=0.05,
					relative_tolerance=1e-11,
					absolute_tolerance=1e-13,
					maximum_step=0.005,
					audit_relative_tolerance=1e-11,
					audit_absolute_tolerance=1e-13,
					audit_maximum_step=0.0025,
				),
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
				project_root=root,
			)
			accuracy = run_abba4_implicit_1_accuracy_study(
				potential,
				configuration,
				reference_result.trajectory,
				config=ABBA4Implicit1AccuracyConfig(
					integration_steps=(0.1, 0.05),
					t_span=(0.0, 0.2),
					save_interval=0.1,
					rho=0.05,
					absolute_tolerance=1e-14,
					relative_tolerance=1e-14,
				),
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
			)
			self.assertEqual(len(accuracy.summaries()), 2)
			self.assertEqual(len(accuracy.convergence_orders()), 1)
			for values in accuracy.series.values():
				np.testing.assert_array_equal(values.distances[:, 0], 0.0)
			figure, axis = plot_single_method_accuracy_refinement(
				accuracy.summaries(),
				expected_order=4.0,
				reference_floor=accuracy.reference_floor,
			)
			time_figure, time_axes = plot_trajectory_accuracy_over_time(
				accuracy.times,
				{"h=0.05": accuracy.series[0.05]},
				reference_floor=accuracy.reference_floor,
			)
			self.assertEqual(len(axis.lines), 3)
			self.assertEqual(time_axes.shape, (2,))
			plt.close(figure)
			plt.close(time_figure)

			symplecticity = run_abba4_implicit_1_trajectory_symplecticity_study(
				potential,
				configuration,
				notebook_path=(
					root / "notebooks/developements/symplecticity/abba4.ipynb"
				),
				project_root=root,
				config=TrajectorySymplecticityConfig(
					steps=(AreaStep("h=0.1", 0.1), AreaStep("h=0.05", 0.05)),
					t_span=(0.0, 0.2),
					save_interval=0.1,
					rho=0.05,
					chunk_size=4,
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
			)
			self.assertEqual([row.step_count for row in symplecticity.summaries()], [2, 4])
			for row in symplecticity.summaries():
				self.assertLess(row.max_mean_local_defect, 1e-12)
				self.assertLess(row.max_mean_accumulated_defect, 1e-11)


if __name__ == "__main__":
	unittest.main()
