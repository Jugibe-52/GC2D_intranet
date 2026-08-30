"""Focused contracts for fourth-order ABBA with one exterior projection."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import (
	abba4_implicit_single_projection_step_particle_jacobians,
	central_difference_jacobian,
)
from initial_conditions import GCInitialConfiguration
from simulation import (
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods.abba.order4_implicit_single_projection import (
	_evaluate_single_projection_residual,
	_solve_abba4_single_projection_step,
)
from studies import (
	ABBA4ProjectionComparisonConfig,
	HighPrecisionReferenceConfig,
	RandomPotentialConfig,
	random_gc_configuration,
	run_abba4_projection_comparison_study,
	run_high_precision_reference_trajectory,
)
from visualization import (
	plot_abba4_projection_accuracy,
	plot_abba4_projection_multiplier_scaling,
	plot_abba4_projection_newton_work,
	plot_abba4_projection_order_reduction,
	plot_abba4_projection_runtime,
)


class _TimeDependentRotationDynamics:
	"""Exactly solvable canonical rotation with a polynomial angular rate."""

	state_dimension = 2

	@staticmethod
	def rate(t: float) -> float:
		"""Return the instantaneous angular frequency."""
		return 1.0 + float(t) + float(t) ** 2

	@staticmethod
	def phase(t: float) -> float:
		"""Return an antiderivative of the angular frequency."""
		value = float(t)
		return value + 0.5 * value**2 + value**3 / 3.0

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Apply the time-scaled planar rotation in component-major layout."""
		particle_count = state.size // 2
		x = state[:particle_count]
		y = state[particle_count:]
		frequency = self.rate(t)
		return np.concatenate((-frequency * y, frequency * x))

	def particle_vector_field_jacobians(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return the exact rotation-generator block for every particle."""
		frequency = self.rate(t)
		block = np.asarray([[0.0, -frequency], [frequency, 0.0]])
		return np.broadcast_to(block, (state.size // 2, 2, 2)).copy()


class _NonlinearHamiltonianDynamics:
	"""Smooth canonical field whose exact Jacobian varies along every stage."""

	state_dimension = 2
	_coupling = 0.1

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate the field of ``H=c(t)(x^2+y^2+2 a x^2 y)/2``."""
		particle_count = state.size // 2
		x = state[:particle_count]
		y = state[particle_count:]
		scale = 1.0 + 0.2 * np.sin(float(t))
		a = self._coupling
		return scale * np.concatenate((-(y + a * x**2), x + 2.0 * a * x * y))

	def particle_vector_field_jacobians(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Differentiate the nonlinear field into particle-major blocks."""
		particle_count = state.size // 2
		x = state[:particle_count]
		y = state[particle_count:]
		scale = 1.0 + 0.2 * np.sin(float(t))
		a = self._coupling
		blocks = np.empty((particle_count, 2, 2), dtype=float)
		blocks[:, 0, 0] = -2.0 * a * x
		blocks[:, 0, 1] = -1.0
		blocks[:, 1, 0] = 1.0 + 2.0 * a * y
		blocks[:, 1, 1] = 2.0 * a * x
		return scale * blocks


def _problem(
	dynamics: _TimeDependentRotationDynamics | _NonlinearHamiltonianDynamics,
	*,
	x: np.ndarray,
	y: np.ndarray,
) -> InitialValueProblem:
	"""Build a packed planar initial-value problem for custom test dynamics."""
	return InitialValueProblem(
		dynamics,
		GCInitialConfiguration.from_components(x=x, y=y),
	)


def _dense_particle_blocks(blocks: np.ndarray) -> np.ndarray:
	"""Expand particle-major planar blocks into component-major packed order."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


class ABBA4ImplicitSingleProjectionTests(unittest.TestCase):
	"""Verify order, one solve, exact Newton differentiation, and symmetry."""

	_solver = {
		"absolute_tolerance": 1e-14,
		"relative_tolerance": 1e-14,
		"max_iterations": 16,
		"nonlinear_solver": "newton",
	}

	def test_non_autonomous_problem_has_fourth_order_accuracy(self) -> None:
		start = 0.0
		end = 0.8
		initial = np.asarray([1.0, 0.2])
		dynamics = _TimeDependentRotationDynamics()
		problem = _problem(
			dynamics,
			x=initial[:1],
			y=initial[1:],
		)
		angle = dynamics.phase(end) - dynamics.phase(start)
		exact = np.asarray(
			[
				initial[0] * np.cos(angle) - initial[1] * np.sin(angle),
				initial[0] * np.sin(angle) + initial[1] * np.cos(angle),
			]
		)
		errors = []
		for step in (0.2, 0.1, 0.05):
			solution = simulate(
				problem,
				ABBA4ImplicitSingleProjection(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
				SimulationRequest.uniform(
					t_span=(start, end),
					max_step=step,
					sample_count=2,
				),
			)
			errors.append(float(np.linalg.norm(solution.states[:, -1] - exact)))
		for gain in np.asarray(errors[:-1]) / np.asarray(errors[1:]):
			self.assertGreater(float(gain), 14.0)
			self.assertLess(float(gain), 18.0)

	def test_outer_step_uses_one_nonlinear_solve(self) -> None:
		events = []
		solution = simulate(
			_problem(
				_NonlinearHamiltonianDynamics(),
				x=np.asarray([0.8]),
				y=np.asarray([0.3]),
			),
			ABBA4ImplicitSingleProjection(
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
				step_observer=events.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.2),
				max_step=0.1,
				sample_count=3,
			),
		)
		self.assertEqual(solution.diagnostics["nonlinear_solves_per_step"], 1)
		self.assertEqual(len(events), 2)
		self.assertEqual(len(events[0].substeps), 3)
		self.assertLess(events[0].substeps[1].duration, 0.0)
		self.assertEqual(solution.diagnostics["nonlinear_iterations"].shape, (2,))

		exact = abba4_implicit_single_projection_step_particle_jacobians(
			events[0]
		)[0]
		numerical = central_difference_jacobian(
			events[0].map_state,
			events[0].state_before,
			relative_step=1e-5,
		)
		self.assertLess(
			float(
				np.linalg.norm(exact - numerical, ord="fro")
				/ np.linalg.norm(numerical, ord="fro")
			),
			2e-8,
		)
		self.assertAlmostEqual(float(np.linalg.det(exact)), 1.0, places=13)

	def test_exact_residual_jacobian_matches_centered_differences(self) -> None:
		dynamics = _NonlinearHamiltonianDynamics()
		state = np.asarray([0.8, 1.1, 0.3, -0.2])
		multiplier = np.asarray([2e-4, -3e-4, 1e-4, 4e-4])
		evaluation = _evaluate_single_projection_residual(
			dynamics,
			0.2,
			state,
			0.1,
			multiplier,
		)
		exact = _dense_particle_blocks(evaluation.jacobian)
		numerical = central_difference_jacobian(
			lambda candidate: _evaluate_single_projection_residual(
				dynamics,
				0.2,
				state,
				0.1,
				candidate,
			).residual,
			multiplier,
			relative_step=1e-5,
		)
		relative_error = float(
			np.linalg.norm(exact - numerical, ord="fro")
			/ np.linalg.norm(numerical, ord="fro")
		)
		self.assertLess(relative_error, 2e-8)

	def test_outer_projection_multiplier_has_fifth_order_scaling(self) -> None:
		dynamics = _NonlinearHamiltonianDynamics()
		state = np.asarray([0.8, 0.3])
		norms = []
		for step in (0.2, 0.1, 0.05):
			result = _solve_abba4_single_projection_step(
				dynamics,
				0.0,
				state,
				step,
				**self._solver,
			)
			norms.append(float(np.linalg.norm(result.multiplier, ord=np.inf)))
		for gain in np.asarray(norms[:-1]) / np.asarray(norms[1:]):
			# A fifth-order quantity decreases by 32 under step halving.
			self.assertGreater(float(gain), 28.0)
			self.assertLess(float(gain), 40.0)

	def test_complete_map_is_reversible_and_distinct_from_projecting_each_factor(
		self,
	) -> None:
		dynamics = _NonlinearHamiltonianDynamics()
		state = np.asarray([0.8, 0.3])
		forward = _solve_abba4_single_projection_step(
			dynamics,
			0.2,
			state,
			0.15,
			**self._solver,
		)
		backward = _solve_abba4_single_projection_step(
			dynamics,
			0.35,
			forward.state,
			-0.15,
			**self._solver,
		)
		np.testing.assert_allclose(backward.state, state, rtol=0.0, atol=5e-14)
		np.testing.assert_allclose(
			backward.multiplier,
			-forward.multiplier,
			rtol=0.0,
			atol=5e-14,
		)

		problem = _problem(
			dynamics,
			x=state[:1],
			y=state[1:],
		)
		request = SimulationRequest.uniform(
			t_span=(0.2, 0.35),
			max_step=0.15,
			sample_count=2,
		)
		old = simulate(problem, ABBA4Implicit(), request).states[:, -1]
		new = simulate(
			problem,
			ABBA4ImplicitSingleProjection(),
			request,
		).states[:, -1]
		difference = float(np.linalg.norm(new - old))
		self.assertGreater(difference, 1e-12)
		self.assertLess(difference, 1e-3)

	def test_broyden_matches_the_newton_root(self) -> None:
		problem = _problem(
			_NonlinearHamiltonianDynamics(),
			x=np.asarray([0.8]),
			y=np.asarray([0.3]),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.2),
			max_step=0.1,
			sample_count=3,
		)
		newton = simulate(problem, ABBA4ImplicitSingleProjection(), request)
		broyden = simulate(
			problem,
			ABBA4ImplicitSingleProjection(nonlinear_solver="broyden"),
			request,
		)
		self.assertEqual(broyden.diagnostics["nonlinear_solver"], "broyden")
		np.testing.assert_allclose(
			broyden.states,
			newton.states,
			rtol=0.0,
			atol=2e-12,
		)


class ABBA4SingleProjectionStudyTests(unittest.TestCase):
	"""Exercise the aligned comparison summaries and all dedicated plots."""

	def test_comparison_normalizes_map_work_and_builds_plots(self) -> None:
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
			reference = run_high_precision_reference_trajectory(
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
			).trajectory
			result = run_abba4_projection_comparison_study(
				potential,
				configuration,
				reference,
				config=ABBA4ProjectionComparisonConfig(
					integration_steps=(0.1, 0.05),
					t_span=(0.0, 0.2),
					save_interval=0.1,
					rho=0.05,
					absolute_tolerance=1e-14,
					relative_tolerance=1e-14,
					timing_warmups=0,
					timing_repeats=2,
				),
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
			)

		summaries = result.summaries()
		orders = result.convergence_orders()
		self.assertEqual(len(summaries), 4)
		self.assertEqual(len(orders), 2)
		for row in summaries:
			maps_per_evaluation = (
				3 if row.method_name == "ABBA4ImplicitSingleProjection" else 1
			)
			self.assertEqual(
				row.total_unprojected_abba_map_evaluations,
				maps_per_evaluation * row.total_residual_evaluations,
			)
			self.assertEqual(
				row.total_newton_tangent_abba_map_evaluations,
				maps_per_evaluation * row.total_iterations,
			)
			self.assertLessEqual(row.maximum_residual_to_tolerance, 1.0)

		accuracy_figure, _ = plot_abba4_projection_accuracy(
			summaries,
			reference_floor=result.reference_floor,
		)
		order_figure, order_axes = plot_abba4_projection_order_reduction(orders)
		newton_figure, newton_axes = plot_abba4_projection_newton_work(summaries)
		multiplier_figure, _ = plot_abba4_projection_multiplier_scaling(summaries)
		runtime_figure, _ = plot_abba4_projection_runtime(summaries)
		self.assertEqual(order_axes.shape, (2,))
		self.assertEqual(newton_axes.shape, (2, 2))
		for figure in (
			accuracy_figure,
			order_figure,
			newton_figure,
			multiplier_figure,
			runtime_figure,
		):
			plt.close(figure)


if __name__ == "__main__":
	unittest.main()
