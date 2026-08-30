"""Contracts for ABBA integration with Hairer's symmetric projection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from dynamics import (
	FullCyclotronDynamics,
	GuidingCenterDynamics,
)
from initial_conditions import (
	FCInitialConfiguration,
	TrajectoryGC,
)
from potential import Potential
from simulation import (
	ImplicitABBA1,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods.abba._projection import (
	_evaluate_residual,
	_solve_projected_step,
)
from diagnostics.symplecticity import (
	central_difference_jacobian,
	gc_physical_symplectic_form,
)
from studies import (
	ABBASymplecticityConfig,
	RandomPotentialConfig,
	centered_square,
	pi_area_steps,
	run_abba_symplecticity_study,
)


def deterministic_potential(*, interpolation_order: int = 5) -> Potential:
	"""Build the small reproducible nonautonomous field used by ABBA tests."""
	return Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=interpolation_order,
	)


def gc_dynamics(*, interpolation_order: int = 5) -> GuidingCenterDynamics:
	"""Build GC dynamics with a Hessian-capable effective potential."""
	return GuidingCenterDynamics(
		deterministic_potential(interpolation_order=interpolation_order),
		rho=0.05,
	)


def dense_component_major_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Expand batched per-particle matrices into the packed GC layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count))
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


class _TimeOnlyPlanarDynamics:
	"""Record ABBA endpoint evaluations for a state-independent vector field."""

	state_dimension = 2

	def __init__(self) -> None:
		self.vector_field_times: list[float] = []
		self.jacobian_times: list[float] = []

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Return ``(t, t**2)`` for every packed particle."""
		self.vector_field_times.append(float(t))
		particle_count = state.size // self.state_dimension
		return np.concatenate(
			(
				np.full(particle_count, t),
				np.full(particle_count, t**2),
			)
		)

	def particle_vector_field_jacobians(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return the zero spatial Jacobian and record its evaluation time."""
		self.jacobian_times.append(float(t))
		particle_count = state.size // self.state_dimension
		return np.zeros((particle_count, 2, 2))


class GuidingCenterJacobianTests(unittest.TestCase):
	"""Verify the exact Hessian-based GC vector-field Jacobian."""

	def test_batched_jacobian_matches_centered_differences(self) -> None:
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.4, 1.2, 1.6])
		time = 0.3
		blocks = dynamics.particle_vector_field_jacobians(time, state)
		self.assertEqual(blocks.shape, (2, 2, 2))
		dense = dense_component_major_jacobian(blocks)
		numerical = central_difference_jacobian(
			lambda candidate: dynamics.vector_field(time, candidate),
			state,
			relative_step=1e-5,
		)
		np.testing.assert_allclose(dense, numerical, rtol=2e-7, atol=2e-8)

		# Every per-particle W is Hamiltonian for the document's J_0 convention.
		j_zero = np.asarray([[0.0, -1.0], [1.0, 0.0]])
		for block in blocks:
			np.testing.assert_allclose(
				block.T @ j_zero + j_zero @ block,
				0.0,
				atol=1e-13,
			)

	def test_quadratic_spline_is_rejected_with_a_clear_message(self) -> None:
		dynamics = gc_dynamics(interpolation_order=2)
		with self.assertRaisesRegex(ValueError, "interpolation_order >= 3"):
			dynamics.particle_vector_field_jacobians(
				0.0,
				np.asarray([1.0, 1.2]),
			)


class ImplicitABBA1Tests(unittest.TestCase):
	"""Verify the ABBA stages, exact Newton solve and method integration."""

	def test_reduced_residual_jacobian_matches_centered_differences(self) -> None:
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.4, 1.2, 1.6])
		multiplier = np.asarray([1e-3, -2e-3, 3e-3, -1e-3])
		time = 0.2
		step = 0.05
		evaluation = _evaluate_residual(
			dynamics,
			time,
			state,
			step,
			multiplier,
		)
		dense = dense_component_major_jacobian(evaluation.jacobian)
		numerical = central_difference_jacobian(
			lambda candidate: _evaluate_residual(
				dynamics,
				time,
				state,
				step,
				candidate,
			).residual,
			multiplier,
			relative_step=1e-5,
		)
		np.testing.assert_allclose(dense, numerical, rtol=2e-7, atol=2e-8)

	def test_non_autonomous_stages_use_both_step_endpoints(self) -> None:
		dynamics = _TimeOnlyPlanarDynamics()
		initial_state = np.asarray([1.0, 1.2])
		start = 0.2
		step = 0.1
		solution = simulate(
			InitialValueProblem(
				dynamics,
				TrajectoryGC(initial_state, rho=0.05),
			),
			ImplicitABBA1(step_observer=lambda _step: None),
			SimulationRequest.uniform(
				t_span=(start, start + step),
				max_step=step,
				sample_count=2,
			),
		)
		expected_increment = step / 2 * np.asarray(
			[start + start + step, start**2 + (start + step) ** 2]
		)
		np.testing.assert_allclose(
			solution.states[:, -1],
			initial_state + expected_increment,
			rtol=0.0,
			atol=1e-15,
		)
		self.assertEqual(
			dynamics.vector_field_times,
			[start, start, start + step, start + step],
		)
		# The state-independent field converges without a Newton correction, so
		# only the capability preflight requires a spatial Jacobian.
		self.assertEqual(dynamics.jacobian_times, [start])

		# Without an observer, only the capability preflight needs a Hessian.
		unobserved_dynamics = _TimeOnlyPlanarDynamics()
		simulate(
			InitialValueProblem(
				unobserved_dynamics,
				TrajectoryGC(initial_state, rho=0.05),
			),
			ImplicitABBA1(),
			SimulationRequest.uniform(
				t_span=(start, start + step),
				max_step=step,
				sample_count=2,
			),
		)
		self.assertEqual(unobserved_dynamics.jacobian_times, [start])

	def test_projected_step_is_reversible_and_symplectic(self) -> None:
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.2])
		time = 0.3
		step = 0.07
		solver = {
			"absolute_tolerance": 1e-14,
			"relative_tolerance": 1e-14,
			"max_iterations": 12,
		}
		forward = _solve_projected_step(
			dynamics,
			time,
			state,
			step,
			**solver,
		)
		backward = _solve_projected_step(
			dynamics,
			time + step,
			forward.state,
			-step,
			**solver,
		)
		np.testing.assert_allclose(backward.state, state, rtol=0.0, atol=2e-14)
		np.testing.assert_allclose(
			backward.multiplier,
			-forward.multiplier,
			rtol=0.0,
			atol=2e-14,
		)

		jacobian = central_difference_jacobian(
			lambda candidate: _solve_projected_step(
				dynamics,
				time,
				candidate,
				step,
				**solver,
			).state,
			state,
			relative_step=1e-5,
		)
		form = gc_physical_symplectic_form(1)
		defect = jacobian.T @ form @ jacobian - form
		self.assertLess(float(np.linalg.norm(defect, ord="fro")), 1e-8)
		self.assertLess(abs(float(np.linalg.det(jacobian)) - 1.0), 1e-8)

	def test_numerical_tangent_preserves_two_particle_layout(self) -> None:
		"""Keep independent-particle blocks in component-major order."""
		dynamics = gc_dynamics()
		state = np.asarray([1.0, 1.4, 1.2, 1.6])
		time = 0.2
		step = 0.05
		solver = {
			"absolute_tolerance": 1e-14,
			"relative_tolerance": 1e-14,
			"max_iterations": 12,
		}
		numerical = central_difference_jacobian(
			lambda candidate: _solve_projected_step(
				dynamics,
				time,
				candidate,
				step,
				**solver,
			).state,
			state,
			relative_step=1e-5,
		)
		cross_particle = numerical[np.ix_((0, 2), (1, 3))]
		np.testing.assert_array_equal(cross_particle, 0.0)

	def test_method_has_second_order_global_accuracy(self) -> None:
		dynamics = gc_dynamics()
		source = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		problem = InitialValueProblem(dynamics, source)

		def final_state(step: float) -> np.ndarray:
			return simulate(
				problem,
				ImplicitABBA1(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 0.4),
					max_step=step,
					sample_count=2,
				),
			).states[:, -1]

		reference = final_state(5e-4)
		coarse_error = float(np.linalg.norm(final_state(0.1) - reference))
		fine_error = float(np.linalg.norm(final_state(0.05) - reference))
		self.assertGreater(coarse_error / fine_error, 3.8)
		self.assertLess(coarse_error / fine_error, 4.2)

	def test_observation_and_diagnostics_ignore_shadow_steps(self) -> None:
		dynamics = gc_dynamics()
		source = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		problem = InitialValueProblem(dynamics, source)
		events = []
		observed = simulate(
			problem,
			ImplicitABBA1(step_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=11,
			),
		)
		sparse = simulate(
			problem,
			ImplicitABBA1(),
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=3,
			),
		)
		self.assertEqual(observed.n_steps, 3)
		self.assertEqual(len(events), observed.n_steps)
		self.assertEqual(observed.diagnostics["newton_iterations"].shape, (3,))
		np.testing.assert_array_equal(observed.states[:, -1], sparse.states[:, -1])
		for event in events:
			np.testing.assert_allclose(
				event.map_state(event.state_before),
				event.state_after,
				rtol=0.0,
				atol=0.0,
			)

	def test_zero_field_is_identity_and_invalid_inputs_fail_early(self) -> None:
		zero_potential = Potential.random(
			A=0.0,
			M=2,
			nx=8,
			ny=8,
			seed=27,
			interpolation_order=3,
		)
		source = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		zero_problem = InitialValueProblem(
			GuidingCenterDynamics(zero_potential, rho=0.05),
			source,
		)
		solution = simulate(
			zero_problem,
			ImplicitABBA1(),
			SimulationRequest.uniform(
				t_span=(0.0, 0.1),
				max_step=0.02,
				sample_count=3,
			),
		)
		np.testing.assert_array_equal(
			solution.states,
			np.repeat(source.initial_state[:, None], 3, axis=1),
		)
		np.testing.assert_array_equal(solution.diagnostics["newton_iterations"], 0)

		with self.assertRaises(ValueError):
			ImplicitABBA1(newton_absolute_tolerance=True)
		with self.assertRaises(ValueError):
			ImplicitABBA1(newton_relative_tolerance=0.0)
		with self.assertRaises(ValueError):
			ImplicitABBA1(newton_max_iterations=0)

		fc_source = FCInitialConfiguration(
			np.asarray([1.0, 1.2, 0.4, -0.3]),
		)
		fc_problem = InitialValueProblem(
			FullCyclotronDynamics(
				deterministic_potential(),
				rho=0.2,
				eta=0.1,
			),
			fc_source,
		)
		with self.assertRaisesRegex(TypeError, "GuidingCenterJacobianSystem"):
			simulate(
				fc_problem,
				ImplicitABBA1(),
				SimulationRequest.uniform(
					t_span=(0.0, 0.01),
					max_step=0.01,
					sample_count=2,
				),
			)

	def test_nonconvergence_reports_time_step_and_residual(self) -> None:
		problem = InitialValueProblem(
			gc_dynamics(),
			TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
		)
		with self.assertRaisesRegex(
			RuntimeError,
			"t=.*step=.*residual norm",
		):
			simulate(
				problem,
				ImplicitABBA1(
					newton_absolute_tolerance=1e-30,
					newton_relative_tolerance=1e-30,
					newton_max_iterations=1,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 0.05),
					max_step=0.05,
					sample_count=2,
				),
			)


class ABBASymplecticityStudyTests(unittest.TestCase):
	"""Verify the reusable physical-flow study used by the development notebook."""

	def test_short_study_returns_solver_and_symplecticity_diagnostics(self) -> None:
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
		config = ABBASymplecticityConfig(
			steps=pi_area_steps(400, 800),
			t_span=(0.0, np.pi / 100),
			save_interval=np.pi / 100,
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_abba_symplecticity_study(
				potential,
				area,
				notebook_path=root / "notebooks" / "developements" / "abba.ipynb",
				config=config,
				project_root=root,
			)
			metadata_files = sorted(
				result.output_directories[config.steps[0].label].glob(
					"*_metadata_*.json"
				)
			)
			self.assertEqual(len(metadata_files), len(config.steps))
			for metadata_path in metadata_files:
				payload = json.loads(metadata_path.read_text(encoding="utf-8"))
				self.assertEqual(
					payload["metadata"]["step_jacobian"],
					"centered_difference_of_emitted_solver_map",
				)

		self.assertEqual(result.method_name, "ImplicitABBA1")
		self.assertEqual(len(result.summaries()), 2)
		self.assertEqual(len(result.projection_multiplier_orders()), 1)
		for summary in result.summaries():
			self.assertIsNotNone(summary.max_newton_iterations)
			self.assertIsNotNone(summary.max_newton_residual_norm)
			# Complete finite-tolerance steps are differentiated numerically, so the
			# measured structural defect includes the centered-difference floor.
			self.assertLess(summary.max_local_defect, 1e-9)
			self.assertLess(summary.max_flow_defect, 1e-8)
		figure, axes = result.plot_solver_diagnostics()
		self.assertEqual(axes.shape, (2,))
		plt.close(figure)


if __name__ == "__main__":
	unittest.main()
