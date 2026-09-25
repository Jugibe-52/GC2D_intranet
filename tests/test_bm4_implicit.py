"""Contracts for the physical reduced Hairer-projected BM4 method."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	BM4Implicit,
	ImplicitBM4IntegrationStep,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from formulations.gc import GCDoubledMaps
from methods.extended.bm4_composition import _advance_composition
from studies import (
	AreaStep,
	BM4ImplicitSymplecticityConfig,
	RandomPotentialConfig,
	centered_square,
	run_bm4_implicit_symplecticity_study,
)


def _potential(*, amplitude: float = 0.08) -> Potential:
	"""Return the compact reproducible field used by projected BM4 tests."""
	return Potential.random(
		A=amplitude,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=3,
	)


def _problem() -> InitialValueProblem:
	"""Build one planar guiding-centre initial-value problem."""
	potential = _potential()
	return InitialValueProblem(
		GuidingCenterDynamics(potential, rho=0.05),
		GCInitialConfiguration(np.asarray([1.0, 1.2])),
	)


class BM4ImplicitMethodTests(unittest.TestCase):
	"""Verify state space, order, observations and parameter validation."""

	def test_energy_reuses_converged_shears_without_extra_spatial_work(self) -> None:
		"""Compare traced quadrature with an independent energy-augmented replay."""
		for particles, coupling in ((1, 0.0), (3, 0.4)):
			problem = InitialValueProblem(
				GuidingCenterDynamics(_potential(), rho=0.05),
				GCInitialConfiguration.from_components(
					x=np.linspace(1.0, 1.2, particles),
					y=np.linspace(1.2, 1.4, particles),
				),
			)
			request = SimulationRequest.uniform(t_span=(0.3, 0.5), max_step=0.2, sample_count=2)
			for solver, jacobian in (('newton', 'analytic'), ('newton', 'finite_difference'), ('broyden', 'analytic')):
				for step in (0.2, -0.2):
					with self.subTest(particles=particles, coupling=coupling, solver=solver, jacobian=jacobian, step=step):
						results = []
						field_counts = []
						for tracking in (False, True):
							run = BM4Implicit(
								track_energy=tracking, coupling_frequency=coupling,
								nonlinear_solver=solver, newton_jacobian_method=jacobian,
								newton_absolute_tolerance=1e-14, newton_relative_tolerance=1e-14,
							).new_run(problem, request)
							with patch.object(problem.dynamics, 'vector_field', wraps=problem.dynamics.vector_field) as vector_field, \
							     patch.object(problem.dynamics, 'extended_momentum_derivative', wraps=problem.dynamics.extended_momentum_derivative) as energy_rate:
								result = run.advance(0.3, run.initial_state, step)
								field_counts.append(vector_field.call_count)
								self.assertEqual(energy_rate.call_count, 24 if tracking else 0)
							self.assertEqual(len(result.details.energy_points), 24 if tracking else 0)
							results.append(result)
						plain, tracked = results
						self.assertEqual(field_counts[0], field_counts[1])
						np.testing.assert_array_equal(plain.details.state, tracked.details.state)
						np.testing.assert_array_equal(plain.details.multiplier, tracked.details.multiplier)
						self.assertEqual(plain.statistics, tracked.statistics)
						self.assertGreater(tracked.details.iterations, 0)
						prepared = GCDoubledMaps(problem, coupling, track_energy=True)
						replayed = _advance_composition(
							prepared, 0.3,
							np.concatenate((tracked.details.internal_input, np.zeros(particles))), step,
							step_index=0, stage_observer=None,
							formulation_name='energy_reference', method_name='BM4Implicit',
						)
						np.testing.assert_array_equal(
							run.state_formulation.momentum(tracked.state), replayed[-particles:] / 2.0,
						)

	def test_one_cycle_uses_the_reduced_physical_hairer_projection(self) -> None:
		problem = _problem()
		events: list[ImplicitBM4IntegrationStep] = []
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.2),
			max_step=0.2,
			sample_count=2,
		)
		solution = simulate(
			problem,
			BM4Implicit(
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
				step_observer=events.append,
			),
			request,
		)

		self.assertEqual(solution.states.shape, (2, 2))
		self.assertEqual(len(events), 1)
		event = events[0]
		self.assertEqual(event.state_before.shape, (2,))
		self.assertEqual(event.state_after.shape, (2,))
		self.assertEqual(event.multiplier.shape, (2,))
		self.assertEqual(len(event.base_stages), 12)
		self.assertTrue(
			all(stage.state_before.shape == (4,) for stage in event.base_stages)
		)
		internal_input = np.concatenate(
			(
				event.state_before + event.multiplier,
				event.state_before - event.multiplier,
			)
		)
		np.testing.assert_array_equal(
			event.base_stages[0].state_before,
			internal_input,
		)
		mapped = event.base_stages[-1].state_after
		corrected_first = mapped[:2] + event.multiplier
		corrected_second = mapped[2:] - event.multiplier
		np.testing.assert_allclose(
			corrected_first,
			corrected_second,
			rtol=0.0,
			atol=3e-14,
		)
		np.testing.assert_allclose(
			corrected_first,
			event.state_after,
			rtol=0.0,
			atol=2e-14,
		)
		self.assertEqual(
			solution.diagnostics["projection_solver_formulation"],
			"bm4_implicit_reduced",
		)

	def test_reduced_method_has_fourth_order_global_accuracy(self) -> None:
		problem = _problem()

		def final_state(step: float) -> np.ndarray:
			return simulate(
				problem,
				BM4Implicit(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 0.4),
					max_step=step,
					sample_count=2,
				),
			).states[:, -1]

		reference = final_state(0.00625)
		coarse_error = float(np.linalg.norm(final_state(0.1) - reference))
		fine_error = float(np.linalg.norm(final_state(0.05) - reference))
		self.assertGreater(coarse_error / fine_error, 15.0)
		self.assertLess(coarse_error / fine_error, 17.0)

	def test_analytic_and_finite_difference_newton_jacobians_agree(self) -> None:
		problem = InitialValueProblem(
			GuidingCenterDynamics(_potential(), rho=0.05),
			GCInitialConfiguration(np.asarray([1.0, 1.2, 1.1, 0.9])),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.05,
			sample_count=3,
		)
		common = {
			"newton_absolute_tolerance": 1e-14,
			"newton_relative_tolerance": 1e-14,
		}
		analytic = simulate(
			problem,
			BM4Implicit(**common, newton_jacobian_method="analytic"),
			request,
		)
		finite_difference = simulate(
			problem,
			BM4Implicit(**common, newton_jacobian_method="finite_difference"),
			request,
		)

		np.testing.assert_allclose(
			analytic.states,
			finite_difference.states,
			rtol=0.0,
			atol=2e-14,
		)
		self.assertEqual(analytic.diagnostics["newton_jacobian_method"], "analytic")
		self.assertEqual(
			finite_difference.diagnostics["newton_jacobian_method"],
			"finite_difference",
		)

	def test_step_observer_receives_only_main_grid_steps(self) -> None:
		events = []
		solution = simulate(
			_problem(),
			BM4Implicit(step_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=11,
			),
		)
		self.assertEqual(len(events), solution.n_steps)
		for event in events:
			np.testing.assert_allclose(
				event.map_state(event.state_before),
				event.state_after,
				rtol=0.0,
				atol=0.0,
			)

	def test_invalid_solver_parameters_fail_during_configuration(self) -> None:
		with self.assertRaises(ValueError):
			BM4Implicit(coupling_frequency=-1.0)
		with self.assertRaises(ValueError):
			BM4Implicit(newton_absolute_tolerance=0.0)
		with self.assertRaises(ValueError):
			BM4Implicit(newton_max_iterations=0)
		with self.assertRaises(ValueError):
			BM4Implicit(newton_jacobian_relative_step=np.inf)
		with self.assertRaises(ValueError):
			BM4Implicit(newton_jacobian_method="complex_step")  # type: ignore[arg-type]


class BM4ImplicitStudyTests(unittest.TestCase):
	"""Verify the reusable physical projected-BM4 symplecticity study."""

	def test_short_study_returns_physical_symplecticity_diagnostics(self) -> None:
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
		config = BM4ImplicitSymplecticityConfig(
			steps=(AreaStep(label="h=0.05", value=0.05),),
			t_span=(0.0, 0.05),
			save_interval=0.05,
			chunk_size=2,
			newton_absolute_tolerance=1e-14,
			newton_relative_tolerance=1e-14,
		)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_bm4_implicit_symplecticity_study(
				potential,
				area,
				notebook_path=root / "notebooks" / "developements" / "bm4.ipynb",
				config=config,
				project_root=root,
			)

		self.assertEqual(result.method_name, "BM4Implicit")
		self.assertEqual(tuple(result.solutions), ("h=0.05",))
		self.assertEqual(result.jacobian_method, "finite_difference")
		self.assertEqual(len(result.summaries()), 1)
		summary = result.summaries()[0]
		self.assertLess(summary.max_local_defect, 1e-8)
		self.assertLess(summary.max_flow_defect, 1e-8)
		self.assertLess(summary.max_determinant_error, 1e-8)
		self.assertIsNotNone(summary.max_newton_iterations)


if __name__ == "__main__":
	unittest.main()
