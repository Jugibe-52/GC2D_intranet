"""Contracts for midpoint ABBA with arithmetic diagonal projection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import TrajectoryGC
from potential import Potential
from simulation import (
	MidpointABBA,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods.abba.midpoint import _midpoint_abba_step
from studies import (
	MidpointABBASymplecticityConfig,
	RandomPotentialConfig,
	centered_square,
	pi_area_steps,
	run_midpoint_abba_symplecticity_study,
)


class _TimeOnlyPlanarDynamics:
	"""Record endpoint evaluations of a state-independent vector field."""

	state_dimension = 2

	def __init__(self) -> None:
		self.vector_field_times: list[float] = []

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


class _LinearRotationDynamics:
	"""Canonical oscillator ``f(y)=J_0 y`` from the midpoint-ABBA note."""

	state_dimension = 2

	def vector_field(self, _t: float, state: np.ndarray) -> np.ndarray:
		"""Rotate one packed planar state by the canonical matrix ``J_0``."""
		particle_count = state.size // self.state_dimension
		return np.concatenate(
			(-state[particle_count:], state[:particle_count])
		)


def _deterministic_gc_dynamics() -> GuidingCenterDynamics:
	"""Build a small reproducible nonautonomous GC field."""
	potential = Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)
	return GuidingCenterDynamics(potential, rho=0.05)


class MidpointABBATests(unittest.TestCase):
	"""Verify stages, geometric limitation, accuracy and observation behavior."""

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
			MidpointABBA(),
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

	def test_quadratic_example_has_the_documented_determinant_defect(self) -> None:
		dynamics = _LinearRotationDynamics()
		step = 0.4
		basis = np.eye(2)
		matrix = np.column_stack(
			[
				_midpoint_abba_step(dynamics, 0.0, column, step).state
				for column in basis.T
			]
		)
		j_zero = np.asarray([[0.0, -1.0], [1.0, 0.0]])
		expected = (
			(1.0 - step**2 / 2.0) * np.eye(2)
			+ (step - step**3 / 8.0) * j_zero
		)
		np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1e-15)
		self.assertAlmostEqual(
			float(np.linalg.det(matrix)) - 1.0,
			step**6 / 64.0,
			places=14,
		)

	def test_method_has_second_order_global_accuracy(self) -> None:
		problem = InitialValueProblem(
			_deterministic_gc_dynamics(),
			TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
		)

		def final_state(step: float) -> np.ndarray:
			"""Return one final state on a fixed integration interval."""
			return simulate(
				problem,
				MidpointABBA(),
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

	def test_observations_and_diagnostics_ignore_shadow_steps(self) -> None:
		problem = InitialValueProblem(
			_deterministic_gc_dynamics(),
			TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
		)
		events = []
		observed = simulate(
			problem,
			MidpointABBA(step_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=11,
			),
		)
		sparse = simulate(
			problem,
			MidpointABBA(),
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=3,
			),
		)
		self.assertEqual(len(events), observed.n_steps)
		self.assertEqual(
			observed.diagnostics["copy_separation_norms"].shape,
			(observed.n_steps,),
		)
		np.testing.assert_array_equal(observed.states[:, -1], sparse.states[:, -1])
		for event in events:
			np.testing.assert_array_equal(
				event.map_state(event.state_before),
				event.state_after,
			)


class MidpointABBASymplecticityStudyTests(unittest.TestCase):
	"""Verify the reusable experiment study and persisted metadata."""

	def test_short_study_returns_defects_and_copy_separation(self) -> None:
		potential_config = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		)
		potential = potential_config.build()
		area = centered_square(
			potential,
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		config = MidpointABBASymplecticityConfig(
			steps=pi_area_steps(40, 80),
			t_span=(0.0, np.pi / 20),
			save_interval=np.pi / 20,
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_midpoint_abba_symplecticity_study(
				potential,
				area,
				notebook_path=(
					root / "notebooks" / "developements" / "midpoint_abba.ipynb"
				),
				config=config,
				project_root=root,
				metadata=potential_config.metadata(),
			)
			metadata_files = sorted(
				result.output_directories[config.steps[0].label].glob(
					"*_metadata_*.json"
				)
			)
			self.assertEqual(len(metadata_files), len(config.steps))
			payload = json.loads(metadata_files[0].read_text(encoding="utf-8"))
			self.assertEqual(payload["metadata"]["projection"], "arithmetic_mean")
			self.assertFalse(payload["metadata"]["projection_is_symplectic"])

		self.assertEqual(result.method_name, "MidpointABBA")
		self.assertEqual(len(result.summaries()), 2)
		self.assertEqual(len(result.convergence_orders()), 1)
		for summary in result.summaries():
			self.assertGreater(summary.max_copy_separation_norm, 0.0)
			self.assertIsNone(summary.max_newton_iterations)
			self.assertTrue(np.isfinite(summary.max_local_defect))
			self.assertTrue(np.isfinite(summary.max_flow_defect))


if __name__ == "__main__":
	unittest.main()
