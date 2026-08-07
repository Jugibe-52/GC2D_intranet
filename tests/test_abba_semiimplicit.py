"""Contracts for semi-implicit ABBA and its exact tangent workflow."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from classes import (
	GuidingCenterDynamics,
	InitialValueProblem,
	Potential,
	SemiImplicitABBA,
	SimulationRequest,
	SymmetricProjectedABBA,
	TrajectoryGC,
	simulate,
)
from research.symplecticity import (
	central_difference_jacobian,
	gc_physical_symplectic_form,
)
from workflows import (
	RandomPotentialConfig,
	SemiImplicitABBASymplecticityConfig,
	centered_square,
	pi_area_steps,
	run_semiimplicit_abba_symplecticity_study,
)


def deterministic_potential() -> Potential:
	"""Build a compact Hessian-capable nonautonomous field."""
	return Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)


class SemiImplicitABBATests(unittest.TestCase):
	"""Verify exact local tangents and accumulated propagation."""

	def test_exact_tangent_matches_the_projected_map_and_accumulates(self) -> None:
		dynamics = GuidingCenterDynamics(deterministic_potential(), rho=0.05)
		problem = InitialValueProblem(
			dynamics,
			TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.05,
			sample_count=3,
		)
		events = []
		method = SemiImplicitABBA(
			newton_absolute_tolerance=1e-14,
			newton_relative_tolerance=1e-14,
			step_observer=events.append,
		)
		solution = simulate(problem, method, request)

		self.assertEqual(len(events), 2)
		accumulated = np.eye(problem.initial_state.size)
		form = gc_physical_symplectic_form(1)
		for event in events:
			self.assertEqual(event.method_name, "SemiImplicitABBA")
			self.assertIsNotNone(event.state_jacobian)
			assert event.state_jacobian is not None
			numerical = central_difference_jacobian(
				event.map_state,
				event.state_before,
				relative_step=1e-5,
			)
			np.testing.assert_allclose(
				event.state_jacobian,
				numerical,
				rtol=2e-8,
				atol=2e-9,
			)
			defect = event.state_jacobian.T @ form @ event.state_jacobian - form
			self.assertLess(float(np.linalg.norm(defect, ord="fro")), 1e-12)
			accumulated = event.state_jacobian @ accumulated

		np.testing.assert_allclose(
			solution.diagnostics["final_state_jacobian"],
			accumulated,
			rtol=0.0,
			atol=0.0,
		)
		self.assertEqual(
			solution.diagnostics["state_jacobian_kind"],
			"exact_implicit_function",
		)

	def test_physical_trajectory_matches_symmetric_projected_abba(self) -> None:
		dynamics = GuidingCenterDynamics(deterministic_potential(), rho=0.05)
		problem = InitialValueProblem(
			dynamics,
			TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.1),
			max_step=0.025,
			sample_count=5,
		)
		semiimplicit = simulate(problem, SemiImplicitABBA(), request)
		projected = simulate(problem, SymmetricProjectedABBA(), request)
		np.testing.assert_array_equal(semiimplicit.states, projected.states)


class SemiImplicitABBAWorkflowTests(unittest.TestCase):
	"""Verify exact-Jacobian persistence and study summaries."""

	def test_short_study_uses_only_exact_step_jacobians(self) -> None:
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
		config = SemiImplicitABBASymplecticityConfig(
			steps=pi_area_steps(400, 800),
			t_span=(0.0, np.pi / 100),
			save_interval=np.pi / 100,
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			result = run_semiimplicit_abba_symplecticity_study(
				potential,
				area,
				notebook_path=(
					root
					/ "notebooks"
					/ "experiments"
					/ "symplecticity"
					/ "semiimplicit.ipynb"
				),
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
				self.assertEqual(payload["step_jacobian_source"], "exact")
				self.assertIsNone(payload["finite_difference_relative_step"])
				self.assertEqual(
					payload["metadata"]["step_jacobian"],
					"exact_implicit_function_tangent",
				)

		self.assertEqual(result.method_name, "SemiImplicitABBA")
		for step, summary in zip(config.steps, result.summaries(), strict=True):
			state_size = area.initial_state.size
			self.assertEqual(
				result.solutions[step.label]
				.diagnostics["final_state_jacobian"]
				.shape,
				(state_size, state_size),
			)
			self.assertLess(summary.max_local_defect, 1e-12)
			self.assertLess(summary.max_flow_defect, 1e-11)

	def test_config_rejects_a_finite_difference_scale(self) -> None:
		with self.assertRaisesRegex(ValueError, "exact step Jacobians"):
			SemiImplicitABBASymplecticityConfig(
				steps=pi_area_steps(400, 800),
				t_span=(0.0, np.pi / 100),
				save_interval=np.pi / 100,
				finite_difference_relative_step=1e-5,
			)


if __name__ == "__main__":
	unittest.main()
