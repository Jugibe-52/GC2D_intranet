"""Contracts for diagnostic Jacobians of implicit ABBA steps."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnostics.symplecticity import (
	GCAreaSymplecticityObserver,
	central_difference_jacobian,
	gc_physical_symplectic_form,
	implicit_function_step_jacobian,
	stage_increment_step_jacobian,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import Area, TrajectoryGC
from potential import Potential
from simulation import (
	ImplicitABBA1,
	ImplicitABBA2,
	InitialValueProblem,
	IntegrationStep,
	SimulationRequest,
	simulate,
)
from studies import (
	DEFAULT_IMPLICIT_ABBA_OBSERVERS,
	ImplicitABBASymplecticityConfig,
	RandomPotentialConfig,
	centered_square,
	run_implicit_abba_symplecticity_study,
)
from studies.area_comparison import AreaStep


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


class _TimeOnlyPlanarDynamics:
	"""Record the exact endpoint times used by analytic diagnostics."""

	state_dimension = 2

	def __init__(self) -> None:
		self.jacobian_times: list[float] = []

	def vector_field(self, time: float, state: np.ndarray) -> np.ndarray:
		"""Return a state-independent nonautonomous planar field."""
		particle_count = state.size // self.state_dimension
		return np.concatenate(
			(
				np.full(particle_count, time),
				np.full(particle_count, time**2),
			)
		)

	def particle_vector_field_jacobians(
		self,
		time: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Record ``time`` and return the zero spatial Jacobian."""
		self.jacobian_times.append(float(time))
		particle_count = state.size // self.state_dimension
		return np.zeros((particle_count, 2, 2))


class ImplicitABBAJacobianTests(unittest.TestCase):
	"""Verify both analytic diagnostic factorizations for both solvers."""

	def test_analytic_jacobians_match_each_other_and_finite_differences(self) -> None:
		dynamics = GuidingCenterDynamics(deterministic_potential(), rho=0.05)
		initial_state = np.asarray([1.0, 1.4, 1.2, 1.6])
		problem = InitialValueProblem(
			dynamics,
			TrajectoryGC(initial_state, rho=0.05),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.05),
			max_step=0.05,
			sample_count=2,
		)
		form = gc_physical_symplectic_form(2)

		for method_type in (ImplicitABBA1, ImplicitABBA2):
			events = []
			solution = simulate(
				problem,
				method_type(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
					step_observer=events.append,
				),
				request,
			)
			self.assertEqual(len(events), 1)
			step = events[0]
			implicit_function = implicit_function_step_jacobian(step)
			stage_increment = stage_increment_step_jacobian(step)
			finite_difference = central_difference_jacobian(
				step.map_state,
				step.state_before,
				relative_step=1e-5,
			)
			np.testing.assert_allclose(
				stage_increment,
				implicit_function,
				rtol=2e-15,
				atol=2e-15,
			)
			np.testing.assert_allclose(
				implicit_function,
				finite_difference,
				rtol=2e-8,
				atol=2e-9,
			)
			defect = implicit_function.T @ form @ implicit_function - form
			self.assertLess(float(np.linalg.norm(defect, ord="fro")), 2e-12)
			cross_particle = implicit_function[np.ix_((0, 2), (1, 3))]
			np.testing.assert_array_equal(cross_particle, 0.0)
			self.assertNotIn("final_state_jacobian", solution.diagnostics)

	def test_analytic_methods_reject_a_generic_step(self) -> None:
		state = np.asarray([1.0, 1.2])
		step = IntegrationStep(
			dynamics_name="GuidingCenterDynamics",
			method_name="RK4",
			step_index=0,
			time=0.1,
			duration=0.1,
			state_before=state,
			state_after=state,
			map_state=lambda value: value.copy(),
		)
		for calculator in (
			implicit_function_step_jacobian,
			stage_increment_step_jacobian,
		):
			with self.assertRaisesRegex(TypeError, "ImplicitABBAIntegrationStep"):
				calculator(step)

	def test_analytic_methods_use_the_exact_observed_endpoint_times(self) -> None:
		dynamics = _TimeOnlyPlanarDynamics()
		start = 0.2
		duration = 0.1
		events = []
		simulate(
			InitialValueProblem(
				dynamics,
				TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
			),
			ImplicitABBA1(step_observer=events.append),
			SimulationRequest.uniform(
				t_span=(start, start + duration),
				max_step=duration,
				sample_count=2,
			),
		)
		self.assertEqual(len(events), 1)
		self.assertEqual(events[0].start_time, start)
		for calculator in (
			implicit_function_step_jacobian,
			stage_increment_step_jacobian,
		):
			dynamics.jacobian_times.clear()
			jacobian = calculator(events[0])
			self.assertEqual(
				dynamics.jacobian_times,
				[start, start, start + duration, start + duration],
			)
			np.testing.assert_array_equal(jacobian, np.eye(2))


class ImplicitABBAObserverStudyTests(unittest.TestCase):
	"""Verify configurable method-observer fan-out and persisted provenance."""

	def test_six_default_combinations_share_two_implicit_integrations(self) -> None:
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
		config = ImplicitABBASymplecticityConfig(
			steps=(AreaStep(label="step", value=np.pi / 400),),
			t_span=(0.0, np.pi / 400),
			save_interval=np.pi / 400,
			chunk_size=2,
		)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			comparison = run_implicit_abba_symplecticity_study(
				potential,
				area,
				notebook_path=(
					root / "notebooks" / "developements" / "implicit_abba.ipynb"
				),
				config=config,
				project_root=root,
			)
			self.assertEqual(
				tuple(comparison.results),
				tuple(observer.label for observer in DEFAULT_IMPLICIT_ABBA_OBSERVERS),
			)
			for formulation in ("implicit_1", "implicit_2"):
				labels = [
					observer.label
					for observer in config.observers
					if observer.formulation == formulation
				]
				solutions = [comparison.results[label].solutions["step"] for label in labels]
				self.assertTrue(all(solution is solutions[0] for solution in solutions[1:]))
				analytic = [
					comparison.results[label]
					for label in labels
					if comparison.results[label].jacobian_method != "finite_difference"
				]
				for result in analytic:
					self.assertLess(result.summaries()[0].max_local_defect, 2e-12)

			for observer in config.observers:
				result = comparison.results[observer.label]
				metadata_path = next(
					result.output_directories["step"].glob(
						f"*{observer.label}*_metadata_*.json"
					)
				)
				payload = json.loads(metadata_path.read_text(encoding="utf-8"))
				self.assertEqual(
					payload["step_jacobian_method"],
					observer.jacobian_method,
				)
				self.assertEqual(
					payload["metadata"]["observer_label"],
					observer.label,
				)
				self.assertEqual(
					payload["metadata"]["step_jacobian_scope"],
					(
						"emitted_finite_tolerance_solver_map"
						if observer.jacobian_method == "finite_difference"
						else "ideal_converged_projection_root"
					),
				)

		self.assertLess(comparison.maximum_state_differences()["step"], 1e-14)

	def test_generic_observer_defaults_to_finite_differences(self) -> None:
		area = Area.square(
			center=(1.0, 1.0),
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		state = area.initial_state
		assert state is not None
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			with GCAreaSymplecticityObserver(
				notebook_path=Path(temporary) / "notebooks/developements/generic.ipynb",
				area=area,
				project_root=temporary,
				verbose=False,
			) as observer:
				observer(
					IntegrationStep(
						dynamics_name="GuidingCenterDynamics",
						method_name="GenericMethod",
						step_index=0,
						time=0.1,
						duration=0.1,
						state_before=state,
						state_after=state,
						map_state=lambda value: value.copy(),
					)
				)
		self.assertEqual(observer.jacobian_method, "finite_difference")
		self.assertLess(observer.records[-1].local_relative_defect, 1e-10)


if __name__ == "__main__":
	unittest.main()
