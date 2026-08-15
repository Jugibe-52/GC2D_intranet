"""Contracts for full-cycle midpoint projection of uncoupled BM4."""

from __future__ import annotations

import unittest

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import TrajectoryGC
from potential import Potential
from simulation import (
	InitialValueProblem,
	MidpointBM4,
	SimulationRequest,
	simulate,
)
from simulation.methods.bm4 import _BM4_ORDERS, _BM4_STAGES


def _deterministic_gc_dynamics() -> GuidingCenterDynamics:
	"""Build a compact nonautonomous field with exact reproducible values."""
	return GuidingCenterDynamics(
		Potential.random(
			A=0.08,
			M=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		),
		rho=0.05,
	)


def _reference_cycle(
	dynamics: GuidingCenterDynamics,
	time: float,
	state: np.ndarray,
	step: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Apply the twelve uncoupled shears without a midpoint projection."""
	first = np.asarray(state, dtype=float).copy()
	second = first.copy()
	stage_time = float(time)
	for coefficient, order in zip(_BM4_STAGES, _BM4_ORDERS, strict=True):
		duration = float(coefficient * step)
		if order == 0:
			evaluation_time = stage_time + duration
			second = second + duration * dynamics.vector_field(
				evaluation_time,
				first,
			)
			first = first + duration * dynamics.vector_field(
				evaluation_time,
				second,
			)
		else:
			evaluation_time = stage_time
			first = first + duration * dynamics.vector_field(
				evaluation_time,
				second,
			)
			second = second + duration * dynamics.vector_field(
				evaluation_time,
				first,
			)
		stage_time += duration
	return first, second


class MidpointBM4Tests(unittest.TestCase):
	"""Verify the complete-cycle map, projection, events, and shadow sampling."""

	def test_method_has_fourth_order_global_accuracy(self) -> None:
		problem = InitialValueProblem(
			_deterministic_gc_dynamics(),
			TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
		)

		def final_state(step: float) -> np.ndarray:
			"""Return one physical endpoint on a fixed interval."""
			return simulate(
				problem,
				MidpointBM4(),
				SimulationRequest.uniform(
					t_span=(0.0, 0.4),
					max_step=step,
					sample_count=2,
				),
			).states[:, -1]

		reference = final_state(0.003125)
		coarse_error = float(np.linalg.norm(final_state(0.1) - reference))
		fine_error = float(np.linalg.norm(final_state(0.05) - reference))
		self.assertGreater(coarse_error / fine_error, 15.0)
		self.assertLess(coarse_error / fine_error, 17.0)

	def test_one_cycle_matches_uncoupled_bm4_then_arithmetic_mean(self) -> None:
		dynamics = _deterministic_gc_dynamics()
		initial_state = np.asarray([1.0, 1.2])
		start = 0.2
		step = 0.1
		events = []
		solution = simulate(
			InitialValueProblem(
				dynamics,
				TrajectoryGC(initial_state, rho=0.05),
			),
			MidpointBM4(stage_observer=events.append),
			SimulationRequest.uniform(
				t_span=(start, start + step),
				max_step=step,
				sample_count=2,
			),
		)
		first, second = _reference_cycle(dynamics, start, initial_state, step)
		extended_reference = np.concatenate((first, second))

		self.assertEqual(len(events), 12)
		np.testing.assert_allclose(
			events[-1].state_after,
			extended_reference,
			rtol=0.0,
			atol=2e-15,
		)
		np.testing.assert_allclose(
			solution.states[:, -1],
			(first + second) / 2.0,
			rtol=0.0,
			atol=2e-15,
		)
		self.assertEqual([event.stage_index for event in events], list(range(12)))
		for event in events:
			self.assertEqual(event.method_name, "MidpointBM4")
			self.assertEqual(
				event.formulation_name,
				"GCStageProjectedFormulation",
			)
			np.testing.assert_allclose(
				event.map_state(event.state_before),
				event.state_after,
				rtol=0.0,
				atol=2e-15,
			)

	def test_complete_cycle_projection_resets_the_next_step_diagonal(self) -> None:
		dynamics = _deterministic_gc_dynamics()
		initial_state = np.asarray([1.0, 1.2])
		events = []
		solution = simulate(
			InitialValueProblem(
				dynamics,
				TrajectoryGC(initial_state, rho=0.05),
			),
			MidpointBM4(stage_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.2),
				max_step=0.1,
				sample_count=3,
			),
		)
		physical_size = initial_state.size
		first_cycle_end = events[11].state_after
		first = first_cycle_end[:physical_size]
		second = first_cycle_end[physical_size:]
		projected = (first + second) / 2.0
		second_cycle_start = events[12].state_before

		self.assertEqual(len(events), 24)
		self.assertGreater(float(np.linalg.norm(first - second)), 0.0)
		np.testing.assert_array_equal(
			second_cycle_start,
			np.concatenate((projected, projected)),
		)
		np.testing.assert_array_equal(solution.states[:, 1], projected)

		expected_separations = np.asarray(
			[
				np.linalg.norm(
					events[index].state_after[:physical_size]
					- events[index].state_after[physical_size:],
					ord=np.inf,
				)
				for index in (11, 23)
			]
		)
		np.testing.assert_array_equal(
			solution.diagnostics["copy_separation_norms"],
			expected_separations,
		)
		self.assertEqual(solution.diagnostics["projection_kind"], "arithmetic_mean")
		self.assertEqual(
			solution.diagnostics["projection_scope"],
			"complete_bm4_cycle",
		)
		self.assertEqual(solution.diagnostics["projections_per_step"], 1)
		self.assertEqual(
			solution.diagnostics["vector_field_evaluations_per_step"],
			24,
		)

	def test_shadow_cycles_do_not_emit_stages_or_copy_diagnostics(self) -> None:
		dynamics = _deterministic_gc_dynamics()
		problem = InitialValueProblem(
			dynamics,
			TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05),
		)
		events = []
		dense = simulate(
			problem,
			MidpointBM4(stage_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=13,
			),
		)
		sparse = simulate(
			problem,
			MidpointBM4(),
			SimulationRequest.uniform(
				t_span=(0.0, 0.05),
				max_step=0.02,
				sample_count=4,
			),
		)

		self.assertEqual(dense.n_steps, 3)
		self.assertEqual(len(events), 12 * dense.n_steps)
		self.assertEqual(
			np.asarray(dense.diagnostics["copy_separation_norms"]).shape,
			(dense.n_steps,),
		)
		np.testing.assert_array_equal(dense.states[:, ::4], sparse.states)
		self.assertEqual(
			[event.step_index for event in events[::12]],
			[0, 1, 2],
		)


if __name__ == "__main__":
	unittest.main()
