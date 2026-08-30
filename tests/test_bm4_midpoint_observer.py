"""Exact-Jacobian diagnostics for full-cycle midpoint BM4."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnostics import (
	MidpointBM4SymplecticityObserver,
	central_difference_jacobian,
	midpoint_bm4_step_particle_jacobians,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	InitialValueProblem,
	IntegrationStage,
	MidpointBM4,
	SimulationRequest,
	simulate,
)
from simulation.methods.bm4._core import _BM4_ORDERS, _BM4_STAGES


def _dynamics() -> GuidingCenterDynamics:
	"""Build a compact GC field with exact Hessian interpolation."""
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


def _dense_component_major_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent planar blocks into the packed physical layout."""
	particle_count = blocks.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = blocks[particle]
	return result


class _UnequalParticleJacobianDynamics:
	"""Return unequal constant Jacobians to expose mean-versus-RMS reduction."""

	state_dimension = 2

	def vector_field(self, _time: float, state: np.ndarray) -> np.ndarray:
		"""Return a zero field because synthetic stage states stay at zero."""
		return np.zeros_like(state, dtype=float)

	def particle_vector_field_jacobians(
		self,
		_time: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return one identity tangent and one deliberately noncanonical tangent."""
		particle_count = state.size // 2
		if particle_count != 2:
			raise ValueError("This test dynamics requires exactly two particles.")
		return np.asarray(
			[
				[[0.0, 0.0], [0.0, 0.0]],
				[[0.4, 0.2], [-0.1, 0.3]],
			]
		)


def _synthetic_complete_step(
	dynamics: _UnequalParticleJacobianDynamics,
	observer: MidpointBM4SymplecticityObserver,
	*,
	step: float,
) -> None:
	"""Emit one ordered zero-state cycle with the production BM4 coefficients."""
	state = np.zeros(8)
	stage_time = 0.0
	for stage_index, (coefficient, order) in enumerate(
		zip(_BM4_STAGES, _BM4_ORDERS, strict=True)
	):
		duration = float(coefficient * step)
		flow_name = "flow" if order == 0 else "adjoint_flow"
		evaluation_time = (
			stage_time + duration if flow_name == "flow" else stage_time
		)
		observer(
			IntegrationStage(
				dynamics_name=type(dynamics).__name__,
				formulation_name="GCStageProjectedFormulation",
				method_name="MidpointBM4",
				flow_name=flow_name,
				step_index=0,
				stage_index=stage_index,
				time=evaluation_time,
				duration=duration,
				state_before=state,
				state_after=state,
				map_state=lambda candidate: np.asarray(candidate, dtype=float).copy(),
				dynamics=dynamics,
			)
		)
		stage_time += duration


class MidpointBM4SymplecticityObserverTests(unittest.TestCase):
	"""Verify exact factorization, arithmetic aggregation, and persistence."""

	def _captured_events(self) -> tuple[GuidingCenterDynamics, list[IntegrationStage]]:
		"""Return one production midpoint-BM4 cycle for validation tests."""
		dynamics = _dynamics()
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0, 1.4]),
			y=np.asarray([1.2, 1.6]),
		)
		events: list[IntegrationStage] = []
		simulate(
			InitialValueProblem(dynamics, configuration),
			MidpointBM4(stage_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.1),
				max_step=0.1,
				sample_count=2,
			),
		)
		self.assertEqual(len(events), 12)
		return dynamics, events

	def test_explicit_physical_jacobian_matches_centered_difference_audit(
		self,
	) -> None:
		dynamics, events = self._captured_events()
		exact_blocks = midpoint_bm4_step_particle_jacobians(events, dynamics)
		exact = _dense_component_major_jacobian(exact_blocks)
		initial_state = np.asarray(events[0].state_before[:4], dtype=float)
		physical_size = initial_state.size

		def physical_step(candidate: np.ndarray) -> np.ndarray:
			"""Apply the fixed observed stages between diagonal embedding and mean."""
			internal = np.concatenate((candidate, candidate))
			for event in events:
				internal = event.map_state(internal)
			return (
				internal[:physical_size] + internal[physical_size:]
			) / 2.0

		numerical = central_difference_jacobian(
			physical_step,
			initial_state,
			relative_step=1e-5,
		)
		relative_error = float(
			np.linalg.norm(exact - numerical, ord="fro")
			/ np.linalg.norm(numerical, ord="fro")
		)
		self.assertLess(relative_error, 2e-8)

		broken_state = events[1].state_before.copy()
		broken_state[0] += 1e-4
		broken_events = [
			events[0],
			replace(events[1], state_before=broken_state),
			*events[2:],
		]
		with self.assertRaisesRegex(ValueError, "not continuous"):
			midpoint_bm4_step_particle_jacobians(broken_events, dynamics)

	def test_streaming_observer_rejects_disconnected_or_mistimed_stages(
		self,
	) -> None:
		"""Reject event streams that cannot represent one physical tangent."""
		dynamics, events = self._captured_events()
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			observer = MidpointBM4SymplecticityObserver(
				dynamics=dynamics,
				notebook_path=root / "notebooks" / "developements" / "stream.ipynb",
				project_root=root,
				verbose=False,
			)
			observer(events[0])
			broken_state = events[1].state_before.copy()
			broken_state[0] += 1e-4
			with self.assertRaisesRegex(ValueError, "not continuous"):
				observer(replace(events[1], state_before=broken_state))

			mistimed = MidpointBM4SymplecticityObserver(
				dynamics=dynamics,
				notebook_path=root / "notebooks" / "developements" / "time.ipynb",
				project_root=root,
				verbose=False,
			)
			mistimed(events[0])
			with self.assertRaisesRegex(ValueError, "evaluation times"):
				mistimed(replace(events[1], time=events[1].time + 1e-4))

	def test_observer_rejects_a_different_same_named_dynamics_instance(self) -> None:
		"""Prevent exact Hessians from being evaluated for the wrong potential."""
		dynamics, events = self._captured_events()
		other_dynamics = GuidingCenterDynamics(
			Potential.random(
				A=0.08,
				M=3,
				nx=16,
				ny=16,
				seed=28,
				interpolation_order=5,
			),
			rho=0.05,
		)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			observer = MidpointBM4SymplecticityObserver(
				dynamics=other_dynamics,
				notebook_path=(
					root / "notebooks" / "developements" / "wrong_dynamics.ipynb"
				),
				project_root=root,
				verbose=False,
			)
			with self.assertRaisesRegex(TypeError, "exact configured dynamics"):
				observer(events[0])

		with self.assertRaisesRegex(ValueError, "exact configured dynamics"):
			midpoint_bm4_step_particle_jacobians(events, other_dynamics)

	def test_streaming_observer_rejects_an_incomplete_cycle_on_close(self) -> None:
		"""Prevent silently persisting a partially composed stage tangent."""
		dynamics, events = self._captured_events()
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			observer = MidpointBM4SymplecticityObserver(
				dynamics=dynamics,
				notebook_path=(
					root / "notebooks" / "developements" / "incomplete.ipynb"
				),
				project_root=root,
				verbose=False,
			)
			observer(events[0])
			with self.assertRaisesRegex(RuntimeError, "incomplete"):
				observer.close()

			# A failed close leaves the observer usable once the cycle is completed.
			for event in events[1:]:
				observer(event)
			observer.close()
			self.assertEqual(len(observer.records), 2)

	def test_context_cleanup_does_not_mask_a_stage_validation_error(self) -> None:
		"""Preserve the original integration failure during observer cleanup."""
		dynamics, events = self._captured_events()
		broken_state = events[1].state_before.copy()
		broken_state[0] += 1e-4
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			with self.assertRaisesRegex(ValueError, "not continuous"):
				with MidpointBM4SymplecticityObserver(
					dynamics=dynamics,
					notebook_path=(
						root
						/ "notebooks"
						/ "developements"
						/ "failed_stream.ipynb"
					),
					project_root=root,
					verbose=False,
				) as observer:
					observer(events[0])
					observer(replace(events[1], state_before=broken_state))

	def test_record_uses_arithmetic_particle_mean_instead_of_dense_rms(self) -> None:
		dynamics = _UnequalParticleJacobianDynamics()
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			with MidpointBM4SymplecticityObserver(
				dynamics=dynamics,
				notebook_path=(
					root
					/ "notebooks"
					/ "developements"
					/ "mean_reduction.ipynb"
				),
				project_root=root,
				run_date="2026-08-15",
				chunk_size=8,
				verbose=False,
			) as observer:
				_synthetic_complete_step(dynamics, observer, step=0.2)

			self.assertEqual(len(observer.records), 2)
			block = observer.output_blocks[0]
			with np.load(block.jacobians_path) as arrays:
				defects = arrays["accumulated_relative_defects"][-1]
				determinant_errors = arrays[
					"accumulated_determinant_errors"
				][-1]

		final = observer.records[-1]
		expected_mean = float(np.mean(defects))
		rms = float(np.sqrt(np.mean(defects**2)))
		self.assertEqual(defects[0], 0.0)
		self.assertGreater(defects[1], 1e-6)
		self.assertAlmostEqual(
			final.mean_accumulated_relative_defect,
			expected_mean,
			places=15,
		)
		self.assertAlmostEqual(
			final.std_accumulated_relative_defect,
			float(np.std(defects)),
			places=15,
		)
		self.assertAlmostEqual(
			final.max_accumulated_relative_defect,
			float(np.max(defects)),
			places=15,
		)
		self.assertAlmostEqual(
			final.mean_accumulated_determinant_error,
			float(np.mean(determinant_errors)),
			places=15,
		)
		self.assertGreater(abs(rms - expected_mean), 1e-6)

	def test_cadence_keeps_initial_sample_and_forces_the_final_step(self) -> None:
		dynamics = _dynamics()
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0, 1.4]),
			y=np.asarray([1.2, 1.6]),
		)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			with MidpointBM4SymplecticityObserver(
				dynamics=dynamics,
				notebook_path=(
					root / "notebooks" / "developements" / "cadence.ipynb"
				),
				project_root=root,
				run_date="2026-08-15",
				record_every=2,
				chunk_size=2,
				verbose=False,
			) as observer:
				simulate(
					InitialValueProblem(dynamics, configuration),
					MidpointBM4(stage_observer=observer),
					SimulationRequest.uniform(
						t_span=(0.0, 0.3),
						max_step=0.1,
						sample_count=4,
					),
				)

			self.assertEqual(
				[record.step_index for record in observer.records],
				[-1, 1, 2],
			)
			self.assertEqual(len(observer.output_blocks), 2)
			self.assertEqual(
				[block.sample_count for block in observer.output_blocks],
				[2, 1],
			)
			first_block = observer.output_blocks[0]
			with np.load(first_block.jacobians_path) as arrays:
				self.assertEqual(arrays["local_jacobians"].shape, (2, 2, 2, 2))
				self.assertEqual(
					arrays["accumulated_relative_defects"].shape,
					(2, 2),
				)
			with first_block.metadata_path.open(encoding="utf-8") as stream:
				metadata = json.load(stream)
			self.assertEqual(
				metadata["jacobian_method"],
				"explicit_uncoupled_stage_factorization",
			)
			self.assertEqual(metadata["particle_aggregation"], "arithmetic_mean")


if __name__ == "__main__":
	unittest.main()
