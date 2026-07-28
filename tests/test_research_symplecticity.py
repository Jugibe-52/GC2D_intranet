"""Contracts for research-only GC symplecticity observations."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from classes import Area, IntegrationStage, IntegrationStep
from research.symplecticity import (
	GCAreaSymplecticityObserver,
	SymplecticityObserver,
	central_difference_jacobian,
	gc_extended_symplectic_form,
	notebook_output_directory,
)


class SymplecticityResearchTests(unittest.TestCase):
	"""Verify differentiation, canonical layout and indexed output blocks."""

	def test_centered_jacobian_recovers_a_linear_map(self) -> None:
		matrix = np.asarray(
			[
				[1.0, 0.2, 0.0, 0.0],
				[0.0, 1.0, 0.0, 0.0],
				[0.0, 0.0, 1.0, -0.3],
				[0.0, 0.0, 0.0, 1.0],
			]
		)
		state = np.asarray([0.3, -0.2, 0.8, 1.1])
		jacobian = central_difference_jacobian(lambda value: matrix @ value, state)
		np.testing.assert_allclose(jacobian, matrix, atol=1e-10)

	def test_observer_writes_notebook_dated_blocks_without_overwriting(self) -> None:
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			notebook = root / "notebooks" / "developements" / "study.ipynb"
			expected_directory = (
				root / "outputs" / "developements" / "study" / "2026-07-20"
			)
			self.assertEqual(
				notebook_output_directory(
					notebook,
					project_root=root,
					run_date=date(2026, 7, 20),
				),
				expected_directory,
			)

			state = np.asarray([0.2, 0.4, 0.6, 0.8])
			identity = np.eye(4)
			with SymplecticityObserver(
				notebook_path=notebook,
				particle_count=1,
				project_root=root,
				run_date="2026-07-20",
				chunk_size=2,
				verbose=False,
				metadata={"coupling_frequency": 0.0},
			) as observer:
				for stage_index in range(3):
					observer(
						IntegrationStage(
							dynamics_name="GuidingCenterDynamics",
							formulation_name="GCExtendedFormulation",
							method_name="BM4Composition",
							flow_name=(
								"flow" if stage_index % 2 else "adjoint_flow"
							),
							step_index=0,
							stage_index=stage_index,
							time=0.1 * stage_index,
							duration=0.01,
							state_before=state,
							state_after=state,
							map_state=lambda value: identity @ value,
						)
					)

			self.assertEqual(len(observer.records), 3)
			self.assertEqual(len(observer.output_blocks), 2)
			for record in observer.records:
				self.assertLess(record.relative_defect, 1e-10)
				self.assertLess(record.determinant_error, 1e-10)
			for block in observer.output_blocks:
				self.assertTrue(block.summary_path.is_file())
				self.assertTrue(block.jacobians_path.is_file())
				self.assertTrue(block.metadata_path.is_file())
				with np.load(block.jacobians_path) as arrays:
					self.assertEqual(arrays["jacobians"].shape[1:], (4, 4))
				with block.metadata_path.open(encoding="utf-8") as stream:
					metadata = json.load(stream)
				self.assertEqual(metadata["metadata"]["coupling_frequency"], 0.0)

			with SymplecticityObserver(
				notebook_path=notebook,
				particle_count=1,
				project_root=root,
				run_date="2026-07-20",
				verbose=False,
			) as next_observer:
				next_observer(
					IntegrationStage(
						dynamics_name="GuidingCenterDynamics",
						formulation_name="GCExtendedFormulation",
						method_name="BM4Composition",
						flow_name="flow",
						step_index=0,
						stage_index=0,
						time=0.0,
						duration=0.01,
						state_before=state,
						state_after=state,
						map_state=lambda value: value.copy(),
					)
				)
			self.assertEqual(next_observer.output_blocks[0].index, 2)

	def test_gc_form_uses_both_component_major_copies(self) -> None:
		form = gc_extended_symplectic_form(2)
		self.assertEqual(form.shape, (8, 8))
		# Each copy is isotropic on its own; the extended canonical pairs cross
		# between the first and second copy used by the triangular maps.
		np.testing.assert_allclose(form[:4, :4], 0.0)
		np.testing.assert_allclose(form[4:, 4:], 0.0)
		self.assertEqual(form[0, 6], 1.0)
		self.assertEqual(form[2, 4], -1.0)
		np.testing.assert_allclose(form.T, -form)
		np.testing.assert_allclose(form @ form, -np.eye(8))

	def test_observer_rejects_non_gc_and_energy_augmented_stages(self) -> None:
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			observer = SymplecticityObserver(
				notebook_path=root / "notebooks" / "developements" / "study.ipynb",
				particle_count=1,
				project_root=root,
				verbose=False,
			)
			base = {
				"formulation_name": "GCExtendedFormulation",
				"method_name": "BM4Composition",
				"flow_name": "flow",
				"step_index": 0,
				"stage_index": 0,
				"time": 0.0,
				"duration": 0.01,
				"map_state": lambda value: value.copy(),
			}
			state = np.ones(4)
			with self.assertRaises(TypeError):
				observer(
					IntegrationStage(
						dynamics_name="FullCyclotronDynamics",
						state_before=state,
						state_after=state,
						**base,
					)
				)
			energy_augmented = np.ones(5)
			with self.assertRaises(ValueError):
				observer(
					IntegrationStage(
						dynamics_name="GuidingCenterDynamics",
						state_before=energy_augmented,
						state_after=energy_augmented,
						**base,
					)
				)
			observer.close()

	def test_physical_area_observer_accumulates_complete_step_jacobians(self) -> None:
		area = Area.square(
			center=(1.0, 1.0),
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		state = area.initial_state
		assert state is not None
		identity = np.eye(state.size)
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			with GCAreaSymplecticityObserver(
				notebook_path=(
					root
					/ "notebooks"
					/ "experiments"
					/ "symplecticity"
					/ "rk4.ipynb"
				),
				area=area,
				project_root=root,
				run_date="2026-07-20",
				record_every=2,
				chunk_size=2,
				verbose=False,
			) as observer:
				for step_index in range(2):
					observer(
						IntegrationStep(
							dynamics_name="GuidingCenterDynamics",
							method_name="RK4",
							step_index=step_index,
							time=0.1 * (step_index + 1),
							duration=0.1,
							state_before=state,
							state_after=state,
							map_state=lambda value: identity @ value,
						)
					)

			self.assertEqual(len(observer.records), 2)
			self.assertEqual(observer.records[0].step_index, -1)
			self.assertEqual(observer.records[-1].step_index, 1)
			self.assertLess(observer.records[-1].relative_defect, 1e-10)
			self.assertLess(observer.records[-1].local_relative_defect, 1e-10)
			self.assertEqual(len(observer.output_blocks), 1)
			block = observer.output_blocks[0]
			with np.load(block.jacobians_path) as arrays:
				self.assertEqual(
					arrays["accumulated_jacobians"].shape,
					(2, state.size, state.size),
				)


if __name__ == "__main__":
	unittest.main()
