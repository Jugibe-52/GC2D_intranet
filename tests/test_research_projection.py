"""Contracts for projected GC symplecticity and area research."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from classes import Area, IntegrationStage, Potential, SystemGC
from research.projection import (
	ProjectedSymplecticityAreaObserver,
	gc_average_projection,
	gc_diagonal_embedding,
	gc_physical_symplectic_form,
)


class ProjectionResearchTests(unittest.TestCase):
	"""Verify physical projection geometry, tangent propagation and persistence."""

	def test_embedding_and_projection_restore_the_physical_state(self) -> None:
		embedding = gc_diagonal_embedding(3)
		projection = gc_average_projection(3)
		form = gc_physical_symplectic_form(3)

		self.assertEqual(embedding.shape, (12, 6))
		self.assertEqual(projection.shape, (6, 12))
		np.testing.assert_allclose(projection @ embedding, np.eye(6))
		np.testing.assert_allclose(form.T, -form)
		np.testing.assert_allclose(form @ form, -np.eye(6))

	def test_identity_stages_preserve_projected_symplecticity_and_area(self) -> None:
		area = Area.square(center=(1.0, 1.0), side=0.5, points_per_side=1)
		physical_state = area.state
		assert physical_state is not None
		extended_state = np.concatenate((physical_state, physical_state))

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			with ProjectedSymplecticityAreaObserver(
				notebook_path=(
					root / "notebooks" / "developements" / "projected_area.ipynb"
				),
				area=area,
				project_root=root,
				run_date="2026-07-20",
				record_every=2,
				chunk_size=2,
				verbose=False,
			) as observer:
				for stage_index in range(12):
					observer(
						IntegrationStage(
							system_name="SystemGC",
							flow_name=(
								"adjoint_flow" if stage_index % 2 == 0 else "flow"
							),
							step_index=0,
							stage_index=stage_index,
							time=0.01 * (stage_index + 1) / 12,
							duration=0.01 / 12,
							state_before=extended_state,
							state_after=extended_state,
							map_state=lambda value: value.copy(),
						)
					)

			self.assertEqual(len(observer.records), 2)
			initial, final = observer.records
			self.assertEqual(initial.step_index, -1)
			self.assertEqual(final.step_index, 0)
			self.assertLess(final.relative_defect, 1e-10)
			self.assertLess(final.determinant_error, 1e-10)
			self.assertAlmostEqual(final.relative_area_error, 0.0)
			self.assertAlmostEqual(final.copy_separation, 0.0)
			self.assertEqual(len(observer.output_blocks), 1)
			block = observer.output_blocks[0]
			self.assertIn(
				"outputs/developements/projected_area/2026-07-20",
				str(block.summary_path),
			)
			with np.load(block.jacobians_path) as arrays:
				self.assertEqual(arrays["projected_jacobians"].shape, (2, 8, 8))
				self.assertEqual(arrays["projected_states"].shape, (2, 8))

	def test_real_gc_projection_matches_the_saved_area(self) -> None:
		area = Area.square(
			center=(np.pi, np.pi),
			side=0.5,
			points_per_side=1,
			rho=0.05,
		)
		potential = Potential.random(
			A=0.08,
			M=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		)
		system = SystemGC(potential, area, coupling_frequency=0.0)

		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			with ProjectedSymplecticityAreaObserver(
				notebook_path=root / "notebooks" / "developements" / "real.ipynb",
				area=area,
				period=potential.grid.period,
				project_root=root,
				verbose=False,
			) as observer:
				solution = system.simulate(
					step=0.01,
					t_span=(0.0, 0.01),
					n_output_samples=2,
					check_energy=False,
					progress=False,
					stage_observer=observer,
				)

			final = observer.records[-1]
			saved_area = float(
				area.calculate_area(solution.y[:, -1], period=potential.grid.period)
			)
			self.assertAlmostEqual(final.signed_area, saved_area)
			self.assertTrue(np.isfinite(final.relative_defect))
			self.assertTrue(np.isfinite(final.relative_area_error))


if __name__ == "__main__":
	unittest.main()
