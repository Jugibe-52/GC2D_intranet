"""Uniform diagnostic chunking, finalization, and preservation of active errors."""

from __future__ import annotations

from collections.abc import Callable
import csv
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from contracts.observation import IntegrationStage, IntegrationStep
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from diagnostics.abba_jacobian.observer import ImplicitABBAJacobianObserver
from diagnostics.buffering import DiagnosticBuffer
from diagnostics.implicit_iterations import ImplicitABBAIterationObserver
from diagnostics.output import DiagnosticBlockPaths
from diagnostics.projection.observer import ProjectedSymplecticityAreaObserver
from diagnostics.symplecticity.area import GCAreaSymplecticityObserver
from diagnostics.symplecticity.observer import SymplecticityObserver
from diagnostics.trajectory_symplecticity.jacobians import abba2_implicit_step_particle_jacobians
from diagnostics.trajectory_symplecticity.observer import GCTrajectorySymplecticityObserver
from dynamics.gc import GuidingCenterDynamics
from initial_conditions.area import Area
from initial_conditions.gc import GCInitialConfiguration
from methods.extended.abba import ABBA2Implicit
from potential.potential import Potential
from simulation.runner import simulate


def _buffer(directory: Path, *, chunk_size: int = 2) -> DiagnosticBuffer[int, DiagnosticBlockPaths]:
	"""Create a tiny real output stream with observable chunk contents."""
	def write(index: int, samples: tuple[int, ...]) -> DiagnosticBlockPaths:
		return output.write(
			block_index=index,
			rows=[{"value": value} for value in samples],
			arrays={"values": np.asarray(samples)},
			metadata={},
		)

	output = DiagnosticBuffer[int, DiagnosticBlockPaths](
		output_directory=directory, block_name="test", chunk_size=chunk_size,
		write_block=write,
	)
	return output


class DiagnosticBufferingTests(unittest.TestCase):
	"""Check common lifecycle contracts once and every observer's composition."""

	def test_chunk_limit_partial_flush_and_idempotent_close(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			output = _buffer(Path(temporary))
			output.append(1)
			self.assertIsNone(output.flush_if_full())
			output.append(2)
			first = output.flush_if_full()
			assert first is not None
			output.append(3)
			output.close()
			output.close()
			self.assertEqual(len(output.blocks), 2)
			for expected, paths in zip(([1, 2], [3]), output.blocks, strict=True):
				with np.load(paths.arrays) as archive:
					np.testing.assert_array_equal(archive["values"], expected)
				self.assertEqual(json.loads(paths.metadata.read_text())["sample_count"], len(expected))
			with self.assertRaises(RuntimeError):
				output.append(4)
			self.assertIsNone(output.flush())

	def test_failed_write_preserves_samples_and_index_for_retry(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			output = _buffer(Path(temporary))
			output.append(7)
			with patch("diagnostics.buffering.write_diagnostic_block", side_effect=OSError("disk full")):
				with self.assertRaisesRegex(OSError, "disk full"):
					output.close()
			self.assertFalse(output.closed)
			self.assertEqual(output.blocks, ())
			output.close()
			self.assertEqual(len(output.blocks), 1)
			metadata = json.loads(output.blocks[0].metadata.read_text())
			self.assertEqual(metadata["block_index"], 0)
			with np.load(output.blocks[0].arrays) as archive:
				np.testing.assert_array_equal(archive["values"], [7])

	def _observers(self, root: Path) -> list[tuple[Any, Callable[[], None]]]:
		"""Provide a valid pending sample for every migrated observer family."""
		common: dict[str, Any] = {
			"notebook_path": "notebooks/experiments/diagnostic.ipynb",
			"project_root": root, "chunk_size": 100, "verbose": False,
		}
		potential = Potential.random(A=0.02, M=2, nx=8, ny=8)
		dynamics = GuidingCenterDynamics(potential)
		configuration = GCInitialConfiguration(np.array([1.0, 0.7]))
		problem = InitialValueProblem(dynamics, configuration)
		events: list[IntegrationStep] = []
		simulate(problem, ABBA2Implicit(step_observer=events.append),
			SimulationRequest.uniform(t_span=(0.0, 0.01), max_step=0.01, sample_count=2))
		event = events[0]
		area = Area.square(center=(1.0, 0.7), side=0.1)
		area_state = area.initial_state
		assert area_state is not None
		area_step = IntegrationStep(
			dynamics_name="GuidingCenterDynamics", method_name="Identity", step_index=0,
			time=0.01, duration=0.01, start_time=0.0,
			state_before=area_state, state_after=area_state, map_state=lambda state: state.copy(),
		)
		doubled = np.concatenate((area_state, area_state))
		stage = IntegrationStage(
			dynamics_name="GuidingCenterDynamics", formulation_name="GCExtendedFormulation",
			method_name="Identity", flow_name="adjoint_flow", step_index=0, stage_index=0,
			time=0.0, duration=0.01, state_before=doubled, state_after=doubled,
			map_state=lambda state: state.copy(),
		)
		symplecticity = SymplecticityObserver(particle_count=4, **common)
		physical_area = GCAreaSymplecticityObserver(area=area, **common)
		projection = ProjectedSymplecticityAreaObserver(area=area, **common)
		jacobian = ImplicitABBAJacobianObserver(**common)
		iterations = ImplicitABBAIterationObserver(**common)
		trajectory = GCTrajectorySymplecticityObserver(
			dynamics=dynamics, initial_configuration=configuration,
			method_name="ABBA2Implicit", jacobian_method="implicit_function",
			jacobian_calculator=abba2_implicit_step_particle_jacobians, **common,
		)
		return [
			(symplecticity, lambda: symplecticity(stage)),
			(physical_area, lambda: physical_area(area_step)),
			(projection, lambda: projection(stage)),
			(jacobian, lambda: jacobian(event)),
			(iterations, lambda: iterations(event)),
			(trajectory, lambda: trajectory(event)),
		]

	def test_all_observers_keep_integration_error_when_output_cleanup_also_fails(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			for observer, feed in self._observers(Path(temporary)):
				with self.subTest(observer=type(observer).__name__):
					error = RuntimeError("integration failed")
					with patch("diagnostics.buffering.write_diagnostic_block", side_effect=OSError("disk full")) as write:
						with self.assertRaises(RuntimeError) as raised:
							with observer:
								feed()
								raise error
						self.assertIs(raised.exception, error)
						self.assertTrue(any("disk full" in note for note in error.__notes__))
						observer.close()
						self.assertEqual(write.call_count, 1)
					recovered = observer.flush()
					self.assertIsNotNone(recovered)
					self.assertEqual(recovered.index, 0)
					with recovered.summary_path.open(newline="") as stream:
						rows = list(csv.DictReader(stream))
					self.assertEqual(len(rows), len(observer.records))
					self.assertIsNone(observer.flush())
					observer.close()
					self.assertEqual(len(observer.output_blocks), 1)
					with self.assertRaises(RuntimeError):
						feed()


if __name__ == "__main__":
	unittest.main()
