"""Processed HDF5 fields survive serialization and spawned study execution."""

from dynamics.gc import GuidingCenterDynamics

from pathlib import Path
import pickle
import tempfile
import unittest
from solvers.nonlinear import NewtonSolver

import numpy as np

from problem import InitialValueProblem

from request import SimulationRequest

from methods.palindromic.bm4_implicit import BM4Implicit
from potential.load import GC2DH5Metadata
from potential.grid import Grid
from potential.potential import Potential
from simulation.runner import simulate
from studies._potential_snapshot import _H5PotentialSnapshot
from studies.bm4_parallel_recurrence import (
	ParallelBM4RecurrenceConfig,
	run_parallel_bm4_recurrence,
)


def _potential(source_path: Path, *, spatial_extension: str = "periodic") -> Potential:
	"""Create a small processed complex field whose source need not exist."""
	grid = Grid.periodic(8, 10)
	x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
	return Potential(
		grid,
		mean=0.01 * np.cos(x) * np.sin(y),
		modes=(0.002 * np.exp(1j * (x + y)))[None],
		frequencies=np.asarray([0.7]),
		interpolation_order=3 if spatial_extension == "reflected" else 4,
		spatial_extension=spatial_extension,
		metadata=GC2DH5Metadata(
			source_field_indices=np.asarray([3]),
			source_x=grid.x,
			source_y=grid.y,
			source_frequencies=np.asarray([7.0]),
			characteristic_length=0.3,
			characteristic_period=0.2,
			normalization_factor=2.0,
			attributes={"shot": np.asarray([42])},
			source_path=source_path,
		),
	)


class PotentialSnapshotTests(unittest.TestCase):
	"""Keep worker reconstruction independent of the original HDF5 file."""

	def test_pickle_round_trip_preserves_fields_derivatives_and_metadata(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			source = Path(directory) / "absent.h5"
			original = _potential(source)
			snapshot = pickle.loads(pickle.dumps(_H5PotentialSnapshot.from_potential(original)))
			restored = snapshot.restore()
			self.assertFalse(source.exists())
		self.assertEqual(restored.grid, original.grid)
		self.assertEqual(restored.interpolation_order, original.interpolation_order)
		self.assertEqual(restored.spatial_extension, original.spatial_extension)
		for name in ("mean", "modes", "frequencies"):
			actual = getattr(restored, name)
			np.testing.assert_array_equal(actual, getattr(original, name))
			self.assertFalse(actual.flags.writeable)
		for name in GC2DH5Metadata.__dataclass_fields__:
			actual = getattr(restored.metadata, name)
			expected = getattr(original.metadata, name)
			if name == "attributes":
				self.assertEqual(actual.keys(), expected.keys())
				for key in actual:
					np.testing.assert_array_equal(actual[key], expected[key])
			else:
				np.testing.assert_equal(actual, expected)
		x = np.asarray([-0.01, 0.4, 2.0 * np.pi + 0.01])
		y = np.asarray([2.0 * np.pi + 0.03, 0.7, -0.02])
		for derivative in ({}, {"dx": 1}, {"dy": 2}, {"dx": 1, "dt": 1}):
			np.testing.assert_array_equal(
				restored.evaluate(0.3, x, y, **derivative),
				original.evaluate(0.3, x, y, **derivative),
			)

	def test_snapshot_rejects_a_potential_without_hdf5_provenance(self) -> None:
		with self.assertRaisesRegex(TypeError, "GC2D HDF5 metadata"):
			_H5PotentialSnapshot.from_potential(Potential(Grid.periodic(4, 4)))

	def test_reflected_gyroaverage_preserves_independent_endpoint_samples(self) -> None:
		original = _potential(Path("absent.h5"), spatial_extension="reflected").gyroaverage(0.37)
		restored = pickle.loads(pickle.dumps(_H5PotentialSnapshot.from_potential(original))).restore()
		self.assertEqual(restored.spatial_extension, "reflected")
		self.assertEqual(restored.spatial_period, 2 * original.grid.period)
		np.testing.assert_array_equal(restored.reflected_samples, original.reflected_samples)
		x = np.array([-0.1, 0.5, original.grid.period, 1.4 * original.grid.period])
		y = np.array([0.4, -0.2, 0.9, original.grid.period])
		for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1)):
			np.testing.assert_array_equal(restored.evaluate(.37, x, y, dx=dx, dy=dy),
				original.evaluate(.37, x, y, dx=dx, dy=dy))

	def test_untagged_snapshot_retains_historical_periodic_interpretation(self) -> None:
		original = _potential(Path("absent.h5"))
		snapshot = _H5PotentialSnapshot.from_potential(original)
		object.__delattr__(snapshot, "spatial_extension")
		object.__delattr__(snapshot, "reflected_samples")
		restored = snapshot.restore()
		self.assertEqual(restored.spatial_extension, "periodic")
		np.testing.assert_array_equal(restored.evaluate_grid(.2), original.evaluate_grid(.2))

	def test_bm4_spawned_workers_match_direct_trajectories(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			potential = _potential(Path(directory) / "absent.h5")
			configuration = GuidingCenterDynamics.coordinates.pack_components(np.asarray([1.1, 1.7]), np.asarray([1.3, 0.8]))
			config = ParallelBM4RecurrenceConfig(
				particle_count=2, t_span=(0.0, 1.0), steps_per_cycle=4,
				saved_samples_per_cycle=2, rho=0.0, worker_count=2, progress=False,
			)
			result = run_parallel_bm4_recurrence(potential, configuration, config=config)
			dynamics = GuidingCenterDynamics(potential, rho=config.rho)
			request = SimulationRequest.uniform(
				t_span=config.t_span,
				sample_count=config.output_sample_count,
			)
			for particle, (x, y) in enumerate(result.initial_positions):
				problem = InitialValueProblem(dynamics, GuidingCenterDynamics.coordinates.pack_components(np.asarray([x]), np.asarray([y])), layout=GuidingCenterDynamics.coordinates)
				solution = simulate(
					problem.with_request(request),
					BM4Implicit(step=config.integration_step,
						solver=NewtonSolver(config.absolute_tolerance, config.relative_tolerance, config.max_iterations, jacobian_method='analytic'),
					),
				)
				x_values, y_values = solution.positions()
				np.testing.assert_array_equal(result.times, solution.t)
				np.testing.assert_allclose(
					result.positions[particle], np.stack((x_values[0], y_values[0])),
					rtol=0.0, atol=1e-14,
				)
				self.assertEqual(
					result.total_newton_iterations[particle],
					int(np.sum(solution.diagnostics["nonlinear_iterations"])),
				)


if __name__ == "__main__":
	unittest.main()
