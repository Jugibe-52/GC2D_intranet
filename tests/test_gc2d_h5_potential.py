"""Contracts for primary GC2D HDF5 potential loading and dynamics use."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import GC2DH5Potential, Potential, load_gc2d_h5_potential
from simulation import ABBA4Implicit1, InitialValueProblem, SimulationRequest, simulate


def _h5_interpolate(
	x: np.ndarray,
	y: np.ndarray,
	field: np.ndarray,
	xi: np.ndarray,
	yi: np.ndarray,
	*,
	order: int = 3,
	dx: int = 0,
	dy: int = 0,
) -> np.ndarray:
	"""Evaluate one field with the GC2D HDF5 zero-padded spline recipe."""
	padding = order + 1
	x_extended = np.pad(
		x,
		(padding, padding),
		mode="linear_ramp",
		end_values=(x[0] - padding * (x[1] - x[0]), x[-1] + padding * (x[1] - x[0])),
	)
	y_extended = np.pad(
		y,
		(padding, padding),
		mode="linear_ramp",
		end_values=(y[0] - padding * (y[1] - y[0]), y[-1] + padding * (y[1] - y[0])),
	)
	field_extended = np.pad(
		field,
		((padding, padding), (padding, padding)),
		mode="constant",
		constant_values=0,
	)
	real = RectBivariateSpline(
		x_extended,
		y_extended,
		field_extended.real,
		kx=order,
		ky=order,
	).ev(xi, yi, dx=dx, dy=dy)
	imag = RectBivariateSpline(
		x_extended,
		y_extended,
		field_extended.imag,
		kx=order,
		ky=order,
	).ev(xi, yi, dx=dx, dy=dy)
	return np.asarray(real + 1j * imag)


def _h5_resample(
	x: np.ndarray,
	y: np.ndarray,
	field: np.ndarray,
	*,
	nx: int,
	ny: int,
) -> np.ndarray:
	"""Perform the first of the HDF5 loader's two interpolation stages."""
	xi = np.linspace(x[0], x[-1], nx)
	yi = np.linspace(y[0], y[-1], ny)
	x_mesh, y_mesh = np.meshgrid(xi, yi, indexing="ij")
	return _h5_interpolate(x, y, field, x_mesh, y_mesh)


class GC2DH5PotentialTests(unittest.TestCase):
	"""Verify GC2D HDF5 semantics against the current field contracts."""

	def setUp(self) -> None:
		"""Create a compact square HDF5 fixture with deliberately ordered modes."""
		self.temporary_directory = tempfile.TemporaryDirectory(dir="/tmp")
		self.path = Path(self.temporary_directory.name) / "potential.h5"
		self.x = 0.10 + 0.05 * np.arange(6)
		self.y = 0.02 + 0.05 * np.arange(6)
		row, column = np.indices((6, 6))
		self.mean = 2.0 + 0.7 * row + 0.2 * column
		self.low_mode = (row + 2.0 * column) + 1j * (0.3 * row - column)
		self.high_mode = 10.0 * (2.0 * row + column) + 1j * (row - column)
		negative_mode = 1000.0 * (row + column)
		self.frequencies = np.asarray([0.0, -11.0, 3.0, 7.0])
		with h5py.File(self.path, "w") as h5:
			h5.create_dataset("Rcells", data=self.x)
			h5.create_dataset("Zcells", data=self.y)
			h5.create_dataset("freqs", data=self.frequencies)
			h5.create_dataset(
				"fields",
				data=np.asarray(
					[self.mean, negative_mode, self.low_mode, self.high_mode],
					dtype=np.complex128,
				),
			)
			h5.attrs["shot"] = 42

	def tearDown(self) -> None:
		"""Release the temporary HDF5 fixture."""
		self.temporary_directory.cleanup()

	def test_defaults_select_mean_and_dominant_mode_with_B_1_5(self) -> None:
		"""Use the primary-file defaults when no loader options are supplied."""
		potential = load_gc2d_h5_potential(self.path)
		normalization = 2.0 * np.pi * 7.0 * 1.5

		self.assertAlmostEqual(potential.normalization_factor, normalization)
		np.testing.assert_array_equal(potential.source_field_indices, [3])
		np.testing.assert_allclose(potential.frequencies, [7.0])
		np.testing.assert_allclose(potential.mean_value, self.mean / normalization)
		assert potential.fluctuations is not None
		np.testing.assert_allclose(
			potential.fluctuations[0],
			self.high_mode / normalization,
		)

	def test_filter_sort_selection_normalization_and_positive_phase(self) -> None:
		"""Match the HDF5 meaning of indices, f0, B and exp(+i f t)."""
		B = 1.5
		potential = load_gc2d_h5_potential(
			self.path,
			B=B,
			indx=(0, 2, 1),
			interpolation_order=3,
		)
		normalization = 2.0 * np.pi * 7.0 * B

		self.assertIsInstance(potential, GC2DH5Potential)
		self.assertIsInstance(potential, Potential)
		self.assertAlmostEqual(potential.normalization_factor, normalization)
		np.testing.assert_array_equal(potential.source_field_indices, [2, 3])
		np.testing.assert_allclose(potential.frequencies, [3.0, 7.0])
		np.testing.assert_allclose(potential.freqs, potential.frequencies)
		# No transpose is applied: the deliberately asymmetric raw array is retained.
		np.testing.assert_allclose(potential.mean_value, self.mean / normalization)
		assert potential.fluctuations is not None
		np.testing.assert_allclose(potential.fluctuations[0], self.low_mode / normalization)
		np.testing.assert_allclose(potential.fluctuations[1], self.high_mode / normalization)
		self.assertEqual(int(potential.attributes["shot"]), 42)
		self.assertFalse(potential.frequencies.flags.writeable)

		time = 0.037
		expected_dynamic = 2.0 * np.real(
			self.low_mode / normalization * np.exp(1j * 3.0 * time)
			+ self.high_mode / normalization * np.exp(1j * 7.0 * time)
		)
		np.testing.assert_allclose(potential.dynamic_part(time), expected_dynamic)
		np.testing.assert_allclose(
			potential.evaluate(time),
			self.mean / normalization + expected_dynamic,
		)

	def test_denoising_and_resampling_match_both_hdf5_interpolation_stages(self) -> None:
		"""Keep filtering before the inclusive-grid resampling used by GC2D."""
		B = 2.0
		sigma = 0.6
		potential = load_gc2d_h5_potential(
			self.path,
			B=B,
			indx=(0, 1),
			nx=8,
			ny=8,
			denoising=True,
			sigma=sigma,
			interpolation_order=3,
		)
		normalization = 2.0 * np.pi * 7.0 * B
		expected_mean = _h5_resample(
			self.x,
			self.y,
			ndimage.gaussian_filter(self.mean / normalization, sigma=sigma),
			nx=8,
			ny=8,
		).real
		expected_mode = _h5_resample(
			self.x,
			self.y,
			ndimage.gaussian_filter(self.high_mode.real / normalization, sigma=sigma)
			+ 1j
			* ndimage.gaussian_filter(self.high_mode.imag / normalization, sigma=sigma),
			nx=8,
			ny=8,
		)

		np.testing.assert_allclose(potential.x, np.linspace(self.x[0], self.x[-1], 8))
		np.testing.assert_allclose(potential.y, np.linspace(self.y[0], self.y[-1], 8))
		np.testing.assert_allclose(potential.mean_value, expected_mean)
		assert potential.fluctuations is not None
		np.testing.assert_allclose(potential.fluctuations[0], expected_mode)

		# Runtime evaluation is the second zero-padded interpolation stage.
		query_x = np.asarray([potential.x[2] + 0.01])
		query_y = np.asarray([potential.y[4] - 0.008])
		time = 0.013
		expected = _h5_interpolate(
			potential.x,
			potential.y,
			expected_mean.astype(np.complex128),
			query_x,
			query_y,
		).real
		expected += 2.0 * np.real(
			_h5_interpolate(
				potential.x,
				potential.y,
				expected_mode,
				query_x,
				query_y,
			)
			* np.exp(1j * 7.0 * time)
		)
		np.testing.assert_allclose(potential.evaluate(time, query_x, query_y), expected)

	def test_spatial_hessians_time_derivative_and_clipping(self) -> None:
		"""Expose exact spline Hessians and the positive-frequency time derivative."""
		potential = load_gc2d_h5_potential(
			self.path,
			B=1.5,
			indx=(0, 1),
			interpolation_order=3,
		)
		assert potential.mean_value is not None
		assert potential.fluctuations is not None
		frequency = float(potential.frequencies[0])
		time = 0.021
		# Both out-of-domain coordinates are clipped before spline evaluation.
		query_x = np.asarray([self.x[2] + 0.013, self.x[-1] + 1.0])
		query_y = np.asarray([self.y[3] - 0.009, self.y[0] - 1.0])
		clipped_x = np.clip(query_x, self.x[0], self.x[-1])
		clipped_y = np.clip(query_y, self.y[0], self.y[-1])
		for dx, dy in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
			mean = _h5_interpolate(
				self.x,
				self.y,
				potential.mean_value.astype(np.complex128),
				clipped_x,
				clipped_y,
				dx=dx,
				dy=dy,
			).real
			mode = _h5_interpolate(
				self.x,
				self.y,
				potential.fluctuations[0],
				clipped_x,
				clipped_y,
				dx=dx,
				dy=dy,
			)
			expected = mean + 2.0 * np.real(mode * np.exp(1j * frequency * time))
			np.testing.assert_allclose(
				potential.evaluate(time, query_x, query_y, dx=dx, dy=dy),
				expected,
			)

		mode = _h5_interpolate(
			self.x,
			self.y,
			potential.fluctuations[0],
			clipped_x,
			clipped_y,
		)
		expected_time_derivative = 2.0 * np.real(
			mode * (1j * frequency) * np.exp(1j * frequency * time)
		)
		np.testing.assert_allclose(
			potential.evaluate(time, query_x, query_y, dt=1),
			expected_time_derivative,
		)
		ex, ey = potential.electric_field(time, query_x, query_y)
		np.testing.assert_allclose(ex, -potential.evaluate(time, query_x, query_y, dx=1))
		np.testing.assert_allclose(ey, -potential.evaluate(time, query_x, query_y, dy=1))

	def test_zero_gyroaverage_and_abba4_implicit_1_are_compatible(self) -> None:
		"""Pass the strict Potential check and supply Hessians to implicit ABBA4."""
		potential = load_gc2d_h5_potential(
			self.path,
			B=1.5,
			indx=(0, 1),
			interpolation_order=3,
		)
		self.assertIs(potential.gyroaverage(0.0), potential)
		averaged = potential.gyroaverage(0.01)
		self.assertIsInstance(averaged, GC2DH5Potential)
		self.assertTrue(np.all(np.isfinite(averaged.evaluate(0.02))))

		angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
		configuration = GCInitialConfiguration.from_components(
			x=0.22 + 0.005 * np.cos(angles),
			y=0.15 + 0.005 * np.sin(angles),
		)
		dynamics = GuidingCenterDynamics(potential, rho=0.0)
		jacobians = dynamics.particle_vector_field_jacobians(
			0.0,
			configuration.initial_state,
		)
		self.assertEqual(jacobians.shape, (5, 2, 2))
		self.assertTrue(np.all(np.isfinite(jacobians)))

		solution = simulate(
			InitialValueProblem(dynamics, configuration),
			ABBA4Implicit1(newton_max_iterations=20),
			SimulationRequest.uniform(
				t_span=(0.0, 1e-3),
				max_step=5e-4,
				sample_count=3,
			),
		)
		self.assertEqual(solution.states.shape, (10, 3))
		self.assertTrue(np.all(np.isfinite(solution.states)))

	def test_invalid_selection_and_incomplete_resampling_are_rejected(self) -> None:
		"""Give concise errors for common HDF5-loader configuration mistakes."""
		with self.assertRaisesRegex(ValueError, "range"):
			load_gc2d_h5_potential(self.path, indx=(0, 3))
		with self.assertRaisesRegex(ValueError, "both"):
			load_gc2d_h5_potential(self.path, nx=8)
		with self.assertRaisesRegex(ValueError, "non-zero"):
			load_gc2d_h5_potential(self.path, B=0.0)


if __name__ == "__main__":
	unittest.main()
