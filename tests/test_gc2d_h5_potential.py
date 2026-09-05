"""Contracts for primary GC2D HDF5 potential loading and dynamics use."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from diagnostics import central_difference_jacobian
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import GC2DH5Potential, Potential, load_gc2d_h5_potential
from simulation import ABBA4Implicit, InitialValueProblem, SimulationRequest, simulate
from simulation.methods._fully_extended import (
	_extended_vector_field,
	_extended_vector_field_jacobian,
)


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
	"""Evaluate one field with the GC2D HDF5 periodic spline recipe."""
	margin = order + 1
	padding = (margin, margin + 1)
	x_extended = np.pad(
		x,
		padding,
		mode="linear_ramp",
		end_values=(
			x[0] - padding[0] * (x[1] - x[0]),
			x[-1] + padding[1] * (x[1] - x[0]),
		),
	)
	y_extended = np.pad(
		y,
		padding,
		mode="linear_ramp",
		end_values=(
			y[0] - padding[0] * (y[1] - y[0]),
			y[-1] + padding[1] * (y[1] - y[0]),
		),
	)
	field_extended = np.pad(
		field,
		(padding, padding),
		mode="wrap",
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
	"""Perform periodic HDF5 resampling without duplicating an endpoint."""
	xi = x[0] + x.size * (x[1] - x[0]) * np.arange(nx) / nx
	yi = y[0] + y.size * (y[1] - y[0]) * np.arange(ny) / ny
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
		self.characteristic_length = 0.30
		self.runtime_x = (
			2.0 * np.pi * (self.x - self.x[0]) / self.characteristic_length
		)
		self.runtime_y = (
			2.0 * np.pi * (self.y - self.y[0]) / self.characteristic_length
		)
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
		length_scale = 0.06
		normalization = 7.0 * length_scale**2 * 1.5 / (2.0 * np.pi) ** 2

		self.assertAlmostEqual(potential.normalization_factor, normalization)
		np.testing.assert_array_equal(potential.source_field_indices, [3])
		np.testing.assert_allclose(potential.source_frequencies, [7.0])
		np.testing.assert_allclose(potential.frequencies, [1.0])
		self.assertEqual(potential.characteristic_length, length_scale)
		self.assertAlmostEqual(potential.characteristic_frequency, 7.0)
		self.assertAlmostEqual(potential.characteristic_period, 2.0 * np.pi / 7.0)
		np.testing.assert_allclose(potential.source_x, self.x)
		np.testing.assert_allclose(potential.source_y, self.y)
		np.testing.assert_allclose(
			potential.x,
			2.0 * np.pi * (self.x - self.x[0]) / length_scale,
		)
		np.testing.assert_allclose(potential.mean_value, self.mean / normalization)
		assert potential.fluctuations is not None
		np.testing.assert_allclose(
			potential.fluctuations[0],
			self.high_mode / normalization,
		)
		query_x = np.asarray([0.41])
		query_y = np.asarray([1.73])
		np.testing.assert_allclose(
			potential.evaluate(0.37, query_x, query_y),
			potential.evaluate(0.37 + 1.0, query_x, query_y),
		)

	def test_filter_sort_selection_normalization_and_positive_phase(self) -> None:
		"""Match HDF5 indices, normalization, and cycle-based positive phase."""
		B = 1.5
		potential = load_gc2d_h5_potential(
			self.path,
			B=B,
			characteristic_length=self.characteristic_length,
			indx=(0, 2, 1),
			interpolation_order=3,
		)
		normalization = (
			7.0 * self.characteristic_length**2 * B / (2.0 * np.pi) ** 2
		)

		self.assertIsInstance(potential, GC2DH5Potential)
		self.assertIsInstance(potential, Potential)
		self.assertAlmostEqual(potential.normalization_factor, normalization)
		np.testing.assert_array_equal(potential.source_field_indices, [2, 3])
		np.testing.assert_allclose(potential.source_frequencies, [3.0, 7.0])
		np.testing.assert_allclose(potential.frequencies, [3.0 / 7.0, 1.0])
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
			self.low_mode
			/ normalization
			* np.exp(2j * np.pi * (3.0 / 7.0) * time)
			+ self.high_mode / normalization * np.exp(2j * np.pi * time)
		)
		np.testing.assert_allclose(potential.dynamic_part(time), expected_dynamic)
		np.testing.assert_allclose(
			potential.evaluate(time),
			self.mean / normalization + expected_dynamic,
		)

	def test_explicit_characteristic_frequency_controls_time_and_amplitude(self) -> None:
		"""Honor an explicitly supplied source frequency instead of the dominant mode."""
		frequency_scale = 14.0
		potential = load_gc2d_h5_potential(
			self.path,
			B=1.5,
			characteristic_length=self.characteristic_length,
			characteristic_frequency=frequency_scale,
			indx=(0, 1),
		)
		normalization = (
			frequency_scale
			* self.characteristic_length**2
			* 1.5
			/ (2.0 * np.pi) ** 2
		)

		self.assertAlmostEqual(potential.characteristic_frequency, frequency_scale)
		self.assertAlmostEqual(
			potential.characteristic_period,
			2.0 * np.pi / frequency_scale,
		)
		self.assertAlmostEqual(potential.normalization_factor, normalization)
		np.testing.assert_allclose(potential.frequencies, [0.5])
		np.testing.assert_allclose(potential.mean_value, self.mean / normalization)

	def test_explicit_frequency_normalizes_a_mean_only_file(self) -> None:
		"""Use the requested article scales even when no oscillatory mode is stored."""
		path = Path(self.temporary_directory.name) / "mean-only.h5"
		with h5py.File(path, "w") as h5:
			h5.create_dataset("Rcells", data=self.x)
			h5.create_dataset("Zcells", data=self.y)
			h5.create_dataset("freqs", data=np.asarray([0.0]))
			h5.create_dataset(
				"fields",
				data=np.asarray([self.mean], dtype=np.complex128),
			)
		potential = load_gc2d_h5_potential(
			path,
			B=2.0,
			characteristic_length=self.characteristic_length,
			characteristic_frequency=5.0,
			indx=(0,),
		)
		normalization = (
			5.0 * self.characteristic_length**2 * 2.0 / (2.0 * np.pi) ** 2
		)

		self.assertAlmostEqual(potential.normalization_factor, normalization)
		self.assertAlmostEqual(potential.characteristic_period, 2.0 * np.pi / 5.0)
		np.testing.assert_allclose(potential.mean_value, self.mean / normalization)
		self.assertEqual(potential.frequencies.size, 0)

	def test_denoising_and_resampling_match_both_hdf5_interpolation_stages(self) -> None:
		"""Keep filtering before periodic resampling of dimensionless fields."""
		B = 2.0
		sigma = 0.6
		potential = load_gc2d_h5_potential(
			self.path,
			B=B,
			characteristic_length=self.characteristic_length,
			indx=(0, 1),
			nx=8,
			ny=8,
			denoising=True,
			sigma=sigma,
			interpolation_order=3,
		)
		normalization = (
			7.0 * self.characteristic_length**2 * B / (2.0 * np.pi) ** 2
		)
		expected_mean = _h5_resample(
			self.runtime_x,
			self.runtime_y,
			ndimage.gaussian_filter(self.mean / normalization, sigma=sigma),
			nx=8,
			ny=8,
		).real
		expected_mode = _h5_resample(
			self.runtime_x,
			self.runtime_y,
			ndimage.gaussian_filter(self.high_mode.real / normalization, sigma=sigma)
			+ 1j
			* ndimage.gaussian_filter(self.high_mode.imag / normalization, sigma=sigma),
			nx=8,
			ny=8,
		)

		expected_axis = 2.0 * np.pi * np.arange(8) / 8
		np.testing.assert_allclose(potential.x, expected_axis)
		np.testing.assert_allclose(potential.y, expected_axis)
		np.testing.assert_allclose(potential.mean_value, expected_mean)
		assert potential.fluctuations is not None
		np.testing.assert_allclose(potential.fluctuations[0], expected_mode)

		# Runtime evaluation is the second periodic interpolation stage.
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
			* np.exp(2j * np.pi * time)
		)
		np.testing.assert_allclose(potential.evaluate(time, query_x, query_y), expected)

	def test_spatial_hessians_time_derivative_and_periodic_wrapping(self) -> None:
		"""Expose exact spline derivatives on the periodic normalized domain."""
		potential = load_gc2d_h5_potential(
			self.path,
			B=1.5,
			characteristic_length=self.characteristic_length,
			indx=(0, 1),
			interpolation_order=3,
		)
		assert potential.mean_value is not None
		assert potential.fluctuations is not None
		frequency = float(potential.frequencies[0])
		time = 0.021
		period = potential.grid.period
		query_x = np.asarray([potential.x[2] + 0.013, potential.x[-1] + 1.0])
		query_y = np.asarray([potential.y[3] - 0.009, potential.y[0] - 1.0])
		wrapped_x = (query_x - potential.x[0]) % period + potential.x[0]
		wrapped_y = (query_y - potential.y[0]) % period + potential.y[0]
		for dx, dy in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
			mean = _h5_interpolate(
				potential.x,
				potential.y,
				potential.mean_value.astype(np.complex128),
				wrapped_x,
				wrapped_y,
				dx=dx,
				dy=dy,
			).real
			mode = _h5_interpolate(
				potential.x,
				potential.y,
				potential.fluctuations[0],
				wrapped_x,
				wrapped_y,
				dx=dx,
				dy=dy,
			)
			angular_frequency = 2.0 * np.pi * frequency
			expected = mean + 2.0 * np.real(
				mode * np.exp(1j * angular_frequency * time)
			)
			np.testing.assert_allclose(
				potential.evaluate(time, query_x, query_y, dx=dx, dy=dy),
				expected,
			)

		# Independent grid samples retain their direct spline derivatives.
		endpoint_x = np.asarray((potential.x[0], potential.x[-1]))
		endpoint_y = np.full(2, potential.y[3] - 0.009)
		endpoint_mean_x = _h5_interpolate(
			potential.x,
			potential.y,
			potential.mean_value.astype(np.complex128),
			endpoint_x,
			endpoint_y,
			dx=1,
		).real
		endpoint_mode_x = _h5_interpolate(
			potential.x,
			potential.y,
			potential.fluctuations[0],
			endpoint_x,
			endpoint_y,
			dx=1,
		)
		expected_endpoint_x = endpoint_mean_x + 2.0 * np.real(
			endpoint_mode_x * np.exp(1j * angular_frequency * time)
		)
		np.testing.assert_allclose(
			potential.evaluate(time, endpoint_x, endpoint_y, dx=1),
			expected_endpoint_x,
		)

		mode = _h5_interpolate(
			potential.x,
			potential.y,
			potential.fluctuations[0],
			wrapped_x,
			wrapped_y,
		)
		expected_time_derivative = 2.0 * np.real(
			mode
			* (1j * angular_frequency)
			* np.exp(1j * angular_frequency * time)
		)
		np.testing.assert_allclose(
			potential.evaluate(time, query_x, query_y, dt=1),
			expected_time_derivative,
		)
		ex, ey = potential.electric_field(time, query_x, query_y)
		np.testing.assert_allclose(ex, -potential.evaluate(time, query_x, query_y, dx=1))
		np.testing.assert_allclose(ey, -potential.evaluate(time, query_x, query_y, dy=1))

	def test_outside_domain_gc_jacobian_matches_periodic_vector_field(self) -> None:
		"""Differentiate the periodically wrapped field seen by GC dynamics."""
		potential = load_gc2d_h5_potential(
			self.path,
			B=1.5,
			characteristic_length=self.characteristic_length,
			indx=(0, 2, 1),
			interpolation_order=3,
		)
		dynamics = GuidingCenterDynamics(potential, rho=0.0)
		time = 0.029
		period = potential.grid.period
		states = (
			np.asarray((potential.x[-1] + 0.2, potential.y[3] - 0.007)),
			np.asarray((potential.x[2] + 0.011, potential.y[0] - 0.2)),
			np.asarray((potential.x[-1] + 0.2, potential.y[0] - 0.2)),
		)
		for state in states:
			with self.subTest(state=state):
				analytic = dynamics.particle_vector_field_jacobians(time, state)[0]
				numerical = central_difference_jacobian(
					lambda candidate: dynamics.vector_field(time, candidate),
					state,
				)
				np.testing.assert_allclose(
					analytic,
					numerical,
					rtol=2e-6,
					atol=2e-7,
				)
				np.testing.assert_allclose(
					dynamics.vector_field(time, state),
					dynamics.vector_field(time, state + period),
				)

	def test_multifrequency_second_time_derivative_and_extended_jacobian(
		self,
	) -> None:
		"""Differentiate a mean plus two non-unit HDF5 frequencies exactly."""
		potential = load_gc2d_h5_potential(
			self.path,
			B=1.5,
			characteristic_length=self.characteristic_length,
			indx=(0, 2, 1),
			interpolation_order=3,
		)
		assert potential.mean_value is not None
		assert potential.fluctuations is not None
		time = 0.037
		query_x = np.asarray([potential.x[2] + 0.013])
		query_y = np.asarray([potential.y[3] - 0.009])
		expected_second = np.zeros_like(query_x)
		for field, frequency in zip(
			potential.fluctuations,
			potential.frequencies,
			strict=True,
		):
			mode = _h5_interpolate(
				potential.x,
				potential.y,
				field,
				query_x,
				query_y,
			)
			angular_frequency = 2.0 * np.pi * float(frequency)
			expected_second += 2.0 * np.real(
				mode
				* (1j * angular_frequency) ** 2
				* np.exp(1j * angular_frequency * time)
			)
		second = potential.evaluate(time, query_x, query_y, dt=2)
		np.testing.assert_allclose(second, expected_second)
		self.assertFalse(
			np.allclose(second, -potential.evaluate(time, query_x, query_y))
		)

		dynamics = GuidingCenterDynamics(potential, rho=0.0)
		extended_state = np.asarray(
			(query_x[0], query_y[0], time, 0.31),
			dtype=float,
		)
		analytic = _extended_vector_field_jacobian(dynamics, extended_state)
		numerical = central_difference_jacobian(
			lambda state: _extended_vector_field(dynamics, state),
			extended_state,
		)
		np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-7)

	def test_zero_gyroaverage_and_abba4_implicit_are_compatible(self) -> None:
		"""Pass the strict Potential check and supply Hessians to implicit ABBA4."""
		potential = load_gc2d_h5_potential(
			self.path,
			B=1.5,
			characteristic_length=self.characteristic_length,
			indx=(0, 1),
			interpolation_order=3,
		)
		self.assertIs(potential.gyroaverage(0.0), potential)
		averaged = potential.gyroaverage(0.01)
		self.assertIsInstance(averaged, GC2DH5Potential)
		self.assertEqual(
			averaged.characteristic_frequency,
			potential.characteristic_frequency,
		)
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
			ABBA4Implicit(newton_max_iterations=20),
			SimulationRequest.uniform(
				t_span=(0.0, 1e-3 / (2.0 * np.pi)),
				max_step=5e-4 / (2.0 * np.pi),
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
		with self.assertRaisesRegex(ValueError, "characteristic_length"):
			load_gc2d_h5_potential(self.path, characteristic_length=0.0)
		with self.assertRaisesRegex(ValueError, "characteristic_frequency"):
			load_gc2d_h5_potential(self.path, characteristic_frequency=-1.0)


if __name__ == "__main__":
	unittest.main()
