"""Contracts for the sixteen-configuration ABBA4 trajectory animation."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection

from initial_conditions import GCInitialConfiguration
from potential import GC2DH5Potential
from simulation import Solution
from studies.abba4_configuration_comparison import (
	ABBA4_CONFIGURATION_VARIANTS,
	ABBA4ConfigurationVariant,
)
from visualization.abba4_configuration_comparison import (
	animate_abba4_configuration_trajectories,
)


class _CountingH5Potential(GC2DH5Potential):
	"""Small HDF5-style field that records vectorized animation evaluations."""

	def __init__(self) -> None:
		x = np.linspace(0.1, 0.8, 8)
		y = np.linspace(0.2, 0.9, 8)
		x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
		mean = 0.2 + 0.05 * x_mesh + 0.03 * y_mesh
		mode = np.asarray(
			0.01 * np.exp(1j * (2.0 * x_mesh - 0.5 * y_mesh)),
			dtype=np.complex128,
		)
		super().__init__(
			x,
			y,
			mean,
			mode[None, :, :],
			frequencies=np.asarray([2.0]),
			interpolation_order=3,
		)
		self.evaluation_count = 0

	def evaluate(
		self,
		t: float | np.ndarray,
		x: np.ndarray | None = None,
		y: np.ndarray | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> np.ndarray:
		"""Count one batch call before delegating to the real HDF5 semantics."""
		self.evaluation_count += 1
		return super().evaluate(t, x, y, dx=dx, dy=dy, dt=dt)


def _variants() -> tuple[ABBA4ConfigurationVariant, ...]:
	"""Return the concrete study variants in a non-visual input order."""
	return tuple(reversed(ABBA4_CONFIGURATION_VARIANTS))


def _solution(
	particle: int,
	variant: int,
	*,
	times: np.ndarray | None = None,
) -> Solution:
	"""Build one finite trajectory with a model-independent initial position."""
	time_values = np.asarray([0.0, 0.1, 0.2] if times is None else times, dtype=float)
	x0 = 0.18 + 0.045 * particle
	y0 = 0.28 + 0.035 * particle
	model_shift = 2e-4 * variant * time_values
	x = x0 + 0.015 * time_values + model_shift
	y = y0 - 0.010 * time_values - 0.5 * model_shift
	source = GCInitialConfiguration.from_components(
		x=np.asarray([x0]),
		y=np.asarray([y0]),
	)
	return Solution(
		t=time_values,
		states=np.vstack((x, y)),
		source=source,
	)


def _result() -> SimpleNamespace:
	"""Assemble a duck-typed 16-by-10 comparison result."""
	potential = _CountingH5Potential()
	variants = _variants()
	solutions = {
		variant.key: tuple(
			_solution(particle, variant_index)
			for particle in range(10)
		)
		for variant_index, variant in enumerate(variants)
	}
	return SimpleNamespace(
		variants=variants,
		solutions=solutions,
		potential=potential,
		dynamics=SimpleNamespace(effective_potential=potential),
	)


class ABBA4ConfigurationAnimationTests(unittest.TestCase):
	"""Verify layout, shared HDF5 work, trajectory updates, and validation."""

	def test_animation_facets_all_160_trajectories_over_one_shared_field(self) -> None:
		result = _result()
		animation = animate_abba4_configuration_trajectories(
			result,
			frames=3,
			interval=10,
			repeat=False,
		)
		artists = animation._func(2)
		animation._draw_was_started = True

		# Sixteen data axes share one seventeenth colorbar axis.
		self.assertEqual(len(animation._fig.axes), 17)
		data_axes = animation._fig.axes[:16]
		self.assertEqual(result.potential.evaluation_count, 1)
		for axis in data_axes:
			self.assertEqual(len(axis.images), 1)
			paths = [
				collection
				for collection in axis.collections
				if isinstance(collection, LineCollection)
			]
			markers = [
				collection
				for collection in axis.collections
				if isinstance(collection, PathCollection)
			]
			self.assertEqual(len(paths), 1)
			self.assertEqual(len(paths[0].get_segments()), 10)
			self.assertEqual(markers[0].get_offsets().shape, (10, 2))
			self.assertEqual(axis.get_xlim(), (result.potential.x[0], result.potential.x[-1]))
			self.assertEqual(axis.get_ylim(), (result.potential.y[0], result.potential.y[-1]))

		self.assertEqual(len(animation._fig.legends), 1)
		self.assertEqual(len(animation._fig.legends[0].get_texts()), 10)
		self.assertEqual(len(artists), 49)
		assert animation._fig._suptitle is not None
		self.assertIn("160 trajectories", animation._fig._suptitle.get_text())
		self.assertIn("phase", animation._fig._suptitle.get_text())
		plt.close(animation._fig)

	def test_animation_rejects_an_incomplete_or_misaligned_result(self) -> None:
		result = _result()
		result.variants = result.variants[:-1]
		with self.assertRaisesRegex(ValueError, "exactly 16"):
			animate_abba4_configuration_trajectories(result, frames=3)

		result = _result()
		key = result.variants[0].key
		result.solutions[key] = result.solutions[key][:-1]
		with self.assertRaisesRegex(ValueError, "exactly 10"):
			animate_abba4_configuration_trajectories(result, frames=3)

		result = _result()
		key = result.variants[0].key
		trajectories = list(result.solutions[key])
		trajectories[0] = _solution(
			0,
			0,
			times=np.asarray([0.0, 0.11, 0.2]),
		)
		result.solutions[key] = tuple(trajectories)
		with self.assertRaisesRegex(ValueError, "saved-time grid"):
			animate_abba4_configuration_trajectories(result, frames=3)


if __name__ == "__main__":
	unittest.main()
