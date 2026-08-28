"""Contracts for optional notebook presentation helpers."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import Animation, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.quiver import Quiver
from matplotlib.text import Text

from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import Solution
from visualization import (
	animate_gc_particle_solution,
	animate_gc_particle_trajectories,
	display_animation,
)


class NotebookPresentationTests(unittest.TestCase):
	"""Keep animation embedding usable when external encoders fail."""

	def test_display_animation_falls_back_when_ffmpeg_fails(self) -> None:
		figure = plt.figure()
		animation = FuncAnimation(figure, lambda _frame: (), frames=(0,))
		# Encoding is mocked below, so mark the synthetic animation as rendered to
		# prevent Matplotlib's deletion warning from obscuring the assertion.
		animation._draw_was_started = True
		error = subprocess.CalledProcessError(1, ("ffmpeg",))

		with (
			patch(
				"visualization.notebooks.mpl_animation.writers.is_available",
				return_value=True,
			),
			patch.object(Animation, "to_html5_video", side_effect=error),
			patch.object(
				Animation,
				"to_jshtml",
				return_value="<div>JavaScript animation</div>",
			) as javascript,
			patch("IPython.display.display") as display,
		):
			display_animation(animation)

		javascript.assert_called_once_with(default_mode="once")
		display.assert_called_once()
		self.assertFalse(plt.fignum_exists(figure.number))

	def test_gc_particle_animation_can_show_the_electric_field(self) -> None:
		potential = Potential.random(
			A=0.1,
			M=2,
			nx=8,
			ny=8,
			seed=7,
			interpolation_order=3,
		)
		source = GCInitialConfiguration.from_components(
			x=np.asarray([1.0]),
			y=np.asarray([1.2]),
		)
		solution = Solution(
			t=np.asarray([0.0, 0.1, 0.2]),
			states=np.asarray(
				[
					[1.0, 1.1, 1.2],
					[1.2, 1.25, 1.3],
				]
			),
			source=source,
		)

		animation = animate_gc_particle_solution(
			potential,
			solution,
			frames=3,
			show_electric_field=True,
			frame_annotations=("initial", "hyperbolic", "elliptic"),
		)
		artists = animation._func(1)
		animation._draw_was_started = True

		self.assertTrue(any(isinstance(artist, Quiver) for artist in artists))
		self.assertTrue(
			any(
				isinstance(artist, Text) and artist.get_text() == "hyperbolic"
				for artist in artists
			)
		)
		plt.close(animation._fig)

	def test_gc_trajectory_animation_rotates_field_into_perpendicular_drift(
		self,
	) -> None:
		potential = Potential.random(
			A=0.1,
			M=2,
			nx=8,
			ny=8,
			seed=7,
			interpolation_order=3,
		)
		source = GCInitialConfiguration.from_components(
			x=np.asarray([1.0, 1.4]),
			y=np.asarray([1.2, 1.6]),
		)
		solution = Solution(
			t=np.asarray([0.0, 0.1, 0.2]),
			states=np.asarray(
				[
					[1.0, 1.1, 1.2],
					[1.4, 1.5, 1.6],
					[1.2, 1.25, 1.3],
					[1.6, 1.65, 1.7],
				]
			),
			source=source,
		)

		animation = animate_gc_particle_trajectories(
			potential,
			solution,
			frames=3,
			show_electric_field=True,
			show_perpendicular_drift=True,
		)
		artists = animation._func(1)
		animation._draw_was_started = True
		quivers = [artist for artist in artists if isinstance(artist, Quiver)]
		paths = [artist for artist in artists if isinstance(artist, LineCollection)]

		self.assertEqual(len(quivers), 2)
		self.assertEqual(len(paths), 1)
		x, y = solution.positions()
		field_x, field_y = potential.electric_field(0.1, x[:, 1], y[:, 1])
		drift = quivers[-1]
		np.testing.assert_allclose(drift.U, field_y)
		np.testing.assert_allclose(drift.V, -field_x)
		np.testing.assert_allclose(field_x * drift.U + field_y * drift.V, 0.0)
		plt.close(animation._fig)


if __name__ == "__main__":
	unittest.main()
