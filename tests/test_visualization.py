"""Contracts for optional notebook presentation helpers."""

from __future__ import annotations

from dynamics.gc import GuidingCenterDynamics

from inspect import signature
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import Animation, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.quiver import Quiver
from matplotlib.text import Text


from potential import Potential
from potential.grid import Grid
from simulation import States
from visualization import (
	animate_abba4_configuration_trajectories,
	animate_fc_particle_solution,
	animate_gc_particle_solution,
	animate_gc_particle_trajectories,
	animate_gc_area_comparison,
	animate_gc_area_solution,
	animate_implicit_method_trajectories,
	animate_potential,
	animate_ten_method_trajectory_points,
	animate_trajectory_points,
	display_animation,
	plot_potential,
)


class NotebookPresentationTests(unittest.TestCase):
	"""Keep animation embedding usable when external encoders fail."""

	def test_potential_views_cover_the_complete_periodic_box(self) -> None:
		"""Use the exact periodic boundaries rather than sample-center limits."""
		potential = Potential.random(A=0.1, M=2, nx=8, ny=8, seed=7)
		expected_limits = (0.0, potential.grid.period)

		figure, axis = plot_potential(potential, show=False)
		self.assertEqual(axis.get_xlim(), expected_limits)
		self.assertEqual(axis.get_ylim(), expected_limits)
		plt.close(figure)

		animation = animate_potential(potential, frames=2)
		self.assertEqual(animation._fig.axes[0].get_xlim(), expected_limits)
		self.assertEqual(animation._fig.axes[0].get_ylim(), expected_limits)
		animation._draw_was_started = True
		plt.close(animation._fig)

	def test_animation_defaults_use_saved_samples_at_5_fps(self) -> None:
		"""Use every saved state and a 200 ms frame interval by default."""
		solution_animations = (
			animate_abba4_configuration_trajectories,
			animate_fc_particle_solution,
			animate_gc_area_comparison,
			animate_gc_area_solution,
			animate_gc_particle_solution,
			animate_gc_particle_trajectories,
			animate_implicit_method_trajectories,
			animate_ten_method_trajectory_points,
			animate_trajectory_points,
		)
		for animation_function in solution_animations:
			with self.subTest(animation=animation_function.__name__):
				parameters = signature(animation_function).parameters
				self.assertIsNone(parameters["frames"].default)
				self.assertEqual(parameters["interval"].default, 200)

		potential_parameters = signature(animate_potential).parameters
		self.assertEqual(potential_parameters["t_max"].default, 1.0)
		self.assertIsNone(potential_parameters["frames"].default)
		self.assertEqual(potential_parameters["interval"].default, 200)
		potential = Potential.random(A=0.1, M=2, nx=8, ny=8, seed=7)
		animation = animate_potential(potential, t_max=10.0)
		self.assertEqual(animation._save_count, 100)
		self.assertEqual(animation.event_source.interval, 200)
		animation._draw_was_started = True
		plt.close(animation._fig)

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

	def test_display_animation_can_emit_interactive_html_only_output(self) -> None:
		figure = plt.figure()
		animation = FuncAnimation(figure, lambda _frame: (), frames=(0,))
		animation._draw_was_started = True
		html = "<div>Interactive JavaScript animation</div>"

		with (
			patch.object(Animation, "to_jshtml", return_value=html) as javascript,
			patch.object(Animation, "to_html5_video") as html5_video,
			patch("IPython.display.display") as display,
		):
			display_animation(animation, interactive=True)

		javascript.assert_called_once_with(default_mode="loop")
		html5_video.assert_not_called()
		display.assert_called_once_with({"text/html": html}, raw=True)
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
		source = GuidingCenterDynamics.coordinates.pack_components(np.asarray([1.0]), np.asarray([1.2]))
		solution = States(t=np.asarray([0.0, 0.1, 0.2]), states=(np.asarray(
				[
					[1.0, 1.1, 1.2],
					[1.2, 1.25, 1.3],
				]
			)).T, initial_state=source, layout=GuidingCenterDynamics.coordinates)

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
		source = GuidingCenterDynamics.coordinates.pack_components(np.asarray([1.0, 1.4]), np.asarray([1.2, 1.6]))
		solution = States(t=np.asarray([0.0, 0.1, 0.2]), states=(np.asarray(
				[
					[1.0, 1.1, 1.2],
					[1.4, 1.5, 1.6],
					[1.2, 1.25, 1.3],
					[1.6, 1.65, 1.7],
				]
			)).T, initial_state=source, layout=GuidingCenterDynamics.coordinates)

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

	def test_reflected_particle_display_keeps_saved_states_and_transforms_velocity(self) -> None:
		grid = Grid.periodic(8, 8, period=1.)
		xx, yy = np.meshgrid(grid.x, grid.y, indexing='ij')
		potential = Potential(grid, mean=np.cos(np.pi*xx)*np.cos(np.pi*yy))
		x = np.array([[.2, 1.2, -1.2], [.3, 1.3, 2.3]])
		y = np.array([[.4, .4, -.4], [.6, 1.6, 2.6]])
		packed = np.concatenate((x, y)).T
		solution = States(t=np.array([0., .1, .2]), states=packed,
			initial_state=packed[0], layout=GuidingCenterDynamics.coordinates)
		before = solution.states.copy()
		animation = animate_gc_particle_trajectories(potential, solution,
			show_electric_field=False, show_perpendicular_drift=True)
		artists = animation._func(1)
		animation._draw_was_started = True
		drift = next(artist for artist in artists if isinstance(artist, Quiver))
		field_x, field_y = potential.electric_field(.1, x[:,1], y[:,1])
		np.testing.assert_allclose(drift.get_offsets(), [[.8,.4],[.7,.4]])
		# The x coordinates are in reflected tiles; only particle 2 reflects y.
		np.testing.assert_allclose(drift.U, -field_y)
		np.testing.assert_allclose(drift.V, np.array([-1.,1.])*field_x)
		np.testing.assert_array_equal(solution.states, before)
		plt.close(animation._fig)

	def test_array_only_saved_display_uses_recorded_reflection_and_legacy_wrap(self) -> None:
		from visualization.dop853_energy_balance import plot_paths, trajectory_player
		from studies.dop853_energy_balance import METHODS
		states = np.array([[.8,1.2,1.4],[.4,.5,.6],[.3,.4,.5],
			[.3,.3,.3],[.4,.4,.4],[.6,.6,.6]])
		arrays = {name+'.states': states.copy() for name in METHODS}
		arrays.update(domain=np.array([0.,0.,1.]), initial_state=states[:,0],
			times=np.arange(3.), spatial_extension_reflected=np.asarray(True))
		figure = plot_paths(arrays)
		np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [.8,.8,.6])
		self.assertIn('"reflected": true', trajectory_player(arrays).data)
		plt.close(figure)
		arrays.pop('spatial_extension_reflected')
		figure = plot_paths(arrays)
		np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [.8,np.nan,.4])
		self.assertIn('"reflected": false', trajectory_player(arrays).data)
		np.testing.assert_array_equal(arrays[METHODS[0]+'.states'], states)
		plt.close(figure)


if __name__ == "__main__":
	unittest.main()
