"""Contracts for optional notebook presentation helpers."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from matplotlib import pyplot as plt
from matplotlib.animation import Animation, FuncAnimation

from visualization import display_animation


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


if __name__ == "__main__":
	unittest.main()
