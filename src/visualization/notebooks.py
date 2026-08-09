"""Presentation helpers that keep Jupyter-specific code out of experiments."""

from __future__ import annotations

import subprocess

import matplotlib as mpl
from matplotlib import animation as mpl_animation
from matplotlib.animation import Animation


def display_animation(animation: Animation, *, embed_limit_mb: float = 100.0) -> None:
	"""Display an animation as HTML5 video with a JavaScript fallback."""
	if not isinstance(animation, Animation):
		raise TypeError("`animation` must be a Matplotlib Animation instance.")
	limit = float(embed_limit_mb)
	if limit <= 0:
		raise ValueError("`embed_limit_mb` must be positive.")

	# IPython is a notebook dependency rather than a simulation-core dependency,
	# so import it only when interactive presentation is explicitly requested.
	from IPython.display import HTML, display
	from matplotlib import pyplot as plt

	mpl.rcParams["animation.embed_limit"] = limit
	if mpl_animation.writers.is_available("ffmpeg"):
		try:
			html = animation.to_html5_video()
		except (OSError, subprocess.CalledProcessError):
			# An installed encoder can still reject a frame size or codec. The
			# browser-native representation keeps notebook execution portable.
			html = animation.to_jshtml(default_mode="once")
	else:
		html = animation.to_jshtml(default_mode="once")
	display(HTML(html))
	# Inline backends otherwise emit the animation's first frame as an unrelated
	# static figure after the HTML animation has already been displayed.
	figure = getattr(animation, "_fig", None)
	if figure is not None:
		plt.close(figure)


__all__ = ["display_animation"]
