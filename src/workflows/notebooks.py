"""Presentation helpers that keep Jupyter-specific code out of experiments."""

from __future__ import annotations

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

	mpl.rcParams["animation.embed_limit"] = limit
	html = (
		animation.to_html5_video()
		if mpl_animation.writers.is_available("ffmpeg")
		else animation.to_jshtml(default_mode="once")
	)
	display(HTML(html))


__all__ = ["display_animation"]
