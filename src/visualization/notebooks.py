"""Presentation helpers that keep Jupyter-specific code out of experiments."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
import subprocess
from typing import Any, TypeAlias

import matplotlib as mpl
from matplotlib import animation as mpl_animation
from matplotlib.animation import Animation


TableColumn: TypeAlias = tuple[str, str, str | None]


def records_table_html(
	records: Sequence[object],
	*,
	columns: Sequence[TableColumn],
) -> str:
	"""Render attribute-based records as one compact HTML table.

	Each column is ``(attribute_name, heading, format_spec)``. A ``None`` format
	uses the value's normal string representation.
	"""
	rows = tuple(records)
	column_values = tuple(columns)
	if not rows:
		raise ValueError("At least one table record is required.")
	if not column_values:
		raise ValueError("At least one table column is required.")
	for attribute, heading, format_spec in column_values:
		if not isinstance(attribute, str) or not attribute:
			raise ValueError("Table attribute names must be non-empty strings.")
		if not isinstance(heading, str) or not heading:
			raise ValueError("Table headings must be non-empty strings.")
		if format_spec is not None and not isinstance(format_spec, str):
			raise TypeError("Table format specifications must be strings or None.")

	header = "".join(
		f"<th style='text-align:left'>{escape(heading)}</th>"
		for _, heading, _ in column_values
	)
	body_rows: list[str] = []
	for record in rows:
		cells: list[str] = []
		for attribute, _, format_spec in column_values:
			if not hasattr(record, attribute):
				raise AttributeError(
					f"{type(record).__name__} has no table attribute {attribute!r}."
				)
			value: Any = getattr(record, attribute)
			text = str(value) if format_spec is None else format(value, format_spec)
			cells.append(f"<td>{escape(text)}</td>")
		body_rows.append(f"<tr>{''.join(cells)}</tr>")
	return (
		"<table style='border-collapse:collapse'>"
		f"<thead><tr>{header}</tr></thead>"
		f"<tbody>{''.join(body_rows)}</tbody></table>"
	)


def display_records_table(
	records: Sequence[object],
	*,
	columns: Sequence[TableColumn],
) -> None:
	"""Display attribute-based records as an HTML table in a notebook."""
	from IPython.display import HTML, display

	display(HTML(records_table_html(records, columns=columns)))


def display_animation(
	animation: Animation,
	*,
	embed_limit_mb: float = 100.0,
	interactive: bool = False,
) -> None:
	"""Display an animation as video or an interactive JavaScript player.

	Interactive output exposes Matplotlib's frame slider and playback controls.
	It is emitted only as HTML so document exporters such as LaTeX/PDF omit it.
	"""
	if not isinstance(animation, Animation):
		raise TypeError("`animation` must be a Matplotlib Animation instance.")
	limit = float(embed_limit_mb)
	if limit <= 0:
		raise ValueError("`embed_limit_mb` must be positive.")
	if not isinstance(interactive, bool):
		raise TypeError("`interactive` must be a boolean.")

	# IPython is a notebook dependency rather than a simulation-core dependency,
	# so import it only when interactive presentation is explicitly requested.
	from IPython.display import HTML, display
	from matplotlib import pyplot as plt

	mpl.rcParams["animation.embed_limit"] = limit
	if interactive:
		html = animation.to_jshtml(default_mode="loop")
		# Supplying only the HTML MIME representation prevents text or image
		# fallbacks from appearing in LaTeX/PDF exports.
		display({"text/html": html}, raw=True)
	else:
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


__all__ = [
	"TableColumn",
	"display_animation",
	"display_records_table",
	"records_table_html",
]
