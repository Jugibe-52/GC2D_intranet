"""Animation of completed transported-area comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.animation import FuncAnimation

from visualization.gc_area import animate_gc_area_comparison

if TYPE_CHECKING:
	from studies.area_comparison import AreaComparisonResult


def animate_area_comparison(
	result: AreaComparisonResult,
	*,
	frames: int | None = None,
	interval: int = 200,
	repeat: bool = True,
) -> FuncAnimation:
	"""Build the synchronized contour and projected-diagnostic animation."""
	return animate_gc_area_comparison(
		result.effective_potential,
		result.area,
		result.solutions,
		diagnostic_times=result.diagnostic_times,
		relative_symplecticity_errors=result.relative_symplecticity_errors,
		frames=frames,
		interval=interval,
		repeat=repeat,
	)


__all__ = ["animate_area_comparison"]
