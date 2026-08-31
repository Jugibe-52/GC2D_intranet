"""Faceted trajectory animation for the sixteen ABBA4 configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from potential import Potential
from simulation import Solution

from .particles import _field_normalization, _frame_indices


_ROW_COORDINATES = (
	(
		"ABBA4Implicit",
		"shared_time",
		"ABBA4 · 3 projections\nshared time · base R6",
	),
	(
		"ABBA4Implicit",
		"fully_extended",
		"ABBA4 · 3 projections\nfully extended · base R8",
	),
	(
		"ABBA4ImplicitSingleProjection",
		"shared_time",
		"SP-ABBA4 · 1 projection\nshared time · base R6",
	),
	(
		"ABBA4ImplicitSingleProjection",
		"fully_extended",
		"SP-ABBA4 · 1 projection\nfully extended · base R8",
	),
)
_COLUMN_COORDINATES = (
	("reduced_multiplier", "newton", "Reduced\nNewton"),
	("reduced_multiplier", "broyden", "Reduced\nBroyden"),
	("simultaneous_state_multiplier", "newton", "Simultaneous\nNewton"),
	("simultaneous_state_multiplier", "broyden", "Simultaneous\nBroyden"),
)
_VARIANT_COUNT = 16
_INITIAL_CONDITION_COLORS = tuple(mcolors.TABLEAU_COLORS.values())
_MAX_PARTICLE_COUNT = len(_INITIAL_CONDITION_COLORS)


def _effective_potential(result: object) -> Potential:
	"""Return the exact potential used by every compared numerical trajectory."""
	candidate = getattr(result, "effective_potential", None)
	if isinstance(candidate, Potential):
		return candidate
	dynamics = getattr(result, "dynamics", None)
	candidate = getattr(dynamics, "effective_potential", None)
	if isinstance(candidate, Potential):
		return candidate
	candidate = getattr(result, "potential", None)
	if isinstance(candidate, Potential):
		return candidate
	raise TypeError(
		"`result` must expose a Potential through `effective_potential`, "
		"`dynamics.effective_potential`, or `potential`."
	)


def _variant_coordinate(variant: object) -> tuple[str, str, str, str]:
	"""Read and validate the four binary coordinates of one result variant."""
	values: list[str] = []
	for name in (
		"method_name",
		"state_extension",
		"projection_formulation",
		"nonlinear_solver",
	):
		value = getattr(variant, name, None)
		if not isinstance(value, str) or not value:
			raise ValueError(f"Every variant must provide a non-empty `{name}`.")
		values.append(value)
	return values[0], values[1], values[2], values[3]


def _ordered_variants(result: object) -> tuple[object, ...]:
	"""Place a complete, unique configuration cube in its visual grid order."""
	variants_value = getattr(result, "variants", None)
	if isinstance(variants_value, (str, bytes)) or not isinstance(
		variants_value, Sequence
	):
		raise TypeError("`result.variants` must be a sequence of configuration records.")
	variants = tuple(variants_value)
	if len(variants) != _VARIANT_COUNT:
		raise ValueError("The animation requires exactly 16 ABBA4 variants.")

	by_coordinate: dict[tuple[str, str, str, str], object] = {}
	for variant in variants:
		coordinate = _variant_coordinate(variant)
		if coordinate in by_coordinate:
			raise ValueError(f"Duplicate ABBA4 configuration coordinate {coordinate!r}.")
		by_coordinate[coordinate] = variant

	expected = tuple(
		(method, extension, formulation, solver)
		for method, extension, _ in _ROW_COORDINATES
		for formulation, solver, _ in _COLUMN_COORDINATES
	)
	missing = tuple(coordinate for coordinate in expected if coordinate not in by_coordinate)
	unexpected = tuple(coordinate for coordinate in by_coordinate if coordinate not in expected)
	if missing or unexpected:
		raise ValueError(
			"The variants do not form the required 4 x 4 ABBA4 configuration grid; "
			f"missing={missing!r}, unexpected={unexpected!r}."
		)
	return tuple(by_coordinate[coordinate] for coordinate in expected)


def _aligned_positions(
	result: object,
	variants: Sequence[object],
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], int]:
	"""Collect sixteen arrays sharing one inferred particle count and order."""
	solutions_value = getattr(result, "solutions", None)
	if not isinstance(solutions_value, Mapping):
		raise TypeError("`result.solutions` must map variant keys to trajectories.")
	variant_keys: list[str] = []
	particle_count: int | None = None
	reference_times: np.ndarray | None = None
	reference_initial_positions: np.ndarray | None = None
	positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

	for variant in variants:
		key = getattr(variant, "key", None)
		if not isinstance(key, str) or not key:
			raise ValueError("Every variant must provide a non-empty `key`.")
		if key in variant_keys:
			raise ValueError(f"Duplicate ABBA4 variant key {key!r}.")
		variant_keys.append(key)
		if key not in solutions_value:
			raise ValueError(f"Missing trajectories for ABBA4 variant {key!r}.")
		trajectory_value = solutions_value[key]
		if isinstance(trajectory_value, (str, bytes)) or not isinstance(
			trajectory_value, Sequence
		):
			raise TypeError("Every variant value must be a sequence of Solutions.")
		trajectories = tuple(trajectory_value)
		if particle_count is None:
			particle_count = len(trajectories)
			if not 1 <= particle_count <= _MAX_PARTICLE_COUNT:
				raise ValueError(
					"The animation requires between 1 and "
					f"{_MAX_PARTICLE_COUNT} trajectories per variant."
				)
		elif len(trajectories) != particle_count:
			raise ValueError(
				"Every ABBA4 variant must contain the same number of trajectories."
			)

		x_rows: list[np.ndarray] = []
		y_rows: list[np.ndarray] = []
		for solution in trajectories:
			if not isinstance(solution, Solution):
				raise TypeError("Every trajectory must be a Solution instance.")
			times = np.asarray(solution.t, dtype=float)
			x, y = solution.positions()
			if x.shape != (1, times.size) or y.shape != x.shape:
				raise ValueError(
					"Each configuration entry must be one sampled planar trajectory."
				)
			if reference_times is None:
				reference_times = times
			elif not np.array_equal(times, reference_times):
				raise ValueError(
					"All configuration trajectories must share one saved-time grid."
				)
			x_rows.append(np.asarray(x[0], dtype=float))
			y_rows.append(np.asarray(y[0], dtype=float))

		x_values = np.asarray(x_rows, dtype=float)
		y_values = np.asarray(y_rows, dtype=float)
		if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
			raise ValueError("Trajectory positions must be finite.")
		initial_positions = np.column_stack((x_values[:, 0], y_values[:, 0]))
		if reference_initial_positions is None:
			reference_initial_positions = initial_positions
		elif not np.array_equal(initial_positions, reference_initial_positions):
			raise ValueError(
				"Every variant must preserve the same ordered initial conditions."
			)
		positions[key] = x_values, y_values

	if set(solutions_value) != set(variant_keys):
		raise ValueError("`result.solutions` must contain exactly the 16 variant keys.")
	assert reference_times is not None
	assert particle_count is not None
	return reference_times, positions, particle_count


def _closed_spatial_limits(potential: Potential) -> tuple[float, float, float, float]:
	"""Return sampled closed bounds without applying periodic endpoint wrapping."""
	x_values = np.asarray(getattr(potential, "x", potential.grid.x), dtype=float)
	y_values = np.asarray(getattr(potential, "y", potential.grid.y), dtype=float)
	if (
		x_values.ndim != 1
		or y_values.ndim != 1
		or x_values.size != potential.grid.nx
		or y_values.size != potential.grid.ny
		or not np.all(np.isfinite(x_values))
		or not np.all(np.isfinite(y_values))
		or np.any(np.diff(x_values) <= 0.0)
		or np.any(np.diff(y_values) <= 0.0)
	):
		raise ValueError("The effective potential must expose finite increasing grid axes.")
	return (
		float(x_values[0]),
		float(x_values[-1]),
		float(y_values[0]),
		float(y_values[-1]),
	)


def _fundamental_frequency(potential: Potential) -> float:
	"""Return the first physical HDF5 frequency, or the normalized unit frequency."""
	frequencies = getattr(potential, "frequencies", None)
	if frequencies is None:
		return 1.0
	values = np.asarray(frequencies, dtype=float)
	if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
		return 1.0
	return float(values[0])


def animate_abba4_configuration_trajectories(
	result: object,
	*,
	frames: int | None = 61,
	interval: int = 100,
	repeat: bool = True,
	cmap: str = "Greys",
	**imshow_kwargs: Any,
) -> FuncAnimation:
	"""Animate 16 ABBA4 configurations and up to 10 shared initial conditions.

	Rows encode the projection placement and state extension, while columns encode
	the projection formulation and nonlinear solver. Each panel therefore needs
	only the stable initial-condition colors; no sixteen-entry method legend
	is required. The effective potential is evaluated once on every selected frame
	and the same field array and normalization are reused by all sixteen panels.
	"""
	if (
		isinstance(interval, (bool, np.bool_))
		or not isinstance(interval, (int, np.integer))
		or int(interval) <= 0
	):
		raise ValueError("`interval` must be a positive integer.")
	if not isinstance(repeat, (bool, np.bool_)):
		raise TypeError("`repeat` must be a boolean.")
	variants = _ordered_variants(result)
	times, positions, particle_count = _aligned_positions(result, variants)
	particle_colors = _INITIAL_CONDITION_COLORS[:particle_count]
	indices = _frame_indices(times.size, frames)
	frame_times = times[indices]
	potential = _effective_potential(result)
	fields = np.asarray(potential.evaluate(frame_times), dtype=float)
	if fields.shape != (*potential.grid.shape, indices.size):
		raise ValueError("Potential evaluation returned an unexpected animation shape.")
	xmin, xmax, ymin, ymax = _closed_spatial_limits(potential)

	image_options: dict[str, Any] = {
		"origin": "lower",
		"extent": (xmin, xmax, ymin, ymax),
		"aspect": "equal",
		"cmap": cmap,
		"norm": _field_normalization(fields),
		"alpha": 0.62,
	}
	image_options.update(imshow_kwargs)
	figure, axes = plt.subplots(
		4,
		4,
		figsize=(16, 13),
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	images: list[Any] = []
	paths: dict[str, LineCollection] = {}
	markers: dict[str, Any] = {}
	for index, (axis, variant) in enumerate(
		zip(axes.flat, variants, strict=True)
	):
		key = str(getattr(variant, "key"))
		x_values, y_values = positions[key]
		image = axis.imshow(fields[:, :, 0].T, **image_options)
		path = LineCollection(
			[],
			colors=particle_colors,
			linewidths=1.0,
			alpha=0.82,
			zorder=3,
		)
		axis.add_collection(path)
		marker = axis.scatter(
			x_values[:, 0],
			y_values[:, 0],
			s=22,
			color=particle_colors,
			edgecolor="white",
			linewidth=0.4,
			zorder=5,
		)
		axis.set(
			xlim=(xmin, xmax),
			ylim=(ymin, ymax),
			aspect="equal",
		)
		axis.tick_params(labelsize=7)
		images.append(image)
		paths[key] = path
		markers[key] = marker
		if index < 4:
			axis.set_title(_COLUMN_COORDINATES[index][2], fontsize=10)

	for row, (_, _, label) in enumerate(_ROW_COORDINATES):
		axes[row, 0].set_ylabel(
			label,
			rotation=0,
			ha="right",
			va="center",
			labelpad=42,
			fontsize=9,
		)
	figure.supxlabel("x")
	figure.supylabel("y")
	figure.colorbar(
		images[0],
		ax=axes,
		label="Effective HDF5 potential",
		shrink=0.84,
		pad=0.015,
	)
	legend_handles = [
		Line2D(
			[0],
			[0],
			color=color,
			marker="o",
			markersize=5,
			linestyle="-",
			linewidth=1.0,
			label=f"IC {particle + 1}",
		)
		for particle, color in enumerate(particle_colors)
	]
	figure.legend(
		handles=legend_handles,
		loc="outside lower center",
		ncols=particle_count,
		fontsize=8,
		framealpha=0.9,
	)
	frequency = _fundamental_frequency(potential)
	suptitle = figure.suptitle("")

	def update(frame: int) -> tuple[Any, ...]:
		"""Update one shared field and every accumulated trajectory."""
		sample_index = int(indices[frame])
		artists: list[Any] = []
		for image in images:
			image.set_data(fields[:, :, frame].T)
			artists.append(image)
		for variant in variants:
			key = str(getattr(variant, "key"))
			x_values, y_values = positions[key]
			paths[key].set_segments(
				[
					np.column_stack(
						(
							x_values[particle, : sample_index + 1],
							y_values[particle, : sample_index + 1],
						)
					)
					for particle in range(particle_count)
				]
			)
			markers[key].set_offsets(
				np.column_stack(
					(x_values[:, sample_index], y_values[:, sample_index])
				)
			)
			artists.extend((paths[key], markers[key]))
		time = float(times[sample_index])
		phase = float(np.mod(frequency * time, 2.0 * np.pi))
		suptitle.set_text(
			f"16 ABBA4 configurations × {particle_count} shared initial "
			f"conditions ({_VARIANT_COUNT * particle_count} trajectories) — "
			f"t = {time:.6g}, phase = {phase:.3f} rad"
		)
		artists.append(suptitle)
		return tuple(artists)

	update(0)
	return FuncAnimation(
		figure,
		update,
		frames=indices.size,
		interval=int(interval),
		blit=False,
		repeat=bool(repeat),
	)


__all__ = ["animate_abba4_configuration_trajectories"]
