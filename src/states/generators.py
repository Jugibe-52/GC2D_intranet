"""Equally spaced particles on equally spaced straight radial arms."""

import numpy as np


from formulations.layout import GCStateLayout


def _validated_radial_star_origin(
	center: tuple[float, float], arm_length: float, arms: int,
	particles_per_arm: int, first_angle: float, include_center: bool,
) -> np.ndarray:
	"""Check radial geometry and return its finite two-coordinate origin."""
	for name, value in (("arms", arms), ("particles_per_arm", particles_per_arm)):
		if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
			raise ValueError(f"{name} must be a positive integer.")
	origin = np.asarray(center, dtype=float)
	if origin.shape != (2,) or not np.all(np.isfinite(origin)):
		raise ValueError("center must contain two finite coordinates.")
	if not np.isfinite(arm_length) or arm_length <= 0 or not np.isfinite(first_angle):
		raise ValueError("arm_length must be positive and first_angle must be finite.")
	if not isinstance(include_center, (bool, np.bool_)):
		raise TypeError("include_center must be a boolean.")
	return origin


def radial_star(*, center: tuple[float, float], arm_length: float,
                arms: int, particles_per_arm: int,
                first_angle: float = 0.0,
                inner_radius: float | None = None,
                include_center: bool = False) -> np.ndarray:
    """Build a star in the caller's coordinate units, without normalizing space.

    Particle order is arm-major, then increasing radius. Arm k has angle
    ``first_angle + 2*pi*k/arms``. Each arm includes its outer endpoint and
    excludes the center: radii are ``arm_length * (1, ..., n) / n``. Thus
    no coincident center particles are introduced by different arms. An explicit
    inner_radius instead samples both endpoints of [inner_radius, arm_length].
    include_center prepends exactly one shared central particle, followed by
    the positive-radius arm particles in the same established order.
    """
    origin = _validated_radial_star_origin(
        center, arm_length, arms, particles_per_arm, first_angle, include_center,
    )
    angles = first_angle + 2 * np.pi * np.arange(arms) / arms
    radii = arm_length * np.arange(1, particles_per_arm + 1) / particles_per_arm
    if inner_radius is not None:
        if not np.isfinite(inner_radius) or not 0 < inner_radius <= arm_length:
            raise ValueError("inner_radius must be positive and no greater than arm_length.")
        radii = np.linspace(inner_radius, arm_length, particles_per_arm)
    x = origin[0] + np.cos(angles[:, None]) * radii
    y = origin[1] + np.sin(angles[:, None]) * radii
    x, y = x.ravel(), y.ravel()
    if include_center:
        x, y = np.r_[origin[0], x], np.r_[origin[1], y]
    return np.asarray(GCStateLayout().pack_components(x, y))


def _validated_ranked_star_origin(
	center: tuple[float, float], outer_radius: float, particles: int,
	arms: int, first_angle: float,
) -> np.ndarray:
	"""Check equal arm populations before preparing ranked radial positions."""
	for name, value in (("particles", particles), ("arms", arms)):
		if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
			raise ValueError(f"{name} must be a positive integer.")
	if particles < 2 or particles % arms:
		raise ValueError("Use at least two particles and an equal count per arm.")
	origin = np.asarray(center, dtype=float)
	if origin.shape != (2,) or not np.all(np.isfinite(origin)):
		raise ValueError("center must contain two finite coordinates.")
	if not np.isfinite(outer_radius) or outer_radius <= 0 or not np.isfinite(first_angle):
		raise ValueError("outer_radius must be positive and first_angle finite.")
	return origin


def ranked_radial_star(*, center: tuple[float, float], outer_radius: float,
                       particles: int, arms: int,
                       first_angle: float = 0.0) -> np.ndarray:
    """Place distinct, increasing radii on alternating arms, including one center.

    Particle i (zero based) has radius outer_radius*i/(particles-1) and arm
    i modulo arms. IDs therefore increase strictly with distance from center.
    All distances use the caller's coordinate units.
    """
    origin = _validated_ranked_star_origin(center, outer_radius, particles, arms, first_angle)
    radii = np.linspace(0.0, outer_radius, particles)
    angles = first_angle + 2 * np.pi * (np.arange(particles) % arms) / arms
    return np.asarray(GCStateLayout().pack_components(origin[0] + radii * np.cos(angles), origin[1] + radii * np.sin(angles)))



def _validate_center(center: tuple[float, float]) -> tuple[float, float]:
	"""Normalize a finite two-dimensional centre."""
	try:
		center_x, center_y = center
	except (TypeError, ValueError) as exc:
		raise ValueError("`center` must contain exactly two coordinates.") from exc
	values = np.asarray((center_x, center_y), dtype=float)
	if not np.all(np.isfinite(values)):
		raise ValueError("`center` must contain finite coordinates.")
	return float(values[0]), float(values[1])


def _positive_finite(value: float, name: str) -> float:
	"""Normalize a scalar geometric parameter and enforce positivity."""
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"`{name}` must be positive and finite.") from exc
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def square(
	*,
	center: tuple[float, float],
	side: float,
	points_per_side: int = 1,
) -> np.ndarray:
	"""Sample a square counter-clockwise with equal density on every edge.

	``center`` locates the geometric centre, ``side`` is its full side
	length, and ``points_per_side`` controls boundary resolution.  The final
	state contains ``4 * points_per_side`` vertices.
	"""
	center_x, center_y = _validate_center(center)
	side = _positive_finite(side, "side")
	if (
		isinstance(points_per_side, (bool, np.bool_))
		or not isinstance(points_per_side, (int, np.integer))
		or points_per_side < 1
	):
		raise ValueError("`points_per_side` must be a positive integer.")

	# These four bounds describe the geometry; ``edge`` below is a local
	# coordinate measured from the first corner of each oriented side.
	half_side = side / 2
	left = center_x - half_side
	right = center_x + half_side
	bottom = center_y - half_side
	top = center_y + half_side
	# Omitting each edge endpoint prevents duplicate corners: the following
	# edge supplies that corner, and polygon closure supplies the final one.
	edge = np.linspace(0.0, side, int(points_per_side), endpoint=False)
	x = np.concatenate(
		(
			left + edge,
			np.full_like(edge, right),
			right - edge,
			np.full_like(edge, left),
		)
	)
	y = np.concatenate(
		(
			np.full_like(edge, bottom),
			bottom + edge,
			np.full_like(edge, top),
			top - edge,
		)
	)
	return np.array(np.concatenate((x, y)), dtype=float, copy=True)


def circle(
	*,
	center: tuple[float, float],
	radius: float,
	points: int = 128,
) -> np.ndarray:
	"""Sample a counter-clockwise circle without repeating its first point.

	``center`` and ``radius`` define the initial geometry, while ``points``
	is the total number of boundary vertices rather than a density per arc.
	"""
	center_x, center_y = _validate_center(center)
	radius = _positive_finite(radius, "radius")
	if (
		isinstance(points, (bool, np.bool_))
		or not isinstance(points, (int, np.integer))
		or points < 3
	):
		raise ValueError("`points` must be an integer of at least 3.")

	# The shoelace formula closes the polygon, so sampling 2π again would add
	# a redundant copy of the vertex at angle zero.
	angle = np.linspace(0.0, 2 * np.pi, int(points), endpoint=False)
	x = center_x + radius * np.cos(angle)
	y = center_y + radius * np.sin(angle)
	return np.array(np.concatenate((x, y)), dtype=float, copy=True)


__all__ = ["radial_star", "ranked_radial_star", "square", "circle"]
