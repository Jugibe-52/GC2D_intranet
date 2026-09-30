"""Equally spaced particles on equally spaced straight radial arms."""

import numpy as np

from initial_conditions.gc import GCInitialConfiguration


def radial_star(*, center: tuple[float, float], arm_length: float,
                arms: int, particles_per_arm: int,
                first_angle: float = 0.0,
                inner_radius: float | None = None,
                include_center: bool = False) -> GCInitialConfiguration:
    """Build a star in the caller's coordinate units, without normalizing space.

    Particle order is arm-major, then increasing radius. Arm k has angle
    ``first_angle + 2*pi*k/arms``. Each arm includes its outer endpoint and
    excludes the center: radii are ``arm_length * (1, ..., n) / n``. Thus
    no coincident center particles are introduced by different arms. An explicit
    inner_radius instead samples both endpoints of [inner_radius, arm_length].
    include_center prepends exactly one shared central particle, followed by
    the positive-radius arm particles in the same established order.
    """
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
    return GCInitialConfiguration.from_components(x=x, y=y)


def ranked_radial_star(*, center: tuple[float, float], outer_radius: float,
                       particles: int, arms: int,
                       first_angle: float = 0.0) -> GCInitialConfiguration:
    """Place distinct, increasing radii on alternating arms, including one center.

    Particle i (zero based) has radius outer_radius*i/(particles-1) and arm
    i modulo arms. IDs therefore increase strictly with distance from center.
    All distances use the caller's coordinate units.
    """
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
    radii = np.linspace(0.0, outer_radius, particles)
    angles = first_angle + 2 * np.pi * (np.arange(particles) % arms) / arms
    return GCInitialConfiguration.from_components(
        x=origin[0] + radii * np.cos(angles),
        y=origin[1] + radii * np.sin(angles),
    )


__all__ = ["radial_star", "ranked_radial_star"]
