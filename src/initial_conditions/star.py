"""Equally spaced particles on equally spaced straight radial arms."""

import numpy as np

from initial_conditions.gc import GCInitialConfiguration


def radial_star(*, center: tuple[float, float], arm_length: float,
                arms: int, particles_per_arm: int,
                first_angle: float = 0.0) -> GCInitialConfiguration:
    """Build a star in the caller's coordinate units, without normalizing space.

    Particle order is arm-major, then increasing radius. Arm k has angle
    ``first_angle + 2*pi*k/arms``. Each arm includes its outer endpoint and
    excludes the center: radii are ``arm_length * (1, ..., n) / n``. Thus
    no coincident center particles are introduced by different arms.
    """
    for name, value in (("arms", arms), ("particles_per_arm", particles_per_arm)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must contain two finite coordinates.")
    if not np.isfinite(arm_length) or arm_length <= 0 or not np.isfinite(first_angle):
        raise ValueError("arm_length must be positive and first_angle must be finite.")
    angles = first_angle + 2 * np.pi * np.arange(arms) / arms
    radii = arm_length * np.arange(1, particles_per_arm + 1) / particles_per_arm
    x = origin[0] + np.cos(angles[:, None]) * radii
    y = origin[1] + np.sin(angles[:, None]) * radii
    return GCInitialConfiguration.from_components(x=x.ravel(), y=y.ravel())


__all__ = ["radial_star"]
