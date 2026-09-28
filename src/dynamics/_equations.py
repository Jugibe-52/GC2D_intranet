"""Shared GC and FC algebra for NumPy and JAX particle arrays."""

from typing import Any


def gc_velocity(ex: Any, ey: Any) -> tuple[Any, Any]:
    """Normalized guiding-centre drift for E = -grad(Phi)."""
    return ey, -ex


def fc_velocity(acceleration_x: Any, acceleration_y: Any, vx: Any, vy: Any, *,
                velocity_scale: float,
                larmor_frequency: float) -> tuple[Any, Any, Any, Any]:
    """Position and velocity rates in normalized component-major order."""
    return (vx * velocity_scale, vy * velocity_scale,
            acceleration_x + vy * larmor_frequency,
            acceleration_y - vx * larmor_frequency)


def fc_hamiltonian(phi: Any, vx: Any, vy: Any, *,
                   velocity_scale: float, electric_scale: float) -> Any:
    """Physical kinetic plus electrostatic energy per particle."""
    return (velocity_scale / 2) * (vx**2 + vy**2) + electric_scale * phi


__all__: list[str] = []
