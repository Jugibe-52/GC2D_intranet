"""Passive normalized momentum quadrature from accepted spatial shear inputs."""
from __future__ import annotations
from collections.abc import Callable, Iterable
import numpy as np
from methods.extended.maps import _EnergyQuadraturePoint


def momentum_increment(
    particle_count: int, momentum_rate: Callable[[float, np.ndarray], np.ndarray],
    points: 'Iterable[_EnergyQuadraturePoint]',
) -> np.ndarray:
    """Accumulate physical kappa once from accepted signed shear sources.

    The doubled Hamiltonian uses summed momentum k=2*kappa. Its passive
    quadrature never changes spatial states or nonlinear stopping criteria.
    """
    increment: np.ndarray = np.zeros(particle_count)
    with np.errstate(over="ignore", invalid="ignore"):
        for time, duration, state in points:
            increment += duration * momentum_rate(time, state)
    increment /= 2
    if not np.all(np.isfinite(increment)):
        raise ValueError('The energy-tracking momentum increment became non-finite.')
    return increment
