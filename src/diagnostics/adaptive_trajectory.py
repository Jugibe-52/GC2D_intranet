"""Opt-in retention and evaluation of accepted adaptive dense interpolants."""

from collections.abc import Callable

import numpy as np

from simulation.observation import AdaptiveIntegrationStep


class AdaptiveTrajectoryObserver:
    """Retain a continuous trajectory explicitly, outside the global integrator.

    Each callback stores only its interpolant and interval, not the full event.
    Queries use the left interval at a shared endpoint, as SciPy OdeSolution does.
    The observer is intended for one forward integration and rejects extrapolation.
    """

    def __init__(self) -> None:
        self._bounds: list[float] = []
        self._interpolants: list[Callable[[float | np.ndarray], np.ndarray]] = []
        self._dimension = 0

    def __call__(self, step: AdaptiveIntegrationStep) -> None:
        """Accept consecutive intervals from one run."""
        if step.step_index != len(self._interpolants):
            raise ValueError('Adaptive trajectory observations must be consecutive.')
        if not self._bounds:
            self._bounds.append(step.start_time)
            self._dimension = step.state_before.size
        if step.start_time != self._bounds[-1] or step.time <= step.start_time:
            raise ValueError('Adaptive trajectory intervals must be contiguous and increasing.')
        self._bounds.append(step.time)
        self._interpolants.append(step.dense_state)

    def evaluate(self, time: float | np.ndarray) -> np.ndarray:
        """Evaluate scalar or one-dimensional times within the retained span."""
        times = np.asarray(time, dtype=float)
        if times.ndim > 1 or not np.all(np.isfinite(times)):
            raise ValueError('Dense query times must be finite scalars or vectors.')
        if not self._interpolants:
            raise ValueError('No adaptive trajectory has been observed.')
        if np.any(times < self._bounds[0]) or np.any(times > self._bounds[-1]):
            raise ValueError('Dense query times must lie within the retained span.')
        flat = times.reshape(-1)
        indices = np.clip(np.searchsorted(self._bounds, flat, side='left') - 1,
                          0, len(self._interpolants) - 1)
        states = np.empty((self._dimension, flat.size))
        for index in np.unique(indices):
            mask = indices == index
            states[:, mask] = self._interpolants[int(index)](flat[mask])
        return states[:, 0] if times.ndim == 0 else states


__all__ = ['AdaptiveTrajectoryObserver']
