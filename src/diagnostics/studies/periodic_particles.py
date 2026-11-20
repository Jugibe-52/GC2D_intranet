"""Unwrapped periodic closure and duplicate-orbit checks for particle selection."""
import numpy as np


def orbit_metrics(returns: np.ndarray) -> dict[str, float]:
	"""Measure unwrapped closure and separation from all earlier cycle returns."""
	if (returns.ndim != 2 or returns.shape[1] != 2 or len(returns) < 2
			or not np.isfinite(returns).all()):
		raise ValueError('Expected one orbit with shape (q+1, 2).')
	distances: np.ndarray = np.linalg.norm(returns - returns[0], axis=1)
	return dict(closure=float(distances[-1]),
		earlier_return_min=float(distances[1:-1].min()) if len(returns) > 2 else float('inf'),
		cycle_excursion=float(distances.max()))


def distinct_orbit(first: np.ndarray, second: np.ndarray, tolerance: float = 1e-4) -> bool:
	"""Reject cyclic phase duplicates, including spatial translations by period 2."""
	for orbit in (first, second):
		if orbit.ndim != 2 or orbit.shape[1] != 2 or len(orbit) < 2 or not np.isfinite(orbit).all():
			raise ValueError('Orbit comparison requires finite cycle returns with shape (q+1, 2).')
	difference: np.ndarray = first[:-1, None] - second[None, :-1]
	difference = (difference + 1.) % 2. - 1.
	return bool(np.min(np.linalg.norm(difference, axis=-1)) > tolerance)


__all__ = ['orbit_metrics', 'distinct_orbit']
