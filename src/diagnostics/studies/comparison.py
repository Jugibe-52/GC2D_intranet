"""Reference and physical-energy error reductions for numerical comparisons."""

import numpy as np

from contracts.comparison import AdaptiveReference


def time_integrated_rms_floor(reference: AdaptiveReference) -> float:
	"""Time-average the squared particle-RMS DOP853/Radau discrepancy.

	Distances have shape ``(particles, samples)``. Trapezoidal integration uses
	the saved time grid, retaining the reference's coordinate and time units.
	"""
	particle_rms_squared: np.ndarray = np.mean(reference.audit_distances**2, axis=0)
	return float(np.sqrt(
		np.trapz(particle_rms_squared, reference.times)
		/ float(reference.times[-1] - reference.times[0])
	))


def final_rms_floor(reference: AdaptiveReference) -> float:
	"""Return the final-time particle-RMS DOP853/Radau discrepancy."""
	return float(np.sqrt(np.mean(reference.audit_distances[:, -1] ** 2)))


def rms_error(errors: np.ndarray) -> np.ndarray:
	"""Reduce validated signed ``(particles, samples)`` energy errors over particles.

	The result has shape ``(samples,)`` in the input energy units. Input arrays
	remain untouched; the returned array is independent.
	"""
	return np.asarray(np.sqrt(np.mean(errors**2, axis=0)), dtype=float)


def maximum_absolute_error(errors: np.ndarray) -> np.ndarray:
	"""Return the worst absolute particle-energy error at every saved time."""
	return np.asarray(np.max(np.abs(errors), axis=0), dtype=float)


def running_maximum_absolute_error(errors: np.ndarray) -> np.ndarray:
	"""Accumulate the worst absolute particle-energy error through saved time."""
	return np.maximum.accumulate(maximum_absolute_error(errors))


__all__ = [
	"time_integrated_rms_floor", "final_rms_floor", "rms_error",
	"maximum_absolute_error", "running_maximum_absolute_error",
]
