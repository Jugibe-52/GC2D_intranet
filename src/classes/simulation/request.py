"""Validated time span, step bound, and requested output times."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SimulationRequest:
	"""Method-independent temporal configuration for one simulation."""

	t_span: tuple[float, float]
	max_step: float
	output_times: np.ndarray

	def __post_init__(self) -> None:
		"""Normalize values and enforce a complete increasing output schedule."""
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		if isinstance(self.max_step, (bool, np.bool_)):
			raise ValueError("`max_step` must be positive and finite.")
		max_step = float(self.max_step)
		if not np.isfinite(max_step) or max_step <= 0:
			raise ValueError("`max_step` must be positive and finite.")

		times = np.asarray(self.output_times, dtype=float)
		if (
			times.ndim != 1
			or times.size < 2
			or not np.all(np.isfinite(times))
			or np.any(np.diff(times) <= 0)
		):
			raise ValueError(
				"`output_times` must contain at least two finite, increasing times."
			)
		t0, tf = float(span[0]), float(span[1])
		tolerance = 16 * np.finfo(float).eps * max(1.0, abs(t0), abs(tf))
		if abs(float(times[0]) - t0) > tolerance or abs(float(times[-1]) - tf) > tolerance:
			raise ValueError("`output_times` must include both ends of `t_span`.")
		if float(times[0]) < t0 - tolerance or float(times[-1]) > tf + tolerance:
			raise ValueError("`output_times` must lie inside `t_span`.")

		normalized_times = times.copy()
		normalized_times[0] = t0
		normalized_times[-1] = tf
		# Endpoint normalization must not turn a tolerance-level outlier into a
		# duplicate or decreasing interval.
		if np.any(np.diff(normalized_times) <= 0):
			raise ValueError(
				"`output_times` must remain strictly increasing inside `t_span`."
			)
		normalized_times.setflags(write=False)
		object.__setattr__(self, "t_span", (t0, tf))
		object.__setattr__(self, "max_step", max_step)
		object.__setattr__(self, "output_times", normalized_times)

	@classmethod
	def uniform(
		cls,
		*,
		t_span: tuple[float, float] = (0.0, 2 * np.pi),
		max_step: float,
		sample_count: int = 361,
	) -> SimulationRequest:
		"""Create a request with uniformly spaced output samples."""
		if (
			isinstance(sample_count, (bool, np.bool_))
			or not isinstance(sample_count, (int, np.integer))
			or sample_count < 2
		):
			raise ValueError("`sample_count` must be an integer of at least 2.")
		span = np.asarray(t_span, dtype=float)
		if span.shape != (2,):
			raise ValueError("`t_span` must contain exactly two values.")
		return cls(
			t_span=(float(span[0]), float(span[1])),
			max_step=max_step,
			output_times=np.linspace(
				float(span[0]),
				float(span[1]),
				int(sample_count),
				dtype=float,
			),
		)


__all__ = ["SimulationRequest"]
