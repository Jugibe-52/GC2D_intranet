"""Structural contracts consumed by visualization.trajectory_accuracy."""

from __future__ import annotations

from typing import Protocol
import numpy as np


class AccuracySeriesView(Protocol):
	"""Time-dependent fields consumed by the accuracy comparison plot."""

	@property
	def rms_distance(self) -> np.ndarray:
		"""Particle-RMS error at each saved time."""
		...

	@property
	def maximum_distance(self) -> np.ndarray:
		"""Maximum particle error at each saved time."""
		...


class AccuracySummaryView(Protocol):
	"""Scalar fields consumed by accuracy and cost plots."""

	@property
	def method_name(self) -> str:
		"""Stable method label."""
		...

	@property
	def global_rms_distance(self) -> float:
		"""RMS distance across every particle and saved time."""
		...

	@property
	def final_rms_distance(self) -> float:
		"""Particle-RMS distance at the final time."""
		...

	@property
	def runtime_seconds(self) -> float:
		"""Measured integration runtime."""
		...


class RuntimeSummaryView(Protocol):
	"""Median and interquartile runtime fields consumed by timing plots."""

	@property
	def method_name(self) -> str:
		"""Stable method identifier."""
		...

	@property
	def method_label(self) -> str:
		"""Human-readable method label."""
		...

	@property
	def runtime_seconds(self) -> float:
		"""Median measured integration runtime."""
		...

	@property
	def runtime_first_quartile_seconds(self) -> float:
		"""First runtime quartile."""
		...

	@property
	def runtime_third_quartile_seconds(self) -> float:
		"""Third runtime quartile."""
		...


class StepAccuracySummaryView(Protocol):
	"""Step-dependent fields consumed by the refinement plot."""

	@property
	def integration_step(self) -> float:
		"""Complete integration step size."""
		...

	@property
	def method_name(self) -> str:
		"""Stable method label."""
		...

	@property
	def time_integrated_rms_distance(self) -> float:
		"""Comparable space-time RMS error on the common saved grid."""
		...


__all__ = ['AccuracySeriesView', 'AccuracySummaryView', 'RuntimeSummaryView', 'StepAccuracySummaryView']
