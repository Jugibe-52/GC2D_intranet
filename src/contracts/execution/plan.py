"""Backend-independent instructions for fixed steps and requested samples."""

from dataclasses import dataclass, fields

import numpy as np


SAMPLE_START = 0
SAMPLE_END = 1
SAMPLE_SHADOW = 2


@dataclass(frozen=True)
class FixedStepPlan:
	"""Owned read-only arrays in the problem's runtime time units.

	Step arrays have length S; output arrays have length T. Offsets have length
	S+1 and partition output indices [1, T), excluding the initial state.
	Modes select the start, end or an independent shadow map of the owning step.
	"""

	step_starts: np.ndarray
	step_ends: np.ndarray
	step_sizes: np.ndarray
	output_times: np.ndarray
	sample_offsets: np.ndarray
	sample_modes: np.ndarray
	sample_durations: np.ndarray

	def __post_init__(self) -> None:
		"""Keep plans independent of requests and caller-owned arrays."""
		for item in fields(self):
			value: np.ndarray = np.array(getattr(self, item.name), copy=True)
			value.setflags(write=False)
			object.__setattr__(self, item.name, value)

	@property
	def interior_sample_count(self) -> int:
		return int(np.count_nonzero(self.sample_modes[1:] == SAMPLE_SHADOW))


__all__ = ['FixedStepPlan', 'SAMPLE_START', 'SAMPLE_END', 'SAMPLE_SHADOW']
