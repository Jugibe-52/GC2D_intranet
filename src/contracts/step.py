"""Shared values describing one accepted numerical integration step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np


Detail = TypeVar("Detail")
StepValue = np.ndarray | float | int


@dataclass(frozen=True, slots=True)
class StepResult(Generic[Detail]):
	"""One complete internal-state map and its observer-free metrics.

	``state`` uses the method's packed internal coordinates. ``details`` belongs
	to that method and is never interpreted or retained by the collector.
	"""

	state: np.ndarray
	statistics: Mapping[str, StepValue]
	details: Detail


@dataclass(frozen=True, slots=True)
class StepInfo:
	"""Accepted interval and an independent internal input snapshot."""

	index: int
	time: float
	duration: float
	end_time: float
	state_before: np.ndarray


__all__ = ["StepInfo", "StepResult", "StepValue"]
