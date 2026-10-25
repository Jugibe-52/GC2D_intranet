"""Shared values describing one accepted numerical integration step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from contracts.methods.quadrature import QuadratureData

import numpy as np
from contracts.arrays import BackendArray


Detail = TypeVar("Detail")
StepValue = BackendArray | float | int


@dataclass(frozen=True, slots=True)
class StepResult(Generic[Detail]):
	"""One complete physical-state map and its observer-free metrics.

	``state`` contains one physical copy; optional quadrature is diagnostic data. ``details`` belongs
	to that method and is never interpreted or retained by States.
	"""

	state: BackendArray
	statistics: Mapping[str, StepValue]
	details: Detail
	quadrature: QuadratureData | None = None
	converged: Any = True


@dataclass(frozen=True, slots=True)
class StepInfo:
	"""Accepted interval and an independent physical input snapshot."""

	index: int
	time: float
	duration: float
	end_time: float
	state_before: np.ndarray


__all__ = ["StepInfo", "StepResult", "StepValue"]
