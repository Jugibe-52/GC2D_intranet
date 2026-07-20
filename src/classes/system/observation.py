"""Optional observations emitted by individual composition stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias

import numpy as np


StateMap: TypeAlias = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class IntegrationStage:
	"""Describe one direct or adjoint map inside a composed integration step.

	``state_before`` and ``state_after`` are independent snapshots of the packed
	internal state. ``map_state`` evaluates this exact stage—with its duration and
	evaluation time already fixed—on another state of the same shape. Diagnostic
	code can therefore differentiate a stage without duplicating integrator logic.
	"""

	system_name: str
	flow_name: Literal["flow", "adjoint_flow"]
	step_index: int
	stage_index: int
	time: float
	duration: float
	state_before: np.ndarray
	state_after: np.ndarray
	map_state: StateMap = field(repr=False, compare=False)


StageObserver: TypeAlias = Callable[[IntegrationStage], None]


__all__ = ["IntegrationStage", "StageObserver", "StateMap"]
