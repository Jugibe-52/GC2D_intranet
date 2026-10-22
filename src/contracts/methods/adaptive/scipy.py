"""methods.adaptive.scipy observation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeAlias
import numpy as np


@dataclass(frozen=True, slots=True)
class AdaptiveIntegrationStep:
	"""One accepted adaptive interval, with its dense output and actual work.

	The interpolant evaluates this accepted trajectory only. A variable-step
	solver retains history and may reject trials, so this event intentionally
	exposes no fixed-duration state map for geometric differentiation.
	"""

	dynamics_name: str
	method_name: str
	step_index: int
	start_time: float
	time: float
	duration: float
	state_before: np.ndarray
	state_after: np.ndarray
	dense_state: Callable[[float | np.ndarray], np.ndarray] = field(repr=False, compare=False)
	function_evaluations: int
	jacobian_evaluations: int
	lu_decompositions: int


AdaptiveStepObserver: TypeAlias = Callable[[AdaptiveIntegrationStep], None]


__all__ = ['AdaptiveIntegrationStep', 'AdaptiveStepObserver']
