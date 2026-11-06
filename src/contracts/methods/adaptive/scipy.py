"""SciPy step details shared by numerical advancement and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeAlias
import numpy as np
from contracts.arrays import BackendArray


@dataclass(frozen=True, slots=True)
class AdaptiveStepDetails:
	"""Frozen dense output and its work, shared with the output scheduler."""

	dense_state: Callable[[float | np.ndarray], np.ndarray]
	interpolation_work: tuple[int, int, int] = (0, 0, 0)
	has_interpolant: bool = False


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
	state_after: BackendArray
	dense_state: Callable[[float | np.ndarray], np.ndarray] = field(repr=False, compare=False)
	function_evaluations: int
	jacobian_evaluations: int
	lu_decompositions: int


AdaptiveStepObserver: TypeAlias = Callable[[AdaptiveIntegrationStep], None]


__all__ = ['AdaptiveStepDetails', 'AdaptiveIntegrationStep', 'AdaptiveStepObserver']
