"""Structural contracts consumed by execution._host."""

from __future__ import annotations

from collections.abc import Iterator
import numpy as np
from typing import Protocol, TYPE_CHECKING, TypeVar
from contracts.methods.step import StepInfo, StepResult
if TYPE_CHECKING:
	from methods.base import IntegrationMethod
	from request import SimulationRequest
	from execution.state import IntegrationState

Detail = TypeVar("Detail")


class StepController(Protocol[Detail]):
	"""Deliver accepted intervals and evaluate output samples within them.

	Controllers own trial work and expose accepted intervals only. Built-in
	methods use fixed steps; extensions may supply their own sampling policy.
	"""

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest, initial_state: np.ndarray,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]: ...

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> IntegrationState:
		"""Return a state at any requested time in the interval, including its ends."""
		...


__all__ = ['StepController']
