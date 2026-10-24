"""Structural contracts consumed by execution._host."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, TYPE_CHECKING, TypeVar
from contracts.methods.step import StepInfo, StepResult
if TYPE_CHECKING:
	from methods.base import IntegrationMethod
	from request import SimulationRequest
	from execution.state import IntegrationState

Detail = TypeVar("Detail")


class StepController(Protocol[Detail]):
	"""Deliver accepted intervals and evaluate output samples within them.

	An adaptive implementation must handle its retries internally and expose
	accepted intervals only. Its sampling operation can use dense output. The
	SciPy-backed methods provide adaptive control; other methods use fixed steps.
	"""

	def steps(
		self, method: IntegrationMethod[Detail], request: SimulationRequest,
	) -> Iterator[tuple[StepInfo, StepResult[Detail]]]: ...

	def sample(
		self, method: IntegrationMethod[Detail], info: StepInfo,
		result: StepResult[Detail], time: float,
	) -> IntegrationState:
		"""Return a state at any requested time in the interval, including its ends."""
		...


__all__ = ['StepController']
