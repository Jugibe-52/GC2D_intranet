"""Immutable elementary flows with coefficients measured in full-step units."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

OperationName: TypeAlias = Literal['A', 'B', 'T', 'C']


@dataclass(frozen=True, slots=True)
class Operation:
	"""One exact flow for coefficient*h; only T advances the runtime clock."""

	name: OperationName
	coefficient: float

	def __post_init__(self) -> None:
		if self.name not in ('A', 'B', 'T', 'C'):
			raise ValueError('An operation must be A, B, T or C.')
		if isinstance(self.coefficient, (bool, np.bool_)) or not np.isfinite(self.coefficient):
			raise ValueError('An operation coefficient must be finite and non-Boolean.')
		object.__setattr__(self, 'coefficient', float(self.coefficient))


def A(coefficient: float) -> Operation:
	"""Update u from v at fixed time for coefficient*h."""
	return Operation('A', coefficient)


def B(coefficient: float) -> Operation:
	"""Update v from u at fixed time for coefficient*h."""
	return Operation('B', coefficient)


def T(coefficient: float) -> Operation:
	"""Advance the clock by coefficient*h without changing spatial copies."""
	return Operation('T', coefficient)


def C(coefficient: float) -> Operation:
	"""Apply the independently configured harmonic flow for coefficient*h."""
	return Operation('C', coefficient)


__all__ = ['Operation', 'A', 'B', 'T', 'C']
