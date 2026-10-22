# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Explicit palindromes of A/B/T/C, shared by all extended methods."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np

from formulations.gc import GCDoubledMaps, _EnergyQuadraturePoint
from methods.extended.core.operations import Operation, A, B, T, C
from methods.extended.core.records import CompositionTrace, MapStage, OperationStage

_CUBE_ROOT_TWO: 'float' = float(np.cbrt(2.0))
_GAMMA = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_DELTA = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
_ABBA4_COEFFICIENTS: 'np.ndarray' = np.asarray((_GAMMA, _DELTA, _GAMMA), dtype=float)

# Yoshida's real symmetric order-six solution. Its negative stages are required
# for the odd composition conditions to cancel while preserving self-adjointness.
_ABBA6_COEFFICIENTS: 'np.ndarray' = np.asarray(
	(
		0.78451361047755726382,
		0.23557321335935813368,
		-1.17767998417887100695,
		1.31518632068391121889,
		-1.17767998417887100695,
		0.23557321335935813368,
		0.78451361047755726382,
	),
	dtype=float,
)


@dataclass(frozen=True, slots=True)
class Composition:
	"""A complete chronological palindrome with explicit spatial and time weights.

	A and B each integrate one copy for total duration h; T advances the clock
	by h. C is optional and its total weight scales the chosen coupling rate.
	Symmetry and consistency do not establish an order beyond two.
	"""

	name: str
	operations: tuple[Operation, ...]
	# Grouping retains the established preset observations, not execution logic.
	coefficients: tuple[float, ...] = field(default=(), init=False)
	_group_size: int = field(default=0, init=False, repr=False)

	def __post_init__(self) -> None:
		operations: tuple[Operation, ...] = tuple(self.operations)
		if not isinstance(self.name, str) or not self.name.strip() or not operations or any(not isinstance(op, Operation) for op in operations):
			raise ValueError('A composition requires a name and a nonempty sequence of operations.')
		object.__setattr__(self, 'operations', operations)
		tolerance: float = float(64 * np.finfo(float).eps)
		for left, right in zip(operations, reversed(operations)):
			if left.name != right.name or abs(left.coefficient - right.coefficient) > tolerance:
				raise ValueError('Composition operations and coefficients must be palindromic.')
		for name in ('A', 'B', 'T'):
			if abs(math.fsum(op.coefficient for op in operations if op.name == name) - 1.) > tolerance:
				raise ValueError(f'Composition {name} weights must sum to one.')

	@classmethod
	def from_half(cls, name: str, half: tuple[Operation, ...],
	              center: Operation | None = None) -> Composition:
		"""Reflect a chronological half, optionally around one central operation."""
		left: tuple[Operation, ...] = tuple(half)
		return cls(name, left + (() if center is None else (center,)) + left[::-1])

	@classmethod
	def from_direct_adjoint(cls, name: str, coefficients: tuple[float, ...], *,
	                        coupled: bool = False) -> Composition:
		"""Expand traditional signed stages into an explicit operation palindrome.

		Adjoint groups are A,B,T (or C,A,B,T); direct groups are their reverse.
		No grouping is consulted while executing or differentiating operations.
		"""
		weights: np.ndarray = np.asarray(coefficients, dtype=float)
		tolerance: float = float(64 * np.finfo(float).eps)
		if (weights.ndim != 1 or weights.size == 0 or weights.size % 2
				or not np.all(np.isfinite(weights))
				or not np.allclose(weights, weights[::-1], rtol=0, atol=tolerance)
				or not np.isclose(np.sum(weights), 1., rtol=0, atol=tolerance)):
			raise ValueError('Direct/adjoint weights must be finite, even, palindromic and sum to one.')
		operations: list[Operation] = []
		for index, weight in enumerate(weights):
			group = ((C(float(weight)),) if coupled else ()) + (A(float(weight)), B(float(weight)), T(float(weight)))
			operations.extend(group if index % 2 == 0 else group[::-1])
		result: Composition = cls(name, tuple(operations))
		object.__setattr__(result, 'coefficients', tuple(float(w) for w in weights))
		object.__setattr__(result, '_group_size', 4 if coupled else 3)
		return result

	@property
	def has_coupling(self) -> bool:
		return any(op.name == 'C' for op in self.operations)

	@property
	def shear_count(self) -> int:
		return sum(op.name in ('A', 'B') for op in self.operations)


ABBA2: 'Composition' = Composition.from_direct_adjoint('ABBA2', (0.5, 0.5))
ABBA4: 'Composition' = Composition.from_direct_adjoint('ABBA4', tuple(float(c / 2) for c in _ABBA4_COEFFICIENTS for _ in range(2)))
ABBA6: 'Composition' = Composition.from_direct_adjoint('ABBA6', tuple(float(c / 2) for c in _ABBA6_COEFFICIENTS for _ in range(2)))
_BM4_HALF = (0.0792036964311957, 0.1303114101821663, 0.2228614958676077,
             -0.3667132690474257, 0.3246481886897062, 0.1096884778767498)
BM4: 'Composition' = Composition.from_direct_adjoint('BM4', _BM4_HALF + _BM4_HALF[::-1])
StageMap = Callable[[float, float, np.ndarray], np.ndarray]


def compose(maps: GCDoubledMaps, recipe: Composition, t: float,
            state: np.ndarray, h: float) -> CompositionTrace:
	"""Execute each operation once; only T changes the clock.

	Retained spatial sources serve analytic tangents and passive quadrature.
	Existing grouped stage records are reconstructed without replaying maps.
	"""
	initial: np.ndarray = np.asarray(state, dtype=float)
	if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
		raise ValueError('The doubled composition state must be a finite vector.')
	if not np.isfinite(t) or not np.isfinite(h):
		raise ValueError('Composition time and duration must be finite.')
	if recipe.has_coupling and maps.coupling is None:
		raise ValueError('A composition containing C requires a HarmonicCoupling.')
	current: np.ndarray
	clock: float
	current, clock = initial, float(t)
	operations: list[OperationStage] = []
	for operation in recipe.operations:
		duration: float = float(operation.coefficient * h)
		points: list[_EnergyQuadraturePoint] = []
		if operation.name == 'T':
			after: np.ndarray = current
		else:
			after = maps.apply_spatial(operation.name, duration, clock, current, energy_points=points)
		if after.shape != initial.shape or not np.all(np.isfinite(after)):
			raise ValueError('A composition operation changed shape or became non-finite.')
		record: OperationStage = OperationStage(operation.name, clock, duration, current, after,
		                        points[0][2] if points else None)
		operations.append(record)
		current, clock = after, record.end_time
		if not np.isfinite(clock):
			raise ValueError('The composition clock became non-finite.')
	stages: list[MapStage] = []
	for index, coefficient in enumerate(recipe.coefficients):
		group = operations[index * recipe._group_size:(index + 1) * recipe._group_size]
		quadrature: tuple[tuple[float, float, np.ndarray], ...] = tuple((op.time, op.duration, op.source) for op in group if op.source is not None)
		stages.append(MapStage(group[0].time, quadrature[0][0], coefficient * h,
		                       index % 2 == 1, group[0].state_before, group[-1].state_after, quadrature))
	return CompositionTrace(t, h, initial, current, tuple(stages), tuple(operations), clock)


__all__ = ['Composition']
