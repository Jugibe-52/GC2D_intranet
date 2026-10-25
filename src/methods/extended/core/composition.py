# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Explicit palindromes of A/B/T/C, shared by all extended methods."""
from __future__ import annotations

from typing import Any, TypeAlias
import math

import numpy as np
from contracts.arrays import BackendArray
from types import ModuleType
from contracts.arrays import array_namespace, checked_finite, scan_arrays, select_array_branch

from methods.extended.maps import GCDoubledMaps
from methods.extended.core.operations import Operation
from methods.extended.core.records import CompositionTrace, OperationStage

# Recipes contain only the chronological operations; coefficients multiply h.
Composition: TypeAlias = tuple[Operation, ...]


def validate_composition(recipe: Composition) -> None:
	"""Check one recipe at run preparation, independently of stage observations.

	A/B/T each have total weight one. C is optional; its total weight scales
	the configured coupling. Palindromy alone does not establish higher order.
	"""
	if not isinstance(recipe, tuple) or not recipe or any(not isinstance(op, Operation) for op in recipe):
		raise ValueError('A composition requires a nonempty tuple of operations.')
	tolerance: float = float(64 * np.finfo(float).eps)
	for left, right in zip(recipe, reversed(recipe)):
		if left.name != right.name or abs(left.coefficient - right.coefficient) > tolerance:
			raise ValueError('Composition operations and coefficients must be palindromic.')
	for name in ('A', 'B', 'T'):
		if abs(math.fsum(op.coefficient for op in recipe if op.name == name) - 1.) > tolerance:
			raise ValueError(f'Composition {name} weights must sum to one.')


def compose(maps: GCDoubledMaps, recipe: Composition, t: float,
            state: BackendArray, h: float) -> CompositionTrace:
	"""Execute each operation once; only T changes the clock.

	Retained spatial sources serve analytic tangents and passive quadrature.
	Observation adapters group retained spatial states without replaying maps.
	"""
	xp: ModuleType = array_namespace(state)
	initial: BackendArray = xp.asarray(state, dtype=float)
	initial = checked_finite(initial, 'The doubled composition state must be a finite vector.', ValueError)
	if initial.ndim != 1 or not initial.size:
		raise ValueError('The doubled composition state must be a finite vector.')
	t = checked_finite(t, 'Composition time and duration must be finite.')
	h = checked_finite(h, 'Composition time and duration must be finite.')
	if any(op.name == 'C' for op in recipe) and maps.coupling is None:
		raise ValueError('A composition containing C requires a HarmonicCoupling.')
	# Encode only the existing recipe's operation names; both array routes
	# execute this same ordered recurrence and retain its accepted sources.
	codes: np.ndarray = np.asarray([('A', 'B', 'T', 'C').index(op.name) for op in recipe])
	durations: BackendArray = h * xp.asarray([op.coefficient for op in recipe])
	def apply(carry: Any, operation: Any) -> tuple[Any, Any]:
		current: BackendArray
		clock: Any
		code: Any
		duration: Any
		current, clock = carry
		code, duration = operation
		after: BackendArray = select_array_branch(code, (
			lambda: maps.apply_spatial('A', duration, clock, current),
			lambda: maps.apply_spatial('B', duration, clock, current),
			lambda: current,
			lambda: current if maps.coupling is None else maps.apply_spatial('C', duration, clock, current),
		))
		after = checked_finite(after, 'A composition operation changed shape or became non-finite.')
		if after.shape != initial.shape:
			raise ValueError('A composition operation changed shape or became non-finite.')
		next_time: Any = select_array_branch(code, (
			lambda: clock, lambda: clock, lambda: clock + duration, lambda: clock))
		next_time = checked_finite(next_time, 'The composition clock became non-finite.')
		source: BackendArray = xp.where(code == 0, current[maps.physical_size:], current[:maps.physical_size])
		return (after, next_time), (clock, current, after, source)
	accepted: Any
	history: Any
	accepted, history = scan_arrays(apply, (initial, t), (codes, durations))
	current: BackendArray
	clock: Any
	current, clock = accepted
	operations: list[OperationStage] = []
	for index, operation in enumerate(recipe):
		before: BackendArray = history[1][index]
		after: BackendArray = before if operation.name == 'T' else history[2][index]
		source: BackendArray | None = history[3][index] if operation.name in ('A', 'B') else None
		operations.append(OperationStage(operation.name, history[0][index], durations[index], before, after, source))
	return CompositionTrace(t, h, initial, current, tuple(operations), clock)


__all__ = ['Composition']
