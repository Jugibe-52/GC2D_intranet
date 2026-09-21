"""Full (x,y,t,k) duplication, analytic shears and composed base maps."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from dynamics import GuidingCenterDynamics

_ExtendedMap: TypeAlias = Callable[[np.ndarray], np.ndarray]
_ExtendedJacobian: TypeAlias = Callable[[np.ndarray], np.ndarray]
_IDENTITY_4 = np.eye(4)
_IDENTITY_8 = np.eye(8)
_DIAGONAL_EMBEDDING = np.vstack((_IDENTITY_4, _IDENTITY_4))
_ANTIDIAGONAL_EMBEDDING = np.vstack((_IDENTITY_4, -_IDENTITY_4))
_COPY_DIFFERENCE = np.hstack((_IDENTITY_4, -_IDENTITY_4))
_COPY_AVERAGE = 0.5 * np.hstack((_IDENTITY_4, _IDENTITY_4))

@dataclass(frozen=True, slots=True)
class _AnalyticExtendedMap:
	"""An ``R^8`` map together with its exact stage-product Jacobian."""

	map_state: _ExtendedMap
	jacobian_state: _ExtendedJacobian


def _checked_extended_state(state: np.ndarray, *, duplicated: bool) -> np.ndarray:
	"""Validate one physical ``R^4`` or duplicated ``R^8`` extended state."""
	value = np.asarray(state, dtype=float)
	expected = (8,) if duplicated else (4,)
	if value.shape != expected or not np.all(np.isfinite(value)):
		space = "duplicated R^8" if duplicated else "physical R^4"
		raise ValueError(f"The {space} state must be finite with shape {expected}.")
	return value.copy()


def _synchronized_extended_time(
	state: np.ndarray,
	expected_time: float,
	*,
	context: str,
) -> np.ndarray:
	"""Validate and pin the exactly solvable time coordinate to its grid value.

	The extended Hamiltonian has ``dt/ds = 1``.  Its accepted time is therefore
	known analytically, while repeated floating-point stage additions can otherwise
	accumulate enough roundoff to trip a long-integration consistency guard.
	"""
	value = _checked_extended_state(state, duplicated=False)
	expected = float(expected_time)
	if not np.isfinite(expected):
		raise ValueError("The expected extended time must be finite.")
	tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(expected))
	if not np.isclose(
		float(value[2]),
		expected,
		rtol=0.0,
		atol=float(tolerance),
	):
		raise RuntimeError(f"{context} and integration-grid times diverged.")
	value[2] = expected
	return value


def _extended_vector_field(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate ``X_K=(f(t,z), 1, -partial_t h)`` in ``(x,y,t,k)`` order."""
	value = _checked_extended_state(state, duplicated=False)
	physical = np.asarray(dynamics.vector_field(float(value[2]), value[:2]), dtype=float)
	momentum = np.asarray(
		dynamics.extended_momentum_derivative(float(value[2]), value[:2]),
		dtype=float,
	)
	if physical.shape != (2,) or momentum.size != 1:
		raise ValueError("The one-particle extended GC field changed shape.")
	return np.asarray((physical[0], physical[1], 1.0, momentum.reshape(-1)[0]))


def _extended_vector_field_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
) -> np.ndarray:
	"""Return the analytic ``4 x 4`` Jacobian of ``X_K``.

	Mixed and second time derivatives are evaluated through the effective
	potential contract. This supports static means and arbitrary positive-frequency
	HDF5 modes in addition to the normalized single-frequency potential.
	"""
	value = _checked_extended_state(state, duplicated=False)
	potential = dynamics.effective_potential
	if potential.interpolation_order < 3:
		raise ValueError(
			"Analytic full-state Jacobians require interpolation_order >= 3."
		)
	time = float(value[2])
	x = np.asarray([value[0]])
	y = np.asarray([value[1]])
	spatial = np.asarray(
		dynamics.particle_vector_field_jacobians(time, value[:2]),
		dtype=float,
	)[0]
	h_tx = float(potential.evaluate(time, x, y, dx=1, dt=1)[0])
	h_ty = float(potential.evaluate(time, x, y, dy=1, dt=1)[0])
	h_tt = float(potential.evaluate(time, x, y, dt=2)[0])
	result = np.zeros((4, 4), dtype=float)
	result[:2, :2] = spatial
	result[:2, 2] = (-h_ty, h_tx)
	result[3, :3] = (-h_tx, -h_ty, -h_tt)
	return result


def _flow_first_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Return the analytic Jacobian of the first-Hamiltonian shear."""
	value = _checked_extended_state(state, duplicated=True)
	result = _IDENTITY_8.copy()
	result[4:, :4] = duration * _extended_vector_field_jacobian(
		dynamics,
		value[:4],
	)
	return result


def _flow_second_jacobian(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Return the analytic Jacobian of the second-Hamiltonian shear."""
	value = _checked_extended_state(state, duplicated=True)
	result = _IDENTITY_8.copy()
	result[:4, 4:] = duration * _extended_vector_field_jacobian(
		dynamics,
		value[4:],
	)
	return result


def _flow_first(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Flow ``K(Z_1)`` exactly: hold the first copy and shear the second."""
	value = _checked_extended_state(state, duplicated=True)
	value[4:] += duration * _extended_vector_field(dynamics, value[:4])
	return value


def _flow_second(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	duration: float,
) -> np.ndarray:
	"""Flow ``K(Z_2)`` exactly: hold the second copy and shear the first."""
	value = _checked_extended_state(state, duplicated=True)
	value[:4] += duration * _extended_vector_field(dynamics, value[4:])
	return value


def _abba_base_map(
	dynamics: GuidingCenterDynamics,
	duration: float,
) -> _AnalyticExtendedMap:
	"""Return the palindromic four-shear full-state ABBA map on ``R^8``."""
	half_step = duration / 2.0

	def map_state(candidate: np.ndarray) -> np.ndarray:
		value = _flow_second(dynamics, candidate, half_step)
		value = _flow_first(dynamics, value, half_step)
		value = _flow_first(dynamics, value, half_step)
		return _flow_second(dynamics, value, half_step)

	def jacobian_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		total = _IDENTITY_8.copy()
		for flow, jacobian in (
			(_flow_second, _flow_second_jacobian),
			(_flow_first, _flow_first_jacobian),
			(_flow_first, _flow_first_jacobian),
			(_flow_second, _flow_second_jacobian),
		):
			factor = jacobian(dynamics, value, half_step)
			value = flow(dynamics, value, half_step)
			total = factor @ total
		return total

	return _AnalyticExtendedMap(
		map_state=map_state,
		jacobian_state=jacobian_state,
	)


def _composed_abba_base_map(
	dynamics: GuidingCenterDynamics,
	duration: float,
	coefficients: np.ndarray,
) -> _AnalyticExtendedMap:
	"""Compose unprojected full-state ABBA maps without diagonal projection."""
	composition = np.asarray(coefficients, dtype=float)
	if (
		composition.ndim != 1
		or composition.size == 0
		or not np.all(np.isfinite(composition))
	):
		raise ValueError("ABBA composition coefficients must be finite and non-empty.")
	base_maps = tuple(
		_abba_base_map(dynamics, float(coefficient * duration))
		for coefficient in composition
	)

	def map_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		for base_map in base_maps:
			value = np.asarray(base_map.map_state(value), dtype=float)
		return value

	def jacobian_state(candidate: np.ndarray) -> np.ndarray:
		value = _checked_extended_state(candidate, duplicated=True)
		total = _IDENTITY_8.copy()
		for base_map in base_maps:
			factor = np.asarray(base_map.jacobian_state(value), dtype=float)
			value = np.asarray(base_map.map_state(value), dtype=float)
			total = factor @ total
		return total

	return _AnalyticExtendedMap(
		map_state=map_state,
		jacobian_state=jacobian_state,
	)


__all__: list[str] = []
