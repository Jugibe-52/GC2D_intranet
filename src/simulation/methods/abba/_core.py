"""Shared endpoint-time A-B-B-A map on two physical state copies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import DynamicalSystem


@dataclass(frozen=True, slots=True)
class _ABBAStages:
	"""State points traversed by one explicit endpoint-time A-B-B-A map."""

	u_initial: np.ndarray
	v_initial: np.ndarray
	u_first: np.ndarray
	v_final: np.ndarray
	u_final: np.ndarray
	residual: np.ndarray


def _checked_vector_field(
	dynamics: DynamicalSystem,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	"""Evaluate a finite vector field without allowing a layout change."""
	result = np.asarray(dynamics.vector_field(t, state), dtype=float)
	if result.shape != state.shape or not np.all(np.isfinite(result)):
		raise ValueError("The vector field changed shape or became non-finite.")
	return result


def _evaluate_unprojected_stages(
	dynamics: DynamicalSystem,
	t: float,
	u_initial: np.ndarray,
	v_initial: np.ndarray,
	step: float,
) -> _ABBAStages:
	"""Apply one signed A-B-B-A map to two independent physical copies."""
	half_step = step / 2.0
	final_time = t + step
	u_first = u_initial + half_step * _checked_vector_field(
		dynamics,
		t,
		v_initial,
	)
	v_first = v_initial + half_step * _checked_vector_field(
		dynamics,
		t,
		u_first,
	)
	v_final = v_first + half_step * _checked_vector_field(
		dynamics,
		final_time,
		u_first,
	)
	u_final = u_first + half_step * _checked_vector_field(
		dynamics,
		final_time,
		v_final,
	)

	# The copy separation is the unprojected map residual. Projection-specific
	# formulations add their multiplier contribution outside this neutral core.
	residual = u_final - v_final
	return _ABBAStages(
		u_initial=u_initial,
		v_initial=v_initial,
		u_first=u_first,
		v_final=v_final,
		u_final=u_final,
		residual=residual,
	)


__all__: list[str] = []
