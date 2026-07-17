# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Shared validation, scheduling and stepping infrastructure."""

from dataclasses import dataclass
import math
from typing import Callable, Union

import numpy as xp
from numpy.typing import ArrayLike

from ..integrators import SymplecticIntegrator


Flow = Callable[[float, float, xp.ndarray], xp.ndarray]
StepCallback = Callable[[float, xp.ndarray], None]
@dataclass(frozen=True)
class _SolverInputs:
    t0: float
    tf: float
    state: xp.ndarray
    max_step: float
    n_save_step: int


def _validate_solver_inputs(
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
    n_save_step: int,
) -> _SolverInputs:
    """Normalize solver inputs and reject invalid integration domains."""
    try:
        span = xp.asarray(t_span, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_span` must contain exactly two finite numbers.") from exc
    if span.shape != (2,) or not xp.all(xp.isfinite(span)):
        raise ValueError("`t_span` must contain exactly two finite numbers.")
    t0, tf = map(float, span)
    if t0 > tf:
        raise ValueError("Values in `t_span` are not properly sorted.")

    try:
        max_step = float(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("`step` must be a positive finite number.") from exc
    if not xp.isfinite(max_step) or max_step <= 0:
        raise ValueError("`step` must be a positive finite number.")

    state = xp.asarray(y0)
    if state.ndim == 0 or state.size == 0:
        raise ValueError("`y0` must be a non-empty array-like state.")
    if not xp.issubdtype(state.dtype, xp.number):
        raise ValueError("`y0` must contain numeric values.")
    state = xp.asarray(state, dtype=xp.result_type(state.dtype, xp.float64))
    if not xp.all(xp.isfinite(state)):
        raise ValueError("`y0` must contain only finite values.")

    if not isinstance(n_save_step, (int, xp.integer)) or isinstance(n_save_step, bool):
        raise ValueError("`n_save_step` must be a positive integer.")
    normalized_n_save_step = int(n_save_step)
    if normalized_n_save_step < 1:
        raise ValueError("`n_save_step` must be a positive integer.")
    if t0 != tf and normalized_n_save_step < 2:
        raise ValueError("`n_save_step` must be at least 2 for a non-empty time interval.")
    return _SolverInputs(
        t0=t0,
        tf=tf,
        state=state,
        max_step=max_step,
        n_save_step=normalized_n_save_step,
    )


def _step_count(duration: float, max_step: float) -> int:
    """Return the fewest steps that do not exceed ``max_step``."""
    ratio = duration / max_step
    # Move by one representable float toward the lower integer to neutralize
    # division roundoff without using a tolerance that grows with ``ratio``.
    return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


def _build_output_times(inputs: _SolverInputs) -> xp.ndarray:
    """Create uniformly distributed output times including both endpoints."""
    return xp.asarray(xp.linspace(inputs.t0, inputs.tf, inputs.n_save_step))


def _build_integration_targets(output_times: xp.ndarray, tf: float) -> xp.ndarray:
    """Ensure integration reaches ``tf`` even when it is not sampled."""
    if output_times[-1] == tf:
        return output_times
    return xp.append(output_times, tf)


def _advance_one_step(
    chi: Flow,
    chi_star: Flow,
    t: float,
    y: xp.ndarray,
    stages: tuple[tuple[float, int], ...],
) -> tuple[float, xp.ndarray]:
    """Apply every stage of one symplectic composition step."""
    for stage_step, stage_order in stages:
        if stage_order == 0:
            y = chi(stage_step, t + stage_step, y)
        else:
            y = chi_star(stage_step, t, y)
        t += stage_step
    return t, y


def _integrate_to_target(
    chi: Flow,
    chi_star: Flow,
    integrator: SymplecticIntegrator,
    t: float,
    y: xp.ndarray,
    target: float,
    max_step: float,
    command: Union[StepCallback, None],
) -> tuple[float, xp.ndarray, Union[float, None], int]:
    """Advance to one requested output time without exceeding ``max_step``."""
    duration = target - t
    if duration == 0:
        return target, y, None, 0

    count = _step_count(duration, max_step)
    internal_step = duration / count
    stages = tuple(
        (float(coefficient * internal_step), int(stage_order))
        for coefficient, stage_order in zip(
            integrator.alpha_s, integrator.alpha_o
        )
    )
    segment_start = t
    for index in range(count):
        t, y = _advance_one_step(chi, chi_star, t, y, stages)
        # Prevent accumulated roundoff from changing the integration schedule.
        t = segment_start + (index + 1) * internal_step
        if command is not None:
            command(t, y)
    return target, y, internal_step, count
