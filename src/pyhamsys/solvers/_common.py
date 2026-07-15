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
    save_step: Union[float, None]
    t_eval: Union[xp.ndarray, None]


def _validate_solver_inputs(
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
    save_step: Union[float, None],
    t_eval: Union[ArrayLike, None],
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

    normalized_save_step: Union[float, None] = None
    if save_step is not None:
        if t_eval is not None:
            raise ValueError("`save_step` and `t_eval` are mutually exclusive.")
        try:
            normalized_save_step = float(save_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("`save_step` must be a positive finite number.") from exc
        if not xp.isfinite(normalized_save_step) or normalized_save_step <= 0:
            raise ValueError("`save_step` must be a positive finite number.")

    if t_eval is None:
        return _SolverInputs(
            t0=t0,
            tf=tf,
            state=state,
            max_step=max_step,
            save_step=normalized_save_step,
            t_eval=None,
        )

    try:
        eval_times = xp.asarray(t_eval, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_eval` must be a one-dimensional sequence of finite times.") from exc
    if eval_times.ndim != 1:
        raise ValueError("`t_eval` must be 1-dimensional.")
    if eval_times.size == 0:
        raise ValueError("`t_eval` must contain at least one time.")
    if not xp.all(xp.isfinite(eval_times)):
        raise ValueError("`t_eval` must contain only finite times.")
    if xp.any(eval_times < t0) or xp.any(eval_times > tf):
        raise ValueError("Values in `t_eval` are not within `t_span`.")
    if xp.any(xp.diff(eval_times) <= 0):
        raise ValueError("Values in `t_eval` are not properly sorted.")
    return _SolverInputs(
        t0=t0,
        tf=tf,
        state=state,
        max_step=max_step,
        save_step=None,
        t_eval=eval_times,
    )


def _step_count(duration: float, max_step: float) -> int:
    """Return the fewest steps that do not exceed ``max_step``."""
    ratio = duration / max_step
    # Move by one representable float toward the lower integer to neutralize
    # division roundoff without using a tolerance that grows with ``ratio``.
    return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


def _regular_output_times(t0: float, tf: float, interval: float) -> xp.ndarray:
    """Return a regular grid including both endpoints."""
    if t0 == tf:
        return xp.asarray([t0], dtype=float)

    duration = tf - t0
    regular_count = int(math.floor(duration / interval))
    times = t0 + xp.arange(regular_count + 1, dtype=float) * interval
    endpoint_tolerance = max(
        math.ulp(tf),
        math.ulp(float(times[-1])),
        math.ulp(interval),
    )
    if math.isclose(float(times[-1]), tf, rel_tol=0.0, abs_tol=endpoint_tolerance):
        times[-1] = tf
    elif times[-1] < tf:
        times = xp.append(times, tf)
    else:
        times[-1] = tf
    return times


def _build_output_times(inputs: _SolverInputs) -> xp.ndarray:
    """Create exact output times for automatic or user-supplied sampling."""
    if inputs.t_eval is not None:
        return inputs.t_eval.copy()
    interval = inputs.save_step or inputs.max_step
    return _regular_output_times(inputs.t0, inputs.tf, interval)


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

