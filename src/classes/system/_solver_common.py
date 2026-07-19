# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Validation, output scheduling and stepping shared by system solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .integrators import SymplecticIntegrator


Flow = Callable[[float, float, np.ndarray], np.ndarray]
StepCallback = Callable[[float, np.ndarray], None]


@dataclass(frozen=True)
class SolverInputs:
    """Normalized integration-domain inputs."""

    t0: float
    tf: float
    state: np.ndarray
    max_step: float


def validate_solver_inputs(
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
) -> SolverInputs:
    """Normalize solver inputs and reject invalid integration domains."""
    try:
        span = np.asarray(t_span, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_span` must contain exactly two finite numbers.") from exc
    if span.shape != (2,) or not np.all(np.isfinite(span)):
        raise ValueError("`t_span` must contain exactly two finite numbers.")
    t0, tf = map(float, span)
    if t0 > tf:
        raise ValueError("Values in `t_span` are not properly sorted.")

    try:
        max_step = float(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("`step` must be a positive finite number.") from exc
    if isinstance(step, (bool, np.bool_)) or not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("`step` must be a positive finite number.")

    state = np.asarray(y0)
    if state.ndim == 0 or state.size == 0:
        raise ValueError("`y0` must be a non-empty array-like state.")
    if not np.issubdtype(state.dtype, np.number):
        raise ValueError("`y0` must contain numeric values.")
    state = np.asarray(state, dtype=np.result_type(state.dtype, np.float64))
    if not np.all(np.isfinite(state)):
        raise ValueError("`y0` must contain only finite values.")
    return SolverInputs(t0=t0, tf=tf, state=state, max_step=max_step)


def _positive_interval(value: float, name: str) -> float:
    try:
        interval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a positive finite number.") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(interval) or interval <= 0:
        raise ValueError(f"`{name}` must be a positive finite number.")
    return interval


def _regular_output_times(t0: float, tf: float, interval: float) -> np.ndarray:
    """Return ``t0 + n * interval`` and include ``tf`` exactly once."""
    if t0 == tf:
        return np.asarray([t0], dtype=float)

    duration = tf - t0
    full_intervals = math.floor(duration / interval)
    output = t0 + interval * np.arange(full_intervals + 1, dtype=float)

    # A mathematically exact final multiple can land one ulp beyond ``tf``.
    upper_boundary = np.nextafter(tf, math.inf)
    output = output[output <= upper_boundary]
    if output.size == 0:
        output = np.asarray([t0], dtype=float)

    lower_boundary = np.nextafter(tf, -math.inf)
    if output[-1] >= lower_boundary:
        output[-1] = tf
    else:
        output = np.append(output, tf)
    return np.asarray(output, dtype=float)


def _validate_t_eval(t_eval: ArrayLike, t0: float, tf: float) -> np.ndarray:
    try:
        output = np.asarray(t_eval, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_eval` must be a non-empty one-dimensional sequence of finite times.") from exc
    if output.ndim != 1 or output.size == 0 or not np.all(np.isfinite(output)):
        raise ValueError("`t_eval` must be a non-empty one-dimensional sequence of finite times.")
    if np.any(np.diff(output) <= 0):
        raise ValueError("Values in `t_eval` must be strictly increasing.")
    if output[0] < t0 or output[-1] > tf:
        raise ValueError("Values in `t_eval` must lie within `t_span`.")
    return output.copy()


def build_output_times(
    inputs: SolverInputs,
    t_eval: ArrayLike | None,
    save_step: float | None,
    n_save_step: int | None,
) -> np.ndarray:
    """Build output times from one, and only one, sampling policy."""
    selected = sum(value is not None for value in (t_eval, save_step, n_save_step))
    if selected > 1:
        raise ValueError("`t_eval`, `save_step` and `n_save_step` are mutually exclusive.")

    if t_eval is not None:
        return _validate_t_eval(t_eval, inputs.t0, inputs.tf)
    if save_step is not None:
        interval = _positive_interval(save_step, "save_step")
        return _regular_output_times(inputs.t0, inputs.tf, interval)
    if n_save_step is not None:
        if (
            not isinstance(n_save_step, (int, np.integer))
            or isinstance(n_save_step, (bool, np.bool_))
            or int(n_save_step) < 1
        ):
            raise ValueError("`n_save_step` must be a positive integer.")
        count = int(n_save_step)
        if inputs.t0 != inputs.tf and count < 2:
            raise ValueError("`n_save_step` must be at least 2 for a non-empty time interval.")
        if inputs.t0 == inputs.tf:
            return np.asarray([inputs.t0], dtype=float)
        return np.asarray(np.linspace(inputs.t0, inputs.tf, count), dtype=float)
    return _regular_output_times(inputs.t0, inputs.tf, inputs.max_step)


def build_integration_targets(output_times: np.ndarray, tf: float) -> np.ndarray:
    """Ensure integration reaches ``tf`` even when it is not sampled."""
    if output_times[-1] == tf:
        return output_times
    return np.append(output_times, tf)


def step_count(duration: float, max_step: float) -> int:
    """Return the fewest steps whose size does not exceed ``max_step``."""
    ratio = duration / max_step
    return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


def _checked_flow_result(flow: Flow, h: float, t: float, state: np.ndarray) -> np.ndarray:
    result = np.asarray(flow(h, t, state))
    if result.shape != state.shape:
        raise ValueError("`chi` and `chi_star` must preserve the shape of `y0`.")
    return result


def advance_one_step(
    chi: Flow,
    chi_star: Flow,
    t: float,
    state: np.ndarray,
    stages: tuple[tuple[float, int], ...],
) -> tuple[float, np.ndarray]:
    """Apply every stage of one symplectic composition step."""
    for stage_step, stage_order in stages:
        if stage_order == 0:
            state = _checked_flow_result(chi, stage_step, t + stage_step, state)
        else:
            state = _checked_flow_result(chi_star, stage_step, t, state)
        t += stage_step
    return t, state


def integrate_to_target(
    chi: Flow,
    chi_star: Flow,
    integrator: SymplecticIntegrator,
    t: float,
    state: np.ndarray,
    target: float,
    max_step: float,
    command: StepCallback | None,
) -> tuple[float, np.ndarray, float | None, int]:
    """Advance to one requested output time without exceeding ``max_step``."""
    duration = target - t
    if duration == 0:
        return target, state, None, 0
    if duration < 0:
        raise RuntimeError("Internal integration targets must be sorted.")

    count = step_count(duration, max_step)
    internal_step = duration / count
    stages = tuple(
        (float(coefficient * internal_step), int(stage_order))
        for coefficient, stage_order in zip(
            integrator.alpha_s,
            integrator.alpha_o,
            strict=True,
        )
    )
    segment_start = t
    for index in range(count):
        t, state = advance_one_step(chi, chi_star, t, state, stages)
        # Prevent accumulated roundoff from changing the integration schedule.
        t = segment_start + (index + 1) * internal_step
        if command is not None:
            command(t, state)
    return target, state, internal_step, count


def planned_step_count(inputs: SolverInputs, targets: np.ndarray) -> int:
    """Count complete composition steps in an already validated schedule."""
    total = 0
    start = inputs.t0
    for target in targets:
        duration = float(target) - start
        if duration > 0:
            total += step_count(duration, inputs.max_step)
        start = float(target)
    return total
