# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Symmetric flow-composition solver used by full-cyclotron systems."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ._solver_common import (
    Flow,
    StepCallback,
    build_integration_targets,
    build_output_times,
    integrate_to_target,
    validate_solver_inputs,
)
from .integrators import SymplecticIntegrator
from .solution import OdeSolution


def solve_symplectic(
    chi: Flow,
    chi_star: Flow,
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
    t_eval: ArrayLike | None = None,
    method: str = "BM4",
    command: StepCallback | None = None,
    *,
    save_step: float | None = None,
    n_save_step: int | None = None,
) -> OdeSolution:
    """Integrate flows ``chi`` and ``chi_star`` by symmetric composition.

    ``step`` is the maximum internal step. With no explicit output policy, a
    state is stored every ``step`` and at the final time. ``t_eval`` stores only
    the requested times, while the integration still advances to the end of
    ``t_span``. ``save_step`` and ``n_save_step`` provide regular alternatives.
    """
    if not callable(chi) or not callable(chi_star):
        raise TypeError("`chi` and `chi_star` must be callable.")
    if command is not None and not callable(command):
        raise TypeError("`command` must be callable or None.")

    inputs = validate_solver_inputs(t_span, y0, step)
    output_times = build_output_times(inputs, t_eval, save_step, n_save_step)
    integration_targets = build_integration_targets(output_times, inputs.tf)
    integrator = SymplecticIntegrator(method)

    t = inputs.t0
    state = inputs.state.copy()
    result = np.empty(
        inputs.state.shape + (len(output_times),),
        dtype=inputs.state.dtype,
    )
    output_index = 0
    n_steps = 0
    smallest_step = math.inf
    largest_step = 0.0

    for raw_target in integration_targets:
        target = float(raw_target)
        t, state, internal_step, segment_steps = integrate_to_target(
            chi,
            chi_star,
            integrator,
            t,
            state,
            target,
            inputs.max_step,
            command,
        )
        if internal_step is not None:
            n_steps += segment_steps
            smallest_step = min(smallest_step, internal_step)
            largest_step = max(largest_step, internal_step)

        if output_index >= len(output_times) or raw_target != output_times[output_index]:
            continue
        stored_state = np.asarray(state)
        if stored_state.shape != inputs.state.shape:
            raise ValueError("`chi` and `chi_star` must preserve the shape of `y0`.")
        if not np.can_cast(stored_state.dtype, result.dtype, casting="same_kind"):
            raise ValueError(
                "The flow changed the state dtype incompatibly; provide `y0` "
                "with the required dtype."
            )
        result[..., output_index] = stored_state
        output_index += 1

    if output_index != len(output_times):
        raise RuntimeError("Not every requested output state was stored.")
    if n_steps == 0:
        smallest_step = inputs.max_step
        largest_step = inputs.max_step
    return OdeSolution(
        t=output_times,
        y=result,
        step=largest_step,
        requested_step=inputs.max_step,
        min_step=smallest_step,
        max_step=largest_step,
        n_steps=n_steps,
    )


__all__ = ["solve_symplectic"]
