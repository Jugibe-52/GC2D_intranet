# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Explicit symplectic splitting solver."""

import math
from typing import Union

import numpy as xp
from numpy.typing import ArrayLike

from ..integrators import SymplecticIntegrator
from ..solution import OdeSolution
from ._common import (
    Flow,
    StepCallback,
    _build_integration_targets,
    _build_output_times,
    _integrate_to_target,
    _validate_solver_inputs,
)


def solve_ivp_symp(
    chi: Flow,
    chi_star: Flow,
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
    method: str = "BM4",
    command: Union[StepCallback, None] = None,
    *,
    n_save_step: int = 241,
) -> OdeSolution:
    """Solve a Hamiltonian initial-value problem by symplectic splitting.

    ``step`` is the maximum internal step size. ``n_save_step`` stores exactly
    that many uniformly distributed states, including both ends of ``t_span``.

    ``chi`` and ``chi_star`` receive ``(h, t, y)`` and must return a state with
    the same shape as ``y``. The optional ``command`` callback runs after each
    complete internal step and receives the live state: mutating it changes the
    integration. Integration always reaches the end of ``t_span``, even when
    the last requested sample precedes it.

    The returned :class:`OdeSolution` includes ``requested_step``, ``min_step``,
    ``max_step`` and ``n_steps`` in addition to ``t``, ``y`` and the legacy
    ``step`` attribute. ``step`` is kept as an alias of ``max_step``.
    """
    if not callable(chi) or not callable(chi_star):
        raise TypeError("`chi` and `chi_star` must be callable.")
    if command is not None and not callable(command):
        raise TypeError("`command` must be callable or None.")

    inputs = _validate_solver_inputs(t_span, y0, step, n_save_step)
    output_times = _build_output_times(inputs)
    integration_targets = _build_integration_targets(output_times, inputs.tf)
    integrator = SymplecticIntegrator(method)

    t = inputs.t0
    y = inputs.state.copy()
    result = xp.empty(inputs.state.shape + (len(output_times),), dtype=inputs.state.dtype)
    output_index = 0
    n_steps = 0
    smallest_step = math.inf
    largest_step = 0.0

    for target in integration_targets:
        t, y, internal_step, segment_steps = _integrate_to_target(
            chi,
            chi_star,
            integrator,
            t,
            y,
            float(target),
            inputs.max_step,
            command,
        )
        if internal_step is not None:
            n_steps += segment_steps
            smallest_step = min(smallest_step, internal_step)
            largest_step = max(largest_step, internal_step)

        stored_state = xp.asarray(y)
        if stored_state.shape != inputs.state.shape:
            raise ValueError("`chi` and `chi_star` must preserve the shape of `y0`.")
        if output_index >= len(output_times) or target != output_times[output_index]:
            continue
        if not xp.can_cast(stored_state.dtype, result.dtype, casting="same_kind"):
            raise ValueError(
                "The flow changed the state dtype incompatibly; provide `y0` "
                "with the required dtype."
            )
        result[..., output_index] = stored_state
        output_index += 1

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
