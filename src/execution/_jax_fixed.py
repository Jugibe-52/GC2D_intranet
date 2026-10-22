"""Compiled fixed-step methods with device-resident stages and requested samples.

Only an explicitly selected JAX run imports this module. The time loop is
sequential, while every field evaluation and RK stage batches all particles.
"""

from functools import lru_cache, partial
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from states.history import States
from contracts.states.history import DiagnosticValue
from contracts.methods.compiled import CompiledStep
from execution._host import _step_count, _time_tolerance

if TYPE_CHECKING:
    from methods.base import IntegrationMethod


def _integrate(
    initial: Any, starts: Any, step: Any, times: Any,
    offsets: Any, modes: Any, durations: Any, *,
    kernel: CompiledStep,
) -> tuple[Any, Any, Any, Any, Any]:
    """Collect O(N * saved_times) storage without retaining every main state.

    Sample modes select the start (0), end (1), or an independent shadow map
    (2). Offsets group samples by accepted interval, including sparse output.
    Only the accepted state is fed back into the following main step.
    """
    n = initial.size // (kernel.physical_dimension * kernel.copies + (2 if kernel.track_energy else 0))
    physical_size = n * kernel.physical_dimension

    history: jax.Array = jnp.zeros((initial.size, times.size), dtype=initial.dtype).at[:, 0].set(initial)

    def shadow(time: Any, state: Any, duration: Any) -> Any:
        value: Any
        valid: Any
        value, _, valid = kernel(time, state, duration)
        return value, valid

    def accepted(carry: Any, index: Any) -> tuple[Any, Any]:
        before: Any
        saved: Any
        valid: Any
        converged: Any
        before, saved, valid, converged = carry
        after: Any
        statistics: dict[str, Any]
        solved: Any
        after, statistics, solved = kernel(starts[index], before, step)
        converged = converged & solved
        valid = valid & jnp.all(jnp.isfinite(after))

        def sample(output_index: Any, collected: Any) -> Any:
            values: Any
            finite: Any
            solved: Any
            values, finite, solved = collected
            value: Any
            success: Any
            value, success = jax.lax.switch(modes[output_index], (
                lambda: (before, jnp.array(True)),
                lambda: (after, jnp.array(True)),
                lambda: shadow(starts[index], before, durations[output_index]),
            ))
            values = values.at[:, output_index].set(value)
            return values, finite & jnp.all(jnp.isfinite(value)), solved & success

        saved, valid, converged = jax.lax.fori_loop(
            offsets[index], offsets[index + 1], sample, (saved, valid, converged))
        return (after, saved, valid, converged), statistics

    valid: jax.Array
    converged: jax.Array
    statistics: Any
    (_, history, valid, converged), statistics = jax.lax.scan(
        accepted, (initial, history, jnp.array(True), jnp.array(True)), jnp.arange(starts.size),
    )
    energy: Any | jax.Array = (kernel.energy(times, history[:physical_size])
              if kernel.track_energy else jnp.empty((0,), dtype=initial.dtype))
    return history, energy, valid & jnp.all(jnp.isfinite(energy)), converged, statistics


@lru_cache(maxsize=16)
def _compiled_integrator(kernel: CompiledStep) -> Callable[..., Any]:
    """Bound compilation reuse by physical dynamics, device and step controls."""
    return jax.jit(partial(_integrate, kernel=kernel))


def run_steps(method: "IntegrationMethod[Any]", kernel: CompiledStep,
              saved: States) -> dict[str, DiagnosticValue]:
    """Advance a prepared map and record its complete physical history in bulk.

    The executor validates host-only callbacks and progress before
    invoking this worker. The compiled loop performs no Python callbacks.
    Potential construction and schedule validation remain on the host.
    """
    request = method.request
    t0: float
    tf: float
    t0, tf = request.t_span
    count: int = _step_count(tf - t0, request.max_step)
    step = (tf - t0) / count
    starts = t0 + np.arange(count) * step
    ends = t0 + np.arange(1, count + 1) * step
    times = request.output_times
    tolerance: float = _time_tolerance(request.t_span)
    # Reproduce FixedStepController's endpoint priority and tolerance.
    owners: np.ndarray = np.minimum(np.searchsorted(ends + tolerance, times), count - 1)
    modes: np.ndarray = np.where(np.abs(times - ends[owners]) <= tolerance, 1,
                     np.where(np.abs(times - starts[owners]) <= tolerance, 0, 2))
    durations = times - starts[owners]
    offsets: np.ndarray = np.concatenate(([1], 1 + np.searchsorted(times[1:], ends + tolerance, side="right")))
    device = kernel.device
    inputs = (method.initial_state, starts, np.asarray(step), times,
              offsets.astype(np.int32), modes.astype(np.int32), durations)
    # Inputs select the execution device; spline constants follow compilation.
    # device_get synchronizes timing and copies only completed output.
    history: Any
    energy: Any
    valid: Any
    converged: Any
    statistics: Any
    history, energy, valid, converged, statistics = jax.device_get(_compiled_integrator(kernel)(
        *(jax.device_put(value, device) for value in inputs),
    ))
    if not bool(valid):
        raise ValueError("A numerical step or output sample became non-finite.")
    if not bool(converged):
        raise RuntimeError(f"{method.method_name} nonlinear solve did not converge within its tolerance and iteration limit.")
    assert method.state_formulation is not None
    states: np.ndarray
    auxiliary: dict[str, DiagnosticValue]
    states, auxiliary = method.state_formulation.extract_history(
        times, history, energy=energy if kernel.track_energy else None,
    )
    diagnostics: dict[str, DiagnosticValue] = {}
    diagnostics.update({
        "step_count": count, "step_start_times": starts, "step_times": ends,
        "step_sizes": np.full(count, step),
        "output_interpolation_count": int(np.count_nonzero(modes[1:] == 2)),
    })
    diagnostics.update(statistics)
    diagnostics.update(auxiliary)
    if kernel.finite_difference:
        diagnostics['finite_difference_batching'] = 'independent_particle_columns'
    saved.record_history(states)
    return diagnostics


__all__: list[str] = []
