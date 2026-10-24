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
from diagnostics.methods.energy import PreparedEnergyObserver, EnergyFunction, momentum_increment
from execution._host import _fixed_schedule, _time_tolerance

if TYPE_CHECKING:
    from methods.base import IntegrationMethod


def _integrate(
    initial: Any, initial_momentum: Any, starts: Any, steps: Any, times: Any,
    offsets: Any, modes: Any, durations: Any, *,
    kernel: CompiledStep, hamiltonian: EnergyFunction | None, momentum_rate: EnergyFunction | None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Collect O(N * saved_times) storage without retaining every main state.

    Sample modes select the start (0), end (1), or an independent shadow map
    (2). Offsets group samples by accepted interval, including sparse output.
    Only the accepted state is fed back into the following main step.
    """
    history: jax.Array = jnp.zeros((initial.size, times.size), dtype=initial.dtype).at[:, 0].set(initial)
    momenta: jax.Array = jnp.zeros((initial_momentum.size, times.size), dtype=initial.dtype).at[:, 0].set(initial_momentum)

    def shadow(time: Any, state: Any, momentum: Any, duration: Any) -> Any:
        value: Any
        auxiliary: Any
        valid: Any
        value, quadrature, _, valid = kernel(time, state, duration)
        auxiliary = momentum if momentum_rate is None else momentum + momentum_increment(momentum_rate, quadrature, momentum.size)
        return value, auxiliary, valid

    def accepted(carry: Any, index: Any) -> tuple[Any, Any]:
        before: Any
        momentum_before: Any
        saved: Any
        saved_momenta: Any
        valid: Any
        converged: Any
        before, momentum_before, saved, saved_momenta, valid, converged = carry
        after: Any
        momentum_after: Any
        statistics: dict[str, Any]
        solved: Any
        after, quadrature, statistics, solved = kernel(starts[index], before, steps[index])
        momentum_after = momentum_before if momentum_rate is None else momentum_before + momentum_increment(momentum_rate, quadrature, momentum_before.size)
        converged = converged & solved
        valid = valid & jnp.all(jnp.isfinite(after)) & jnp.all(jnp.isfinite(momentum_after))

        def sample(output_index: Any, collected: Any) -> Any:
            values: Any
            auxiliaries: Any
            finite: Any
            solved: Any
            values, auxiliaries, finite, solved = collected
            value: Any
            momentum: Any
            success: Any
            value, momentum, success = jax.lax.switch(modes[output_index], (
                lambda: (before, momentum_before, jnp.array(True)),
                lambda: (after, momentum_after, jnp.array(True)),
                lambda: shadow(starts[index], before, momentum_before, durations[output_index]),
            ))
            values = values.at[:, output_index].set(value)
            auxiliaries = auxiliaries.at[:, output_index].set(momentum)
            return values, auxiliaries, finite & jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(momentum)), solved & success

        saved, saved_momenta, valid, converged = jax.lax.fori_loop(
            offsets[index], offsets[index + 1], sample, (saved, saved_momenta, valid, converged))
        return (after, momentum_after, saved, saved_momenta, valid, converged), statistics

    valid: jax.Array
    converged: jax.Array
    statistics: Any
    (_, _, history, momenta, valid, converged), statistics = jax.lax.scan(
        accepted, (initial, initial_momentum, history, momenta, jnp.array(True), jnp.array(True)), jnp.arange(starts.size),
    )
    energy: Any | jax.Array = (hamiltonian(times, history)
              if hamiltonian is not None else jnp.empty((0,), dtype=initial.dtype))
    return history, momenta, energy, valid & jnp.all(jnp.isfinite(energy)), converged, statistics


@lru_cache(maxsize=16)
def _compiled_integrator(kernel: CompiledStep, hamiltonian: EnergyFunction | None, momentum_rate: EnergyFunction | None) -> Callable[..., Any]:
    """Bound compilation reuse by physical dynamics, device and step controls."""
    return jax.jit(partial(_integrate, kernel=kernel, hamiltonian=hamiltonian, momentum_rate=momentum_rate))


def run_steps(method: "IntegrationMethod[Any]", kernel: CompiledStep,
              saved: States, *, energy: PreparedEnergyObserver | None = None) -> dict[str, DiagnosticValue]:
    """Advance a prepared map and record its complete physical history in bulk.

    The executor validates host-only callbacks and progress before
    invoking this worker. The compiled loop performs no Python callbacks.
    Potential construction and schedule validation remain on the host.
    """
    request = method.request
    t0: float
    tf: float
    t0, tf = request.t_span
    from methods.base import FixedStepMethod

    if not isinstance(method, FixedStepMethod):
        raise TypeError("Compiled fixed integration requires a FixedStepMethod.")
    starts: np.ndarray
    ends: np.ndarray
    steps: np.ndarray
    starts, ends, steps = _fixed_schedule(request.t_span, method.step)
    count: int = starts.size
    times = request.output_times
    tolerance: float = _time_tolerance(request.t_span)
    # Reproduce FixedStepController's endpoint priority and tolerance.
    owners: np.ndarray = np.minimum(np.searchsorted(ends + tolerance, times), count - 1)
    modes: np.ndarray = np.where(np.abs(times - ends[owners]) <= tolerance, 1,
                     np.where(np.abs(times - starts[owners]) <= tolerance, 0, 2))
    durations = times - starts[owners]
    offsets: np.ndarray = np.concatenate(([1], 1 + np.searchsorted(times[1:], ends + tolerance, side="right")))
    device = kernel.device
    inputs = (method.initial_state, energy.momentum if energy is not None else np.empty(0), starts, steps, times,
              offsets.astype(np.int32), modes.astype(np.int32), durations)
    # Inputs select the execution device; spline constants follow compilation.
    # device_get synchronizes timing and copies only completed output.
    history: Any
    momenta: Any
    energy_history: Any
    valid: Any
    converged: Any
    statistics: Any
    history, momenta, energy_history, valid, converged, statistics = jax.device_get(_compiled_integrator(
        kernel, energy.hamiltonian if energy is not None else None, energy.momentum_rate if energy is not None else None)(
        *(jax.device_put(value, device) for value in inputs),
    ))
    if not bool(valid):
        raise ValueError("A numerical step or output sample became non-finite.")
    if not bool(converged):
        raise RuntimeError(f"{method.method_name} nonlinear solve did not converge within its tolerance and iteration limit.")
    auxiliary: dict[str, DiagnosticValue] = {} if energy is None else energy.diagnose_history(
        times, history, {'extended_momentum': momenta}, energy=energy_history,
    )
    diagnostics: dict[str, DiagnosticValue] = {}
    diagnostics.update({
        "step_count": count, "step_start_times": starts, "step_times": ends,
        "step_sizes": steps,
        "output_interpolation_count": int(np.count_nonzero(modes[1:] == 2)),
    })
    diagnostics.update(statistics)
    diagnostics.update(auxiliary)
    if kernel.finite_difference:
        diagnostics['finite_difference_batching'] = 'independent_particle_columns'
    saved.record_history(history)
    return diagnostics


__all__: list[str] = []
