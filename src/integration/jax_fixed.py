"""Compiled fixed-step RK4 with device-resident stages and requested samples.

Only an explicitly selected JAX run imports this module. The time loop is
sequential, while every field evaluation and RK stage batches all particles.
"""

from functools import partial
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from contracts.execution import Execution
from contracts.result import IntegrationData
from dynamics._jax import JaxDynamics, bind_dynamics
from integration._fixed import _step_count
from integration.core import _time_tolerance
from methods.classical._rk4_core import physical_step, momentum_increment

if TYPE_CHECKING:
    from methods.classical.rk4 import RK4


@partial(jax.jit, static_argnames=("dynamics", "track_energy"))
def _integrate(
    initial: Any, starts: Any, step: Any, times: Any,
    offsets: Any, modes: Any, durations: Any, *,
    dynamics: JaxDynamics, track_energy: bool,
) -> tuple[Any, Any, Any]:
    """Collect O(N * saved_times) storage without retaining every main state.

    Sample modes select the start (0), end (1), or an independent shadow map
    (2). Offsets group samples by accepted interval, including sparse output.
    Only the accepted state is fed back into the following main step.
    """
    n = initial.size // (dynamics.dimension + (2 if track_energy else 0))
    physical_size = n * dynamics.dimension

    def advance(time: Any, before: Any, h: Any) -> Any:
        physical, stages = physical_step(dynamics.vector_field, time, before[:physical_size], h)
        if not track_energy:
            return physical
        increment = momentum_increment(dynamics.momentum_rate, time, h, stages)
        return jnp.concatenate((physical, jnp.full((n,), time + h), before[-n:] + increment))

    history = jnp.zeros((initial.size, times.size), dtype=initial.dtype).at[:, 0].set(initial)

    def accepted(carry: Any, index: Any) -> tuple[Any, None]:
        before, saved, valid = carry
        after = advance(starts[index], before, step)
        valid = valid & jnp.all(jnp.isfinite(after))

        def sample(output_index: Any, collected: Any) -> Any:
            values, finite = collected
            value = jax.lax.switch(modes[output_index], (
                lambda: before,
                lambda: after,
                lambda: advance(starts[index], before, durations[output_index]),
            ))
            values = values.at[:, output_index].set(value)
            return values, finite & jnp.all(jnp.isfinite(value))

        saved, valid = jax.lax.fori_loop(offsets[index], offsets[index + 1], sample, (saved, valid))
        return (after, saved, valid), None

    (_, history, valid), _ = jax.lax.scan(
        accepted, (initial, history, jnp.array(True)), jnp.arange(starts.size),
    )
    energy = (dynamics.hamiltonian(times, history[:physical_size])
              if track_energy else jnp.empty((0,), dtype=initial.dtype))
    return history, energy, valid & jnp.all(jnp.isfinite(energy))


def integrate_rk4(method: "RK4", execution: Execution) -> IntegrationData:
    """Run one fresh RK4 instance and transfer the finished result to NumPy.

    Python step observers and per-step progress belong to the host controller;
    requesting them here is rejected rather than introducing hidden transfers.
    Potential construction and schedule validation remain on the host.
    """
    if method._status != "ready":
        raise RuntimeError("Integration requires a fresh method.new_run(problem, request).")
    method._status = "running"
    try:
        if method.step_observer is not None or method.progress:
            raise NotImplementedError(
                "JAX RK4 requires step_observer=None and progress=False; "
                "Python callbacks are supported by the SciPy execution path."
            )
        dynamics = bind_dynamics(method.dynamics, execution)
        dynamics.evaluator.check_ready()
        request = method.request
        t0, tf = request.t_span
        count = _step_count(tf - t0, request.max_step)
        step = (tf - t0) / count
        starts = t0 + np.arange(count) * step
        ends = t0 + np.arange(1, count + 1) * step
        times = request.output_times
        tolerance = _time_tolerance(request.t_span)
        # Reproduce FixedStepController's endpoint priority and tolerance.
        owners = np.minimum(np.searchsorted(ends + tolerance, times), count - 1)
        modes = np.where(np.abs(times - ends[owners]) <= tolerance, 1,
                         np.where(np.abs(times - starts[owners]) <= tolerance, 0, 2))
        durations = times - starts[owners]
        offsets = np.concatenate(([1], 1 + np.searchsorted(times[1:], ends + tolerance, side="right")))
        device = dynamics.evaluator.device
        inputs = (method.initial_state, starts, np.asarray(step), times,
                  offsets.astype(np.int32), modes.astype(np.int32), durations)
        # All inputs and spline buffers are committed to the requested device.
        # device_get synchronizes timing and copies only completed output.
        history, energy, valid = jax.device_get(_integrate(
            *(jax.device_put(value, device) for value in inputs),
            dynamics=dynamics, track_energy=method.track_energy,
        ))
        if not bool(valid):
            raise ValueError("A numerical step or output sample became non-finite.")
        states, auxiliary = method.state_formulation.extract_history(
            times, history, energy=energy if method.track_energy else None,
        )
        diagnostics = dict(method.metadata)
        diagnostics.update({
            "execution_backend": "jax", "execution_device": execution.device,
            "execution_device_index": execution.device_index,
            "execution_device_kind": str(device.device_kind),
            "step_count": count, "step_start_times": starts, "step_times": ends,
            "step_sizes": np.full(count, step),
            "output_interpolation_count": int(np.count_nonzero(modes[1:] == 2)),
        })
        diagnostics.update(auxiliary)
        return IntegrationData(times, states, diagnostics)
    finally:
        method._status = "finished"


__all__: list[str] = []
