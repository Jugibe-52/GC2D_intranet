"""Compiled fixed-step methods with device-resident stages and requested samples.

Only an explicitly selected JAX run imports this module. The time loop is
sequential, while every field evaluation and RK stage batches all particles.
"""

from functools import lru_cache, partial
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from states.history import States
from contracts.states.history import DiagnosticValue
from contracts.methods.compiled import CompiledStep
from diagnostics.methods.energy import PreparedEnergyObserver, EnergyFunction, endpoint_diagnostics
from contracts.execution.plan import FixedStepPlan

if TYPE_CHECKING:
    from methods.base import IntegrationMethod
    from jax.stages import Compiled


@dataclass(frozen=True)
class PreparedIntegration:
    """A compiled loop with fresh device inputs and host recording information.

    Inputs contain physical state (d*N), main-step and
    sampling arrays. Only the executable is cached; each job owns these inputs.
    """

    executable: "Compiled"
    inputs: tuple[jax.Array, ...]
    step_starts: np.ndarray
    step_times: np.ndarray
    step_sizes: np.ndarray
    output_times: np.ndarray
    interior_sample_count: int
    method_name: str
    finite_difference: bool
    energy: PreparedEnergyObserver | None


def _integrate(
    initial: Any, starts: Any, steps: Any, times: Any,
    offsets: Any, modes: Any, durations: Any, *,
    kernel: CompiledStep, hamiltonian: EnergyFunction | None, time_derivative: EnergyFunction | None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Advance physical states and observe only accepted physical endpoints."""
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
        after, statistics, solved = kernel(starts[index], before, steps[index])
        converged = converged & solved
        valid = valid & jnp.all(jnp.isfinite(after))
        if hamiltonian is not None and time_derivative is not None:
            observations: dict[str, Any] = endpoint_diagnostics(
                hamiltonian(starts[index], before),
                hamiltonian(starts[index] + steps[index], after),
                time_derivative(starts[index], before),
                time_derivative(starts[index] + steps[index], after))
            statistics = {**statistics, **observations}
            for observation in observations.values():
                valid = valid & jnp.all(jnp.isfinite(observation))

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
        accepted, (initial, history, jnp.array(True), jnp.array(True)), jnp.arange(starts.size))
    energy: Any = hamiltonian(times, history) if hamiltonian is not None else jnp.empty((0,))
    derivative: Any = time_derivative(times, history) if time_derivative is not None else jnp.empty((0,))
    return history, derivative, energy, valid & jnp.all(jnp.isfinite(energy)) & jnp.all(jnp.isfinite(derivative)), converged, statistics


@lru_cache(maxsize=16)
def _compiled_integrator(
    kernel: CompiledStep, hamiltonian: EnergyFunction | None,
    time_derivative: EnergyFunction | None, signature: tuple[jax.ShapeDtypeStruct, ...],
) -> "Compiled":
    """Cache executables by physics, controls and input types/placement, not values."""
    return jax.jit(partial(
        _integrate, kernel=kernel, hamiltonian=hamiltonian, time_derivative=time_derivative,
    )).lower(*signature).compile()


def prepare_integration(
    method: "IntegrationMethod[Any]", kernel: CompiledStep, plan: FixedStepPlan,
    *, initial_state: np.ndarray, energy: PreparedEnergyObserver | None = None,
) -> PreparedIntegration:
    """Place a shared temporal plan and compile without advancing the solution."""
    host_inputs = (initial_state,
                   plan.step_starts, plan.step_sizes, plan.output_times,
                   plan.sample_offsets, plan.sample_modes, plan.sample_durations)
    inputs: tuple[jax.Array, ...] = tuple(jax.device_put(value, kernel.device) for value in host_inputs)
    # Abstract inputs preserve placement while keeping initial values out of the
    # executable cache. The same specialization can execute fresh physical data.
    signature: tuple[jax.ShapeDtypeStruct, ...] = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding, weak_type=value.weak_type)
        for value in inputs
    )
    executable: Compiled = _compiled_integrator(
        kernel, energy.hamiltonian if energy is not None else None,
        energy.time_derivative if energy is not None else None, signature,
    )
    return PreparedIntegration(
        executable, inputs, plan.step_starts, plan.step_ends, plan.step_sizes, plan.output_times, plan.interior_sample_count,
        method.method_name, kernel.finite_difference, energy,
    )


def run_steps(prepared: PreparedIntegration, saved: States) -> dict[str, DiagnosticValue]:
    """Execute an already compiled integration and record its physical history."""
    # device_get waits for execution and copies only completed output.
    history: Any
    derivatives: Any
    energy_history: Any
    valid: Any
    converged: Any
    statistics: Any
    history, derivatives, energy_history, valid, converged, statistics = jax.device_get(
        prepared.executable(*prepared.inputs))
    if not bool(valid):
        raise ValueError("A numerical step or output sample became non-finite.")
    if not bool(converged):
        raise RuntimeError(f"{prepared.method_name} nonlinear solve did not converge within its tolerance and iteration limit.")
    energy: PreparedEnergyObserver | None = prepared.energy
    auxiliary: dict[str, DiagnosticValue] = {} if energy is None else energy.diagnose_history(
        prepared.output_times, history, energy=energy_history, derivative=derivatives,
    )
    diagnostics: dict[str, DiagnosticValue] = {}
    diagnostics.update({
        "step_count": prepared.step_sizes.size, "step_start_times": prepared.step_starts,
        "step_times": prepared.step_times, "step_sizes": prepared.step_sizes,
        "output_interpolation_count": prepared.interior_sample_count,
    })
    diagnostics.update(statistics)
    diagnostics.update(auxiliary)
    if prepared.finite_difference:
        diagnostics['finite_difference_batching'] = 'vectorized_global_columns'
    saved.record_history(history.T)
    return diagnostics


__all__: list[str] = []
