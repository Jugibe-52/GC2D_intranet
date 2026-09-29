"""Device snapshots of the built-in particle dynamics; imported only on demand."""

from functools import lru_cache
from typing import Any

import jax.numpy as jnp

from contracts.execution_options import ExecutionOptions
from dynamics._equations import gc_velocity, fc_velocity, fc_hamiltonian
from dynamics.gc import GuidingCenterDynamics
from dynamics.fc import FullCyclotronDynamics
from dynamics.protocols import DynamicalSystem
from potential.potential import Potential
from potential.jax_evaluator import JaxPotentialEvaluator


class JaxDynamics:
    """A fixed field and scalar parameters, shared by successive compiled runs.

    Particle states use (components * N, *sample_axes), with two components for
    GC and four for FC. The physical equations are shared with CPU dynamics.
    """

    def __init__(self, potential: Potential, execution: ExecutionOptions, dimension: int,
                 velocity_scale: float, electric_scale: float, frequency: float) -> None:
        # Reuse the potential's existing device buffers and canonical spline data.
        evaluator = potential._evaluator(execution)
        assert isinstance(evaluator, JaxPotentialEvaluator)
        self.evaluator = evaluator
        self.dimension = dimension
        self.velocity_scale = velocity_scale
        self.electric_scale = electric_scale
        self.frequency = frequency

    def vector_field(self, time: Any, state: Any) -> Any:
        """Vectorize over all particles, leaving their arrays on the device."""
        blocks = jnp.split(state, self.dimension, axis=0)
        ex, ey = self.evaluator.electric_field(time, blocks[0], blocks[1])
        rates = (gc_velocity(ex, ey) if self.dimension == 2 else fc_velocity(
            self.electric_scale * ex, self.electric_scale * ey, blocks[2], blocks[3],
            velocity_scale=self.velocity_scale, larmor_frequency=self.frequency,
        ))
        return jnp.concatenate(rates, axis=0)

    def hamiltonian(self, time: Any, state: Any) -> Any:
        """Evaluate the physical energy at saved states on the same device."""
        blocks = jnp.split(state, self.dimension, axis=0)
        phi = self.evaluator.evaluate(time, blocks[0], blocks[1])
        if self.dimension == 2:
            return phi
        return fc_hamiltonian(phi, blocks[2], blocks[3],
                              velocity_scale=self.velocity_scale,
                              electric_scale=self.electric_scale)

    def momentum_rate(self, time: Any, state: Any) -> Any:
        """Return -partial_t H for each particle."""
        x, y, *_ = jnp.split(state, self.dimension, axis=0)
        return -self.electric_scale * self.evaluator.evaluate(time, x, y, dt=1)

    def particle_jacobians(self, time: Any, state: Any) -> Any:
        """Exact GC Hessian blocks with the CPU component/sign convention."""
        if self.dimension != 2:
            raise TypeError("Analytic particle Jacobians require GC dynamics.")
        if self.evaluator.interpolation_order < 3:
            raise ValueError("Exact GC Jacobians require interpolation_order >= 3.")
        x, y = jnp.split(state, 2)
        xx = self.evaluator.evaluate(time, x, y, dx=2)
        xy = self.evaluator.evaluate(time, x, y, dx=1, dy=1)
        yy = self.evaluator.evaluate(time, x, y, dy=2)
        return jnp.stack((jnp.stack((-xy, -yy), axis=-1),
                          jnp.stack((xx, xy), axis=-1)), axis=-2)


@lru_cache(maxsize=16)
def _bind(potential: Potential, execution: ExecutionOptions, dimension: int,
          velocity_scale: float, electric_scale: float, frequency: float) -> JaxDynamics:
    """Bounded reuse of compiled function identities and immutable field data."""
    return JaxDynamics(potential, execution, dimension, velocity_scale, electric_scale, frequency)


def bind_dynamics(dynamics: DynamicalSystem, execution: ExecutionOptions) -> JaxDynamics:
    """Snapshot only known equations; never discard a subclass's overrides."""
    if type(dynamics) is GuidingCenterDynamics:
        return _bind(dynamics.effective_potential, execution, 2, 0., 1., 0.)
    if type(dynamics) is FullCyclotronDynamics:
        return _bind(dynamics.potential, execution, 4, dynamics.velocity_scale,
                     dynamics.electric_scale, dynamics.larmor_frequency)
    raise TypeError("JAX execution requires built-in GuidingCenterDynamics or FullCyclotronDynamics.")


__all__: list[str] = []
