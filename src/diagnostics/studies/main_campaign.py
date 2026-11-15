"""Compact campaign observations with JAX reductions on accepted step data."""
from __future__ import annotations

from typing import Any
import numpy as np
from states.history import States
from contracts.states.history import DiagnosticValue


def compact_diagnostics(solution: States, stride: int) -> States:
	"""Retain aligned samples, Simpson energy balance and signed projection data.

	No trajectories are integrated here. Simpson and trapezoid use the analytic
	partial time derivative at every accepted endpoint. Their difference is a
	quadrature sensitivity diagnostic, not a certified error bound.
	"""
	import jax
	import jax.numpy as jnp
	if stride < 2 or stride % 2:
		raise ValueError('Simpson blocks require a positive even stride.')
	d: Any = solution.diagnostics
	count: int = int(d['step_count'])
	if count % stride or solution.t.size != count // stride + 1 or d['output_interpolation_count']:
		raise ValueError('Recording must coincide with every stride-th accepted step.')
	sizes: np.ndarray = np.asarray(d['step_sizes'])
	if not np.allclose(sizes, sizes[0], rtol=1e-10, atol=1e-14):
		raise ValueError('Composite Simpson requires uniform accepted steps.')
	with jax.experimental.enable_x64():
		rates: Any = jnp.concatenate((jnp.asarray(d['energy_time_derivative_start'])[:1],
			jnp.asarray(d['energy_time_derivative_end'])), axis=0)
		indices: Any = jnp.arange(count // stride)[:, None] * stride + jnp.arange(stride + 1)
		weights: Any = jnp.where(jnp.arange(stride + 1) % 2, 4., 2.).at[0].set(1.).at[-1].set(1.)
		increments: Any = jnp.sum(rates[indices] * weights[None, :, None], axis=1) * sizes[0] / 3
		trap: Any = ((rates[:-1] + rates[1:]) * sizes[0] / 2).reshape(-1, stride, rates.shape[1]).sum(axis=1)
		zero: Any = jnp.zeros((1, rates.shape[1]), dtype=rates.dtype)
		integral: Any = jnp.concatenate((zero, jnp.cumsum(increments, axis=0)), axis=0).T
		difference: Any = jnp.concatenate((zero, jnp.cumsum(increments - trap, axis=0)), axis=0).T
		energy: Any = jnp.asarray(d['physical_hamiltonian'])
		added: dict[str, Any] = dict(energy_integral=integral, energy_balance=energy-energy[:, :1]-integral,
			energy_quadrature_difference=difference)
		if 'projection_multipliers' in d:
			mu: Any = jnp.asarray(d['projection_multipliers']).reshape(count, 2, -1)
			norms: Any = jnp.max(jnp.abs(mu), axis=1)
			added.update(projection_multipliers_sampled=mu[stride-1::stride],
				projection_particle_norms=norms[stride-1::stride],
				projection_particle_block_max=norms.reshape(-1, stride, norms.shape[1]).max(axis=1))
		additions: Any = jax.device_get(added)
	# Endpoint energy arrays and raw multipliers are consumed above. Keep scalar
	# per-step nonlinear work, but avoid repeating H under multiple aliases.
	drop: set[str] = {'energy_start', 'energy_end', 'energy_change', 'energy_time_derivative_start',
		'energy_time_derivative_end', 'projection_multipliers', 'hamiltonian', 'energy_drift'}
	kept: dict[str, DiagnosticValue] = {k: v for k, v in d.items() if k not in drop}
	kept.update(additions)
	kept.update(diagnostic_backend='jax_float64', diagnostic_stride=stride,
		energy_quadrature='composite_simpson_accepted_endpoints',
		projection_iteration_scope='joint_particle_batch')
	return solution.with_diagnostics(kept, replace=True)


__all__ = ['compact_diagnostics']
