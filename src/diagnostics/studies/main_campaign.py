"""Compact direct physical-energy histories and accepted-step projection data."""
from __future__ import annotations

from typing import Any
import numpy as np
from states.history import States
from contracts.states.history import DiagnosticValue


def compact_diagnostics(solution: States, stride: int) -> States:
	"""Keep direct H at saved physical states and reduce signed projection data.

	EnergyObserver evaluates the same physical Hamiltonian for every method.
	The reported variation is H(t_n, z_n) - H(t_0, z_0), not a conservation error
	for time-dependent fields. The analytic time derivative is never integrated.
	"""
	import jax
	import jax.numpy as jnp
	if type(stride) is not int or stride <= 0:
		raise ValueError('Recording stride must be a positive integer.')
	d: Any = solution.diagnostics
	count: int = int(d['step_count'])
	if count % stride or solution.t.size != count // stride + 1 or d['output_interpolation_count']:
		raise ValueError('Recording must coincide with every stride-th accepted step.')
	energy: np.ndarray = np.asarray(d['physical_hamiltonian'])
	derivative: np.ndarray = np.asarray(d['hamiltonian_time_derivative'])
	expected: tuple[int, int] = (solution.states.shape[1] // 2, solution.t.size)
	if (energy.shape != expected or derivative.shape != expected
			or not np.isfinite(energy).all() or not np.isfinite(derivative).all()):
		raise ValueError('Direct energy histories must contain finite values at every saved state.')
	with jax.experimental.enable_x64():
		added: dict[str, Any] = {}
		if 'projection_multipliers' in d:
			mu: Any = jnp.asarray(d['projection_multipliers']).reshape(count, 2, -1)
			norms: Any = jnp.max(jnp.abs(mu), axis=1)
			added.update(projection_multipliers_sampled=mu[stride-1::stride],
				projection_particle_norms=norms[stride-1::stride],
				projection_particle_block_max=norms.reshape(-1, stride, norms.shape[1]).max(axis=1))
		additions: Any = jax.device_get(added)
	# Endpoint observations are redundant with the sampled physical histories.
	# Also discard legacy quadrature fields rather than carrying them into new archives.
	drop: set[str] = {'energy_start', 'energy_end', 'energy_change', 'energy_time_derivative_start',
		'energy_time_derivative_end', 'projection_multipliers', 'hamiltonian',
		'energy_integral', 'energy_balance', 'energy_quadrature_difference', 'energy_quadrature'}
	kept: dict[str, DiagnosticValue] = {k: v for k, v in d.items() if k not in drop}
	kept.update(additions)
	kept.update(energy_drift=energy-energy[:, :1],
		diagnostic_backend='jax_float64', diagnostic_stride=stride,
		energy_evaluation='physical_hamiltonian_at_saved_states',
		projection_iteration_scope='joint_particle_batch')
	return solution.with_diagnostics(kept, replace=True)


__all__ = ['compact_diagnostics']
