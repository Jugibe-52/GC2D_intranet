"""Optional JAX evaluation of the canonical host splines, without refitting."""

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ._jax_kernels import evaluate_splines

if TYPE_CHECKING:
	from .potential import Potential


def _input_device(values: tuple[Any, ...]) -> Any:
	"""Select one eager device, rejecting incompatible committed placements."""
	arrays = [value for value in values if isinstance(value, jax.Array)]
	committed = [value for value in arrays if value.committed]
	devices = set().union(*(value.devices() for value in committed))
	if len(devices) > 1:
		raise ValueError("JAX evaluation requires input arrays on one compatible device.")
	if devices:
		return next(iter(devices))
	if arrays:
		devices = arrays[0].devices()
		if len(devices) != 1:
			raise ValueError("JAX evaluation requires input arrays on one compatible device.")
		return next(iter(devices))
	return None


def evaluate_jax(
	potential: "Potential", t: Any, x: Any, y: Any, dx: int, dy: int, dt: int,
) -> Any:
	"""Evaluate eager arrays or tracers, keeping all trace-local buffers local.

	Only concrete device constants enter the potential's cache. During tracing,
	host coefficients become constants of the compiled function and inherit its
	placement, so a CPU trace never fixes the device of a later GPU execution.
	"""
	if not jax.config.read("jax_enable_x64"):
		raise RuntimeError(
			"JAX potential evaluation requires float64. Enable it before use: "
			"jax.config.update('jax_enable_x64', True)"
		)
	leaves = tuple(jax.tree.leaves((t, x, y)))
	traced = any(isinstance(value, jax.core.Tracer) for value in leaves)
	device = None if traced else _input_device(leaves)
	if traced:
		time, x, y = (jnp.asarray(value, dtype=jnp.float64) for value in (t, x, y))
	else:
		time, x, y = (
			jnp.asarray(jax.device_put(value, device), dtype=jnp.float64)
			for value in (t, x, y)
		)
	# Closed-over arrays may enter without tracer arguments, while conversions
	# above still trace inside an outer zero-argument JIT.
	traced = traced or any(isinstance(value, jax.core.Tracer) for value in (time, x, y))
	if x.shape != y.shape:
		raise ValueError("`x` and `y` must have the same shape.")
	if not traced and device in potential._jax_buffers:
		buffers = potential._jax_buffers[device]
	else:
		knots_x, knots_y, coefficients = potential._spline_data
		host_data = (
			knots_x, knots_y, coefficients, potential.frequencies,
			np.array([potential.grid.xmin, potential.grid.ymin]), np.asarray(potential.grid.period),
		)
		if traced:
			buffers = tuple(jnp.asarray(value) for value in host_data)
		else:
			# Even an outer trace with constant-only inputs must never leak a
			# tracer into this persistent cache.
			with jax.ensure_compile_time_eval():
				buffers = tuple(jax.device_put(value, device) for value in host_data)
			potential._jax_buffers[device] = buffers
	return evaluate_splines(time, x, y, *buffers,
		degree=potential.interpolation_order, dx=dx, dy=dy, dt=dt)
