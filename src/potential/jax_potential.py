"""Explicit JAX evaluation of the inherited canonical SciPy splines."""

from typing import Any

import numpy as np

from .potential import Potential
from ._evaluation import validate_derivatives


def _input_device(values: tuple[Any, ...]) -> Any:
	"""Select one eager device, rejecting incompatible committed placements."""
	import jax

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


class JaxPotential(Potential):
	"""A potential whose pointwise values and electric fields are JAX arrays.

	Construction, loading, gyroaveraging and grid reconstruction are inherited.
	JAX remains optional until evaluation; enable float64 before the first call.
	"""

	def evaluate(
		self, t: Any, x: Any, y: Any, *, dx: int = 0, dy: int = 0, dt: int = 0,
	) -> Any:
		"""Evaluate paired coordinates on JAX, including scalar and NumPy inputs.

		Only concrete device constants enter the persistent cache. During tracing,
		host coefficients become constants of the compiled function and inherit
		its placement. Derivative orders must be static under JIT.
		"""
		validate_derivatives(self.interpolation_order, dx, dy, dt)
		try:
			import jax
			import jax.numpy as jnp
		except ImportError as exc:
			raise ImportError(
				"JAX potential evaluation requires the optional 'jax' extra. "
				"Install with: python -m pip install -e '.[jax]'"
			) from exc
		from ._jax_kernels import evaluate_splines

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
		if not traced:
			# Scalar/NumPy inputs follow the current default device. Cache by the
			# concrete placement so a later default-device context can change it.
			device = _input_device((time, x, y))
		if x.shape != y.shape:
			raise ValueError("`x` and `y` must have the same shape.")
		if not traced and device in self._jax_buffers:
			buffers = self._jax_buffers[device]
		else:
			knots_x, knots_y, coefficients = self._spline_data
			host_data = (
				knots_x, knots_y, coefficients, self.frequencies,
				np.array([self.grid.xmin, self.grid.ymin]), np.asarray(self.grid.period),
			)
			if traced:
				buffers = tuple(jnp.asarray(value) for value in host_data)
			else:
				# Even an outer trace with constant-only inputs must never leak a
				# tracer into this persistent cache.
				with jax.ensure_compile_time_eval():
					buffers = tuple(jax.device_put(value, device) for value in host_data)
				self._jax_buffers[device] = buffers
		return evaluate_splines(time, x, y, *buffers,
			degree=self.interpolation_order, dx=int(dx), dy=int(dy), dt=int(dt))


__all__ = ["JaxPotential"]
