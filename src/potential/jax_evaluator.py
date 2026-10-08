"""Optional, explicitly placed JAX evaluation of an existing potential.

Importing this module does not import JAX. The evaluator is a device snapshot,
not a replacement for Potential in the existing NumPy integration contracts.
"""

from typing import Any, Literal

import numpy as np

from contracts.execution_options import ExecutionOptions

from ._evaluation import PotentialEvaluator


class JaxPotentialEvaluator(PotentialEvaluator):
    """Evaluate existing splines in float64 on one CPU or GPU device.

    Enable ``jax_enable_x64`` before construction. Results remain JAX arrays;
    callers choose when to copy them to NumPy. Prepare gyroaveraging with
    ``potential.gyroaverage(rho)`` before creating this evaluator. Derivative
    orders are static integers, including when evaluation is enclosed in jit.
    """

    def __init__(
        self, potential: Any, *, device: Literal["cpu", "gpu"] = "cpu",
        device_index: int = 0,
    ) -> None:
        """Copy SciPy knots, coefficients, and sampled fields to one device."""
        super().__init__(potential)
        execution = ExecutionOptions(backend="jax", device=device, device_index=device_index)
        try:
            import jax
        except ImportError as exc:
            raise ImportError(
                "JAX potential evaluation requires the optional 'jax' extra. "
                "Install with: python -m pip install -e '.[jax]'"
            ) from exc
        self._jax = jax
        self.check_ready()
        try:
            devices = jax.devices(device)
        except RuntimeError as exc:
            raise RuntimeError(f"Requested JAX device '{device}' is unavailable.") from exc
        if device_index >= len(devices):
            raise ValueError(f"JAX device index {device_index} is unavailable for '{device}'.")
        self.device = devices[execution.device_index]
        import jax.numpy as jnp

        self.namespace = jnp
        # Construction can occur inside an outer JIT trace. Eagerly
        # materialize constant buffers so the persistent cache never captures
        # tracers from that temporary JIT trace.
        with jax.ensure_compile_time_eval():
            data = self.prepared.spline_data
            self._knots_x = jax.device_put(data.knots_x, self.device)
            self._knots_y = jax.device_put(data.knots_y, self.device)
            self._coefficients = jax.device_put(data.coefficients, self.device)
            self._frequencies = jax.device_put(self.prepared.frequencies, self.device)
            self._origin = jax.device_put(np.array([self.grid.xmin, self.grid.ymin]), self.device)
            self._period = jax.device_put(np.asarray(self.grid.period), self.device)
            self._samples = jax.device_put(self.prepared.samples, self.device)
        from ._jax_kernels import evaluate_samples, evaluate_splines

        self._evaluate_splines = evaluate_splines
        self._evaluate_samples = evaluate_samples

    def check_ready(self) -> None:
        """Reject implicit float32 truncation without changing global settings."""
        if not self._jax.config.read("jax_enable_x64"):
            raise RuntimeError(
                "JAX potential evaluation requires float64. Enable it before use: "
                "jax.config.update('jax_enable_x64', True)"
            )

    def asarray(self, value: Any) -> Any:
        """Place an input on the selected device and normalize its dtype."""
        import jax.numpy as jnp

        # Repeated native calls must not dispatch copies or casts for arrays
        # already resident on this device. Traced inputs use device_put below.
        if (isinstance(value, self._jax.Array) and not isinstance(value, self._jax.core.Tracer)
                and value.dtype == np.dtype("float64")
                and value.devices() == {self.device}):
            return value
        return jnp.asarray(self._jax.device_put(value, self.device), dtype=jnp.float64)

    def _evaluate(self, time: Any, x: Any, y: Any, dx: int, dy: int, dt: int) -> Any:
        """Execute the compiled spatial and shared temporal calculation."""
        return self._evaluate_splines(
            time, x, y, self._knots_x, self._knots_y,
            self._coefficients, self._frequencies, self._origin, self._period,
            degree=self.interpolation_order, dx=dx, dy=dy, dt=dt,
        )

    def _evaluate_grid(self, time: Any, dt: int) -> Any:
        """Reconstruct grid samples using the same common temporal formula."""
        return self._evaluate_samples(time, self._samples, self._frequencies, dt=dt)


__all__ = ["JaxPotentialEvaluator"]
