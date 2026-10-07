"""Optional compiled execution for fixed methods; CPU imports stay lightweight."""

from typing import Generic, TypeVar

from contracts.execution_options import ExecutionOptions
from contracts.result import IntegrationData
from integration.core import IntegrationMethod


Detail = TypeVar("Detail")


class CompiledFixedMethod(IntegrationMethod[Detail], Generic[Detail]):
	"""Prepare a method-owned device map before entering the generic coordinator."""

	def _integrate_jax(self, execution: ExecutionOptions) -> IntegrationData:
		"""Bind only supported built-in equations without silently dropping hooks."""
		if self._status != "ready":
			raise RuntimeError("Integration requires a fresh method.new_run(problem, request).")
		try:
			if (self.step_observer is not None or self.progress
				or getattr(self, "newton_observer", None) is not None):
				raise NotImplementedError(
					"JAX fixed integration requires step_observer=None, "
					"newton_observer=None and progress=False; Python callbacks "
					"are supported by the SciPy execution path."
				)
			try:
				from methods._jax_dispatch import prepare_step
				from integration.jax_fixed import integrate_fixed
			except ImportError as exc:
				raise ImportError("JAX integration requires the optional 'jax' extra: pip install -e '.[jax]'.") from exc
			kernel = prepare_step(self, execution)
		except BaseException:
			# Preparation failures consume this run just like failures in its driver.
			self._status = "finished"
			raise
		return integrate_fixed(self, execution, kernel)


__all__: list[str] = []
