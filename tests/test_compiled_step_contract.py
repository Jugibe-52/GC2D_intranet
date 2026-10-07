"""Prepared device maps are independent of the coordinator's method vocabulary."""

from dataclasses import dataclass
import importlib.util
import unittest
from typing import Any, ClassVar

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from dynamics.gc import GuidingCenterDynamics
from initial_conditions.gc import GCInitialConfiguration
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.rk4 import RK4
from potential.potential import Potential


@unittest.skipUnless(importlib.util.find_spec("jax"), "Optional JAX is not installed")
class CompiledStepContractTests(unittest.TestCase):
	"""Exercise the backend boundary with an independently supplied device kernel."""

	def setUp(self) -> None:
		import jax
		self.jax = jax
		previous = jax.config.read("jax_enable_x64")
		jax.config.update("jax_enable_x64", True)
		self.addCleanup(jax.config.update, "jax_enable_x64", previous)
		potential = Potential.random(A=.1, M=2, nx=12, ny=12, seed=19)
		self.problem = InitialValueProblem(GuidingCenterDynamics(potential),
			GCInitialConfiguration(np.array([.2, .3])))
		self.request = SimulationRequest((0., .2), .1, np.array([0., .03, .1, .2]))
		self.execution = ExecutionOptions(backend="jax")

	def test_coordinator_accepts_an_independent_prepared_map(self) -> None:
		import jax.numpy as jnp
		from integration.jax_fixed import integrate_fixed

		@dataclass(frozen=True)
		class ConstantDrift:
			"""A device map unknown to every built-in method adapter."""

			device: Any
			physical_dimension: ClassVar[int] = 2
			copies: ClassVar[int] = 1
			track_energy: ClassVar[bool] = False
			finite_difference: ClassVar[bool] = False

			def __call__(self, time: Any, state: Any, step: Any) -> tuple[Any, dict[str, Any], Any]:
				return state + step * jnp.array([2., -1.]), {}, jnp.array(True)

			def energy(self, times: Any, physical: Any) -> Any:
				raise AssertionError("An untracked map must not evaluate energy.")

		run = RK4().new_run(self.problem, self.request)
		data = integrate_fixed(run, self.execution, ConstantDrift(self.jax.devices("cpu")[0]))
		expected = self.problem.initial_state[:, None] + np.array([2., -1.])[:, None] * data.t
		np.testing.assert_allclose(data.states, expected, rtol=0., atol=1e-15)
		self.assertEqual(data.diagnostics["step_count"], 2)
		self.assertEqual(data.diagnostics["output_interpolation_count"], 1)
		self.assertEqual(run._status, "finished")

	def test_prepared_controls_do_not_follow_mutable_method_options(self) -> None:
		from methods._jax_dispatch import prepare_step

		run = GaussLegendre4().new_run(self.problem, self.request, execution=self.execution)
		kernel = prepare_step(run, self.execution)
		state, statistics, converged = kernel(0., run.initial_state, .01)
		run.newton_absolute_tolerance = 1e3
		run.newton_relative_tolerance = 1e3
		again, repeated, _ = kernel(0., run.initial_state, .01)
		np.testing.assert_array_equal(again, state)
		np.testing.assert_array_equal(repeated["nonlinear_tolerances"], statistics["nonlinear_tolerances"])
		self.assertTrue(converged)

	def test_python_newton_callbacks_are_rejected_before_compilation(self) -> None:
		events = []
		run = GaussLegendre4(newton_observer=events.append).new_run(
			self.problem, self.request, execution=self.execution)
		with self.assertRaisesRegex(NotImplementedError, "newton_observer"):
			run._integrate_jax(self.execution)
		self.assertEqual(run._status, "finished")
		with self.assertRaisesRegex(RuntimeError, "fresh"):
			run._integrate_jax(self.execution)
		self.assertEqual(events, [])


if __name__ == "__main__":
	unittest.main()
