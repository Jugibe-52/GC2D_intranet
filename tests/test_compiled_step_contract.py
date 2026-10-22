"""Prepared device maps are independent of the coordinator's method vocabulary."""

from formulations.layout import GCStateLayout

from dataclasses import dataclass
import importlib.util
import unittest
from unittest.mock import patch
from execution import Execution
from typing import Any, ClassVar

import numpy as np

from execution.options import ExecutionOptions
from problem import InitialValueProblem
from request import SimulationRequest
from dynamics.gc import GuidingCenterDynamics

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
		self.problem = InitialValueProblem(GuidingCenterDynamics(potential), np.array(np.array([.2, .3]), dtype=float, copy=True), layout=GCStateLayout())
		self.request = SimulationRequest((0., .2), .1, np.array([0., .03, .1, .2]))
		self.execution = ExecutionOptions(backend="jax")

	def prepare_kernel(self, run):
		from execution._jax import prepare_dynamics, resolve_device
		from methods._jax_dispatch import prepare_step
		return prepare_step(run, prepare_dynamics(run.problem.dynamics), resolve_device(self.execution))

	def test_coordinator_accepts_an_independent_prepared_map(self) -> None:
		import jax.numpy as jnp

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
		with patch.object(RK4, "new_run", return_value=run):
			with patch("methods._jax_dispatch.prepare_step", return_value=ConstantDrift(self.jax.devices("cpu")[0])):
				data = Execution().run(self.problem, RK4(), self.request, options=self.execution)
		expected = self.problem.initial_state[:, None] + np.array([2., -1.])[:, None] * data.t
		np.testing.assert_allclose(data.states, expected, rtol=0., atol=1e-15)
		self.assertEqual(data.diagnostics["step_count"], 2)
		self.assertEqual(data.diagnostics["output_interpolation_count"], 1)
		self.assertEqual(run._status, "finished")

	def test_fixed_and_hybrid_runs_finalize_the_allocated_result_once(self) -> None:
		from methods.adaptive.scipy import Radau
		from states import States
		for method in (RK4(track_energy=True), Radau(track_energy=True)):
			with self.subTest(method=type(method).__name__):
				recorders, finalized, runs = [], [], []
				prepare, finalize, new_run = States.prepare, States.finalize, method.new_run
				def allocate(*args, **kwargs):
					recorder = prepare(*args, **kwargs)
					recorders.append(recorder)
					return recorder
				def finish(recorder, *args, **kwargs):
					self.assertFalse(recorder.complete)
					self.assertEqual(runs[0]._status, "running")
					finalized.append(recorder)
					return finalize(recorder, *args, **kwargs)
				def prepare_run(*args, **kwargs):
					run = new_run(*args, **kwargs)
					runs.append(run)
					return run
				with patch.object(States, "prepare", side_effect=allocate), patch.object(States, "finalize", finish):
					with patch.object(type(method), "new_run", side_effect=prepare_run):
						result = Execution().run(self.problem, method, self.request, options=self.execution)
				self.assertEqual(len(recorders), 1)
				self.assertEqual(len(finalized), 1)
				self.assertIs(recorders[0], result)
				self.assertIs(finalized[0], result)
				self.assertEqual(runs[0]._status, "finished")
				self.assertEqual(method._status, "configuration")

	def test_prepared_controls_do_not_follow_mutable_method_options(self) -> None:
		from methods._jax_dispatch import prepare_step

		run = GaussLegendre4().new_run(self.problem, self.request)
		kernel = self.prepare_kernel(run)
		state, statistics, converged = kernel(0., run.initial_state, .01)
		run.newton_absolute_tolerance = 1e3
		run.newton_relative_tolerance = 1e3
		again, repeated, _ = kernel(0., run.initial_state, .01)
		np.testing.assert_array_equal(again, state)
		np.testing.assert_array_equal(repeated["nonlinear_tolerances"], statistics["nonlinear_tolerances"])
		self.assertTrue(converged)

	def test_runs_reuse_prepared_dynamics_and_bounded_compilation(self) -> None:
		from execution._jax import prepare_dynamics
		from execution._jax_fixed import _compiled_integrator
		from methods._jax_dispatch import PreparedJaxStep, prepare_step

		_compiled_integrator.cache_clear()
		self.addCleanup(_compiled_integrator.cache_clear)
		run = RK4().new_run(self.problem, self.request)
		kernel = self.prepare_kernel(run)
		self.assertIsInstance(kernel, PreparedJaxStep)
		assert isinstance(kernel, PreparedJaxStep)
		self.assertIs(kernel.dynamics, prepare_dynamics(self.problem.dynamics))
		self.assertIsNot(kernel.dynamics, self.problem.dynamics)
		compiled = _compiled_integrator(kernel)
		repeated = RK4().new_run(self.problem, self.request)
		self.assertIs(_compiled_integrator(self.prepare_kernel(repeated)), compiled)

		# New physical objects must not retain an unbounded list of JIT closures.
		for _ in range(17):
			problem = InitialValueProblem(GuidingCenterDynamics(self.problem.dynamics.potential), self.problem.initial_state, layout=GCStateLayout())
			run = RK4().new_run(problem, self.request)
			_compiled_integrator(self.prepare_kernel(run))
		self.assertEqual(_compiled_integrator.cache_info().currsize, 16)
		self.assertIsNot(_compiled_integrator(kernel), compiled)

	def test_device_index_is_validated_before_compilation(self) -> None:
		from methods._jax_dispatch import prepare_step

		execution = ExecutionOptions(backend="jax", device_index=len(self.jax.devices("cpu")))
		run = RK4().new_run(self.problem, self.request)
		with self.assertRaisesRegex(ValueError, "device index.*unavailable"):
			Execution().run(self.problem, RK4(), self.request, options=execution)

	def test_python_newton_callbacks_are_rejected_before_compilation(self) -> None:
		events = []
		run = GaussLegendre4(newton_observer=events.append).new_run(
			self.problem, self.request)
		with patch.object(GaussLegendre4, "new_run", return_value=run):
			with self.assertRaisesRegex(NotImplementedError, "newton_observer"):
				Execution().run(self.problem, GaussLegendre4(), self.request, options=self.execution)
			self.assertEqual(run._status, "finished")
			with self.assertRaisesRegex(TypeError, "fresh"):
				Execution().run(self.problem, GaussLegendre4(), self.request, options=self.execution)
		self.assertEqual(events, [])


if __name__ == "__main__":
	unittest.main()
