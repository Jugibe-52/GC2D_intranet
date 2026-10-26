"""Prepared device maps are independent of the coordinator's method vocabulary."""

from dynamics.gc import GuidingCenterDynamics

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

from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.rk4 import RK4
from solvers.nonlinear import NewtonSolver
from diagnostics.methods.energy import EnergyObserver
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
		self.problem = InitialValueProblem(GuidingCenterDynamics(potential), np.array(np.array([.2, .3]), dtype=float, copy=True), layout=GuidingCenterDynamics.coordinates)
		self.request = SimulationRequest((0., .2),  np.array([0., .03, .1, .2]))
		self.execution = ExecutionOptions(backend="jax")

	def prepare_run(self, method, problem=None):
		from execution._jax import prepare_dynamics
		problem = self.problem if problem is None else problem
		dynamics = prepare_dynamics(problem.dynamics)
		return method.new_run(dynamics.vector_field, problem.initial_conditions, self.request,
			jacobian=dynamics.particle_vector_field_jacobians, jacobian_structure='particle')

	def prepare_kernel(self, run):
		from execution._jax import resolve_device
		from execution._jax_step import prepare_step
		return prepare_step(run, resolve_device(self.execution))

	def test_coordinator_accepts_an_independent_prepared_map(self) -> None:
		import jax.numpy as jnp

		@dataclass(frozen=True)
		class ConstantDrift:
			"""A device map unknown to every built-in method adapter."""

			device: Any
			physical_dimension: ClassVar[int] = 2
			copies: ClassVar[int] = 1
			finite_difference: ClassVar[bool] = False

			def __call__(self, time: Any, state: Any, step: Any) -> tuple[Any, Any, dict[str, Any], Any]:
				return state + step * jnp.array([2., -1.]), None, {}, jnp.array(True)

		run = self.prepare_run(RK4(step=0.1))
		with patch.object(RK4, "new_run", return_value=run):
			with patch("execution._jax_step.prepare_step", return_value=ConstantDrift(self.jax.devices("cpu")[0])):
				data = Execution().run(self.problem.with_request(self.request), RK4(step=0.1), options=self.execution)
		expected = self.problem.initial_state[:, None] + np.array([2., -1.])[:, None] * data.t
		np.testing.assert_allclose(data.states.T, expected, rtol=0., atol=1e-15)
		self.assertEqual(data.diagnostics["step_count"], 2)
		self.assertEqual(data.diagnostics["output_interpolation_count"], 1)
		self.assertEqual(run._status, "finished")

	def test_fixed_and_hybrid_runs_finalize_the_allocated_result_once(self) -> None:
		from methods.adaptive.scipy import Radau
		from states import States
		for method in (RK4(), Radau()):
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
						result = Execution().run(self.problem.with_request(self.request), method, options=self.execution, observers=(EnergyObserver(),))
				self.assertEqual(len(recorders), 1)
				self.assertEqual(len(finalized), 1)
				self.assertIs(recorders[0], result)
				self.assertIs(finalized[0], result)
				self.assertEqual(runs[0]._status, "finished")
				self.assertEqual(method._status, "configuration")

	def test_compiled_step_preparation_failure_consumes_only_the_fresh_run(self) -> None:
		from states.history import States

		method = RK4()
		run = self.prepare_run(method)
		def fail_preparation(*args, **kwargs):
			self.assertEqual(run._status, "running")
			raise RuntimeError("Compiled step unavailable")
		with patch.object(RK4, "new_run", return_value=run):
			with patch("execution._jax_step.prepare_step", side_effect=fail_preparation) as compile_step:
				with patch.object(States, "prepare") as allocate:
					with self.assertRaisesRegex(RuntimeError, "Compiled step unavailable"):
						Execution().run(self.problem.with_request(self.request), method, options=self.execution)
					allocate.assert_not_called()
				compile_step.assert_called_once()
			self.assertEqual(run._status, "finished")
			with self.assertRaisesRegex(TypeError, "fresh"):
				Execution().run(self.problem.with_request(self.request), method, options=self.execution)
		self.assertEqual(method._status, "configuration")
		self.assertTrue(Execution().run(self.problem.with_request(self.request), method, options=self.execution).complete)

	def test_prepared_controls_do_not_follow_mutable_method_options(self) -> None:
		from execution._jax_step import prepare_step

		run = self.prepare_run(GaussLegendre4(step=0.1))
		kernel = self.prepare_kernel(run)
		state, _, statistics, converged = kernel(0., run.initial_state, .01)
		run.solver = NewtonSolver(1e3, 1e3)
		again, _, repeated, _ = kernel(0., run.initial_state, .01)
		np.testing.assert_array_equal(again, state)
		np.testing.assert_array_equal(repeated["nonlinear_tolerances"], statistics["nonlinear_tolerances"])
		self.assertTrue(converged)

	def test_runs_reuse_prepared_dynamics_and_bounded_compilation(self) -> None:
		from execution._jax import prepare_dynamics
		from execution._jax_fixed import _compiled_integrator
		from execution._jax_step import PreparedStep, prepare_step

		_compiled_integrator.cache_clear()
		self.addCleanup(_compiled_integrator.cache_clear)
		run = self.prepare_run(RK4(step=0.1))
		kernel = self.prepare_kernel(run)
		self.assertIsInstance(kernel, PreparedStep)
		assert isinstance(kernel, PreparedStep)
		self.assertEqual(kernel.f, prepare_dynamics(self.problem.dynamics).vector_field)
		self.assertFalse(hasattr(kernel, "dynamics"))
		compiled = _compiled_integrator(kernel, None, None)
		repeated = self.prepare_run(RK4(step=0.1))
		self.assertIs(_compiled_integrator(self.prepare_kernel(repeated), None, None), compiled)

		# New physical objects must not retain an unbounded list of JIT closures.
		for _ in range(17):
			problem = InitialValueProblem(GuidingCenterDynamics(self.problem.dynamics.potential), self.problem.initial_state, layout=GuidingCenterDynamics.coordinates)
			run = self.prepare_run(RK4(step=0.1), problem)
			_compiled_integrator(self.prepare_kernel(run), None, None)
		self.assertEqual(_compiled_integrator.cache_info().currsize, 16)
		self.assertIsNot(_compiled_integrator(kernel, None, None), compiled)

	def test_device_index_is_validated_before_compilation(self) -> None:
		from execution._jax_step import prepare_step

		execution = ExecutionOptions(backend="jax", device_index=len(self.jax.devices("cpu")))
		run = self.prepare_run(RK4(step=0.1))
		with self.assertRaisesRegex(ValueError, "device index.*unavailable"):
			Execution().run(self.problem.with_request(self.request), RK4(step=0.1), options=execution)

	def test_python_newton_callbacks_are_rejected_before_compilation(self) -> None:
		events = []
		run = self.prepare_run(GaussLegendre4(step=0.1, solver=NewtonSolver(1e-14, 1e-13, 20, iteration_observer=events.append)))
		with patch.object(GaussLegendre4, "new_run", return_value=run):
			with self.assertRaisesRegex(NotImplementedError, "iteration callbacks"):
				Execution().run(self.problem.with_request(self.request), GaussLegendre4(step=0.1), options=self.execution)
			self.assertEqual(run._status, "finished")
			with self.assertRaisesRegex(TypeError, "fresh"):
				Execution().run(self.problem.with_request(self.request), GaussLegendre4(step=0.1), options=self.execution)
		self.assertEqual(events, [])

	def test_direct_preparation_rejects_host_observation_before_copying(self) -> None:
		from execution._jax_step import prepare_step

		events = []
		for method, callbacks in (
			(RK4(), (events.append,)),
			(RK4(progress=True), ()),
			(GaussLegendre4(solver=NewtonSolver(iteration_observer=events.append)), ()),
		):
			with self.subTest(method=type(method).__name__, callbacks=bool(callbacks), progress=method.progress):
				run = self.prepare_run(method)
				with patch('execution._jax_step.copy', side_effect=AssertionError('Unexpected preparation')):
					with self.assertRaisesRegex(NotImplementedError, 'step or iteration callbacks'):
						prepare_step(run, self.jax.devices('cpu')[0], callbacks=callbacks)
				self.assertEqual(run._status, 'ready')
		self.assertEqual(events, [])

	def test_direct_preparation_rejects_adaptive_and_overridden_methods(self) -> None:
		from execution._jax_step import prepare_step
		from methods.adaptive.scipy import DOP853

		class CustomRK4(RK4):
			pass

		for method in (DOP853(), CustomRK4()):
			with self.subTest(method=type(method).__name__):
				run = self.prepare_run(method)
				with self.assertRaisesRegex(TypeError, 'supported built-in method'):
					prepare_step(run, self.jax.devices('cpu')[0])


if __name__ == "__main__":
	unittest.main()
