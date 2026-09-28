"""Numerical and execution contracts of device-resident classical RK4."""

import importlib.util
import subprocess
import sys
import unittest

import numpy as np

from contracts.execution import Execution
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from dynamics.gc import GuidingCenterDynamics
from dynamics.fc import FullCyclotronDynamics
from initial_conditions.gc import GCInitialConfiguration
from initial_conditions.fc import FCInitialConfiguration
from methods.classical.rk4 import RK4
from potential.grid import Grid
from potential.potential import Potential
from simulation.runner import simulate


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


def problem(fc=False):
    """Three independent particles, including coordinates outside the base cell."""
    potential = Potential.random(A=.1, M=3, nx=16, ny=16, seed=37)
    x, y = np.array([-.03, 2., 6.3]), np.array([6.27, .4, -.02])
    if fc:
        dynamics = FullCyclotronDynamics(potential, rho=.3, eta=-.4)
        initial = FCInitialConfiguration.from_components(x=x, y=y, vx=x*.1, vy=y*.05)
    else:
        dynamics = GuidingCenterDynamics(potential, rho=.3)
        initial = GCInitialConfiguration.from_components(x=x, y=y)
    return InitialValueProblem(dynamics, initial)


class OptionalExecutionTests(unittest.TestCase):
    def test_cpu_integration_does_not_import_jax(self):
        script = '''
import sys
from test_jax_rk4 import problem
class Block:
    def find_spec(self, fullname, *args):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ImportError("JAX deliberately unavailable")
sys.meta_path.insert(0, Block())
from contracts.execution import Execution
from contracts.request import SimulationRequest
from methods.classical.rk4 import RK4
from simulation.runner import simulate
p, r = problem(), SimulationRequest.uniform(t_span=(0., .01), max_step=.01)
simulate(p, RK4(), r, execution=Execution())
try:
    simulate(p, RK4(), r, execution=Execution(backend="jax"))
except ImportError as error:
    assert "optional" in str(error)
else:
    raise AssertionError("Missing dependency was ignored")
'''
        result = subprocess.run([sys.executable, "-c", script], cwd="tests",
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_and_explicit_cpu_execution_match(self):
        p, r = problem(), SimulationRequest.uniform(t_span=(0., .03), max_step=.01)
        a, b = simulate(p, RK4(), r), simulate(p, RK4(), r, execution=Execution())
        np.testing.assert_array_equal(a.states, b.states)
        with self.assertRaisesRegex(TypeError, "Execution"):
            simulate(p, RK4(), r, execution="jax")


@unittest.skipUnless(JAX_AVAILABLE, "Optional JAX dependency is not installed")
class JaxRK4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        cls.jax = jax
        cls.previous_x64 = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        cls.execution = Execution(backend="jax")

    @classmethod
    def tearDownClass(cls):
        cls.jax.config.update("jax_enable_x64", cls.previous_x64)

    def assert_equivalent(self, p, request, tracking, execution=None):
        method = RK4(track_energy=tracking)
        reference = simulate(p, method, request)
        result = simulate(p, method, request, execution=execution or self.execution)
        np.testing.assert_allclose(result.states, reference.states, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(result.t, reference.t)
        np.testing.assert_array_equal(result.states[:, 0], p.initial_state)
        self.assertFalse(result.states.flags.writeable)
        self.assertFalse(hasattr(method, "initial_state"))
        for key in reference.diagnostics:
            if key.startswith("execution_"):
                continue
            actual, expected = result.diagnostics[key], reference.diagnostics[key]
            if isinstance(expected, (np.ndarray, float)):
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12, err_msg=key)
            else:
                self.assertEqual(actual, expected, key)
        return result

    def test_gc_fc_energy_and_irregular_samples(self):
        request = SimulationRequest(t_span=(.3, .5), max_step=.06,
                                    output_times=np.array([.3, .3001, .35, .371, .41, .49, .5]))
        for fc in (False, True):
            for tracking in (False, True):
                with self.subTest(fc=fc, tracking=tracking):
                    self.assert_equivalent(problem(fc), request, tracking)

    def test_sparse_sampling_shadow_independence_and_particle_independence(self):
        p = problem()
        nodes = np.linspace(0., .5, 6)
        dense = np.unique(np.concatenate((nodes, [.02, .08, .21, .22, .49])))
        sampled = simulate(p, RK4(), SimulationRequest((0., .5), .1, dense), execution=self.execution)
        sparse = self.assert_equivalent(p, SimulationRequest((0., .5), .1, nodes[[0, 2, 5]]), False)
        np.testing.assert_allclose(sampled.states[:, np.searchsorted(dense, sparse.t)], sparse.states,
                                   rtol=1e-14, atol=1e-14)
        x, y = p.initial_configuration.layout.split(p.initial_state)
        for i in range(3):
            one = InitialValueProblem(p.dynamics, GCInitialConfiguration.from_components(x=x[i:i+1], y=y[i:i+1]))
            solo = simulate(one, RK4(), SimulationRequest((0., .5), .1, sparse.t))
            np.testing.assert_allclose(sparse.states[[i, i+3]], solo.states, rtol=1e-13, atol=1e-13)

    def test_fourth_order_against_an_exact_time_dependent_shear(self):
        grid = Grid.periodic(24, 16)
        mode = .05 * np.sin(grid.x[:, None]) * np.ones((1, grid.ny))
        potential = Potential(grid, modes=mode[None], frequencies=np.array([.7]))
        p = InitialValueProblem(GuidingCenterDynamics(potential),
                                GCInitialConfiguration.from_components(x=np.array([.73]), y=np.array([.4])))
        # Phi is independent of y: x stays fixed and y' is a cosine in time.
        amplitude = potential.evaluate(0., np.array([.73]), np.array([.4]), dx=1)[0]
        exact = .4 + amplitude * np.sin(2*np.pi*.7) / (2*np.pi*.7)
        errors = []
        for h in (.2, .1, .05):
            result = simulate(p, RK4(), SimulationRequest((0., 1.), h, np.array([0., 1.])), execution=self.execution)
            errors.append(abs(result.states[1, -1] - exact))
        self.assertTrue(all(14 < a/b < 18 for a, b in zip(errors[:-1], errors[1:])), errors)

    def test_precision_and_unsupported_capabilities_fail_explicitly(self):
        p, r = problem(), SimulationRequest.uniform(t_span=(0., .02), max_step=.01)
        simulate(p, RK4(), r, execution=self.execution)
        try:
            self.jax.config.update("jax_enable_x64", False)
            with self.assertRaisesRegex(RuntimeError, "float64"):
                simulate(p, RK4(), r, execution=self.execution)
        finally:
            self.jax.config.update("jax_enable_x64", True)
        for method in (RK4(progress=True), RK4(step_observer=lambda event: None)):
            with self.assertRaisesRegex(NotImplementedError, "callbacks"):
                simulate(p, method, r, execution=self.execution)
        class CustomDynamics(GuidingCenterDynamics):
            def vector_field(self, time, state):
                return np.zeros_like(state)
        custom = InitialValueProblem(CustomDynamics(p.dynamics.potential), p.initial_configuration)
        with self.assertRaisesRegex(TypeError, "built-in"):
            simulate(custom, RK4(), r, execution=self.execution)

    def test_near_endpoints_and_unsaved_nonfinite_states(self):
        p = problem()
        times = np.array([0., 1e-16, .1-1e-16, .1, .1+1e-16, .2])
        self.assert_equivalent(p, SimulationRequest((0., .2), .1, times), True)
        # A finite initial FC state can still overflow during unsaved steps.
        fc = FullCyclotronDynamics(p.dynamics.potential, rho=1., eta=1e-100)
        initial = FCInitialConfiguration.from_components(
            x=np.array([0.]), y=np.array([0.]), vx=np.array([1e200]), vy=np.array([1e200]),
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            simulate(InitialValueProblem(fc, initial), RK4(),
                     SimulationRequest((0., 1.), .1, np.array([0., 1.])), execution=self.execution)

    def test_gpu_equivalence_or_explicit_unavailability(self):
        p, r = problem(), SimulationRequest.uniform(t_span=(0., .02), max_step=.01)
        execution = Execution(backend="jax", device="gpu")
        try:
            devices = self.jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                simulate(p, RK4(), r, execution=execution)
            self.skipTest("No JAX GPU is available; explicit failure was verified")
        self.assert_equivalent(p, r, True, execution)


if __name__ == "__main__":
    unittest.main()
