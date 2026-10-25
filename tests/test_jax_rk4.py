"""Numerical and execution contracts of device-resident classical RK4."""

from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics

import importlib.util
import subprocess
import sys
import unittest

import numpy as np

from execution.options import ExecutionOptions
from problem import InitialValueProblem
from request import SimulationRequest

from methods.classical.rk4 import RK4
from potential.grid import Grid
from potential.potential import Potential
from simulation.runner import simulate
from diagnostics.methods.energy import EnergyObserver


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


def problem(fc=False):
    """Three independent particles, including coordinates outside the base cell."""
    potential = Potential.random(A=.1, M=3, nx=16, ny=16, seed=37)
    x, y = np.array([-.03, 2., 6.3]), np.array([6.27, .4, -.02])
    if fc:
        dynamics = FullCyclotronDynamics(potential, rho=.3, eta=-.4)
        initial = FullCyclotronDynamics.coordinates.pack_components(x, y, x*.1, y*.05)
    else:
        dynamics = GuidingCenterDynamics(potential, rho=.3)
        initial = GuidingCenterDynamics.coordinates.pack_components(x, y)
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
from execution.options import ExecutionOptions
from request import SimulationRequest
from methods.classical.rk4 import RK4
from simulation.runner import simulate
from diagnostics.methods.energy import EnergyObserver
p, r = problem(), SimulationRequest.uniform(t_span=(0., .01))
simulate(p, RK4(), r, options=ExecutionOptions())
assert "jax" not in sys.modules
try:
    from dynamics.jax_coordinates import JaxPhysicalCoordinates
except ImportError as error:
    assert "optional" in str(error)
else:
    raise AssertionError("Missing coordinate dependency was ignored")
try:
    simulate(p, RK4(), r, options=ExecutionOptions(backend="jax"))
except ImportError as error:
    assert "optional" in str(error)
else:
    raise AssertionError("Missing dependency was ignored")
'''
        result = subprocess.run([sys.executable, "-c", script], cwd="tests",
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_and_explicit_cpu_execution_match(self):
        p, r = problem(), SimulationRequest.uniform(t_span=(0., .03))
        a, b = simulate(p, RK4(), r), simulate(p, RK4(), r, options=ExecutionOptions())
        np.testing.assert_array_equal(a.states, b.states)
        with self.assertRaisesRegex(TypeError, "ExecutionOptions"):
            simulate(p, RK4(), r, options="jax")


@unittest.skipUnless(JAX_AVAILABLE, "Optional JAX dependency is not installed")
class JaxRK4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        cls.jax = jax
        cls.previous_x64 = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        cls.execution = ExecutionOptions(backend="jax")

    @classmethod
    def tearDownClass(cls):
        cls.jax.config.update("jax_enable_x64", cls.previous_x64)

    def assert_equivalent(self, p, request, tracking, execution=None, *, step=.1):
        method = RK4(step=step)
        observers = (EnergyObserver(),) if tracking else ()
        reference = simulate(p, method, request, observers=observers)
        result = simulate(p, method, request, options=execution or self.execution, observers=observers)
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
        request = SimulationRequest(t_span=(.3, .5),
                                    output_times=np.array([.3, .3001, .35, .371, .41, .49, .5]))
        for fc in (False, True):
            for tracking in (False, True):
                with self.subTest(fc=fc, tracking=tracking):
                    self.assert_equivalent(problem(fc), request, tracking, step=.06)

    def test_nonzero_observer_momentum_stays_separate_and_starts_fresh(self):
        p = problem()
        request = SimulationRequest((.3, .43), np.array([.3, .301, .35, .41, .43]))
        method = RK4(step=.05)
        initial = np.array([.7, -.2, 1.1])
        observer = EnergyObserver(initial)
        reference = simulate(p, method, request, observers=(observer,))
        zero = simulate(p, method, request, options=self.execution, observers=(EnergyObserver(),))
        tracked = simulate(p, method, request, options=self.execution, observers=(observer,))
        again = simulate(p, method, request, options=self.execution, observers=(observer,))
        np.testing.assert_array_equal(tracked.states, zero.states)
        np.testing.assert_allclose(tracked.diagnostics['extended_momentum'],
            zero.diagnostics['extended_momentum'] + initial[:, None], rtol=0., atol=3e-16)
        np.testing.assert_allclose(tracked.diagnostics['extended_momentum'],
            reference.diagnostics['extended_momentum'], rtol=0., atol=3e-16)
        np.testing.assert_array_equal(tracked.diagnostics['extended_momentum'], again.diagnostics['extended_momentum'])
        np.testing.assert_array_equal(observer.initial_momentum, initial)

    def test_sparse_sampling_shadow_independence_and_particle_independence(self):
        p = problem()
        nodes = np.linspace(0., .5, 6)
        dense = np.unique(np.concatenate((nodes, [.02, .08, .21, .22, .49])))
        sampled = simulate(p, RK4(step=0.1), SimulationRequest((0., .5),  dense), options=self.execution)
        sparse = self.assert_equivalent(p, SimulationRequest((0., .5),  nodes[[0, 2, 5]]), False)
        np.testing.assert_allclose(sampled.states[:, np.searchsorted(dense, sparse.t)], sparse.states,
                                   rtol=1e-14, atol=1e-14)
        x, y = GuidingCenterDynamics.coordinates.split(p.initial_state)
        for i in range(3):
            one = InitialValueProblem(p.dynamics, GuidingCenterDynamics.coordinates.pack_components(x[i:i+1], y[i:i+1]), layout=GuidingCenterDynamics.coordinates)
            solo = simulate(one, RK4(step=0.1), SimulationRequest((0., .5),  sparse.t))
            np.testing.assert_allclose(sparse.states[[i, i+3]], solo.states, rtol=1e-13, atol=1e-13)

    def test_fourth_order_against_an_exact_time_dependent_shear(self):
        grid = Grid.periodic(24, 16)
        mode = .05 * np.sin(grid.x[:, None]) * np.ones((1, grid.ny))
        potential = Potential(grid, modes=mode[None], frequencies=np.array([.7]))
        p = InitialValueProblem(GuidingCenterDynamics(potential), GuidingCenterDynamics.coordinates.pack_components(np.array([.73]), np.array([.4])), layout=GuidingCenterDynamics.coordinates)
        # Phi is independent of y: x stays fixed and y' is a cosine in time.
        amplitude = potential.evaluate(0., np.array([.73]), np.array([.4]), dx=1)[0]
        exact = .4 + amplitude * np.sin(2*np.pi*.7) / (2*np.pi*.7)
        errors = []
        for h in (.2, .1, .05):
            result = simulate(p, RK4(step=h), SimulationRequest((0., 1.),  np.array([0., 1.])), options=self.execution)
            errors.append(abs(result.states[1, -1] - exact))
        self.assertTrue(all(14 < a/b < 18 for a, b in zip(errors[:-1], errors[1:])), errors)

    def test_precision_and_unsupported_capabilities_fail_explicitly(self):
        p, r = problem(), SimulationRequest.uniform(t_span=(0., .02))
        simulate(p, RK4(), r, options=self.execution)
        try:
            self.jax.config.update("jax_enable_x64", False)
            with self.assertRaisesRegex(RuntimeError, "float64"):
                simulate(p, RK4(), r, options=self.execution)
        finally:
            self.jax.config.update("jax_enable_x64", True)
        for method, observers in ((RK4(progress=True), ()), (RK4(), (lambda event: None,))):
            with self.assertRaisesRegex(NotImplementedError, "callbacks"):
                simulate(p, method, r, options=self.execution, observers=observers)
        class CustomDynamics(GuidingCenterDynamics):
            def vector_field(self, time, state):
                return np.zeros_like(state)
        custom = InitialValueProblem(CustomDynamics(p.dynamics.potential), p.initial_state, layout=GuidingCenterDynamics.coordinates)
        with self.assertRaisesRegex(TypeError, "built-in"):
            simulate(custom, RK4(), r, options=self.execution)

    def test_near_endpoints_and_unsaved_nonfinite_states(self):
        p = problem()
        times = np.array([0., 1e-16, .1-1e-16, .1, .1+1e-16, .2])
        self.assert_equivalent(p, SimulationRequest((0., .2),  times), True)
        # A finite initial FC state can still overflow during unsaved steps.
        fc = FullCyclotronDynamics(p.dynamics.potential, rho=1., eta=1e-100)
        initial = FullCyclotronDynamics.coordinates.pack_components(np.array([0.]), np.array([0.]), np.array([1e200]), np.array([1e200]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            simulate(InitialValueProblem(fc, initial, layout=FullCyclotronDynamics.coordinates), RK4(step=0.1),
                     SimulationRequest((0., 1.),  np.array([0., 1.])), options=self.execution)

    def test_gpu_equivalence_or_explicit_unavailability(self):
        p, r = problem(), SimulationRequest.uniform(t_span=(0., .02))
        execution = ExecutionOptions(backend="jax", device="gpu")
        try:
            devices = self.jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                simulate(p, RK4(), r, options=execution)
            self.skipTest("No JAX GPU is available; explicit failure was verified")
        self.assert_equivalent(p, r, True, execution)


if __name__ == "__main__":
    unittest.main()
