"""Check optional JAX evaluation against the canonical SciPy interpolants."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from potential import Grid, JaxPotential, Potential


def _potential(degree: int = 3) -> JaxPotential:
    """Make a non-square, shifted periodic field with two complex modes."""
    grid = Grid(-0.4, 0.7, 2 * np.pi / 12, 2 * np.pi / 14, 12, 14, 2 * np.pi)
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    return JaxPotential(
        grid, mean=0.2 * np.cos(x) * np.sin(y),
        modes=np.stack((0.1 * np.sin(x + y) + 0.04j * np.cos(x - y),
                        0.03 * np.cos(2 * x) + 0.02j * np.sin(y))),
        frequencies=np.array([1.0, 0.37]), interpolation_order=degree,
    )


class OptionalJaxImportTests(unittest.TestCase):
    """Keep normal potential and simulation imports independent of JAX."""

    def test_core_imports_and_clear_optional_dependency_error(self) -> None:
        script = """
import builtins
original = builtins.__import__
def without_jax(name, *args, **kwargs):
    if name == 'jax' or name.startswith('jax.'):
        raise ModuleNotFoundError('JAX intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = without_jax
import simulation
from potential import Grid, JaxPotential, Potential
jp = JaxPotential(Grid.periodic(8, 8))
try:
    jp.evaluate(0., 0., 0.)
except ImportError as exc:
    assert "optional 'jax' extra" in str(exc)
else:
    raise AssertionError('Missing JAX must fail explicitly')
p = Potential(Grid.periodic(8, 8))
assert p.evaluate(0., 0., 0.) == 0.
try:
    from execution._jax import resolve_device
    from contracts.execution_options import ExecutionOptions
    resolve_device(ExecutionOptions(backend="jax"))
except ImportError as exc:
    assert "optional 'jax' extra" in str(exc)
else:
    raise AssertionError('Missing JAX must fail explicitly')
assert p.evaluate(0., 0., 0.) == 0.
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("jax"), "Optional JAX is not installed")
class JaxPotentialTests(unittest.TestCase):
    """Numerical, placement, and precision contracts for device evaluation."""

    @classmethod
    def setUpClass(cls) -> None:
        import jax

        cls.jax = jax
        cls.previous_x64 = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        cls.potential = _potential()


    @classmethod
    def tearDownClass(cls) -> None:
        cls.jax.config.update("jax_enable_x64", cls.previous_x64)

    def test_values_gradients_hessians_and_time_derivatives(self) -> None:
        rng = np.random.default_rng(27)
        x, y = rng.uniform(-20, 20, (2, 256))
        for dx, dy, dt in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0),
                           (1, 1, 0), (0, 2, 0), (0, 0, 1), (0, 0, 2), (1, 1, 2)):
            with self.subTest(dx=dx, dy=dy, dt=dt):
                actual = self.potential.evaluate(self.jax.numpy.asarray(0.23), x, y, dx=dx, dy=dy, dt=dt)
                expected = Potential.evaluate(self.potential, 0.23, x, y, dx=dx, dy=dy, dt=dt)
                self.assertIsInstance(actual, self.jax.Array)
                self.assertEqual(actual.dtype, np.dtype("float64"))
                self.assertEqual(actual.devices(), {self.jax.devices("cpu")[0]})
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def test_all_supported_degrees_and_high_derivatives(self) -> None:
        for degree in (2, 3, 4, 5):
            potential = _potential(degree)
            for dx, dy in ((0, 0), (1, 1), (degree - 1, 0), (0, degree - 1)):
                with self.subTest(degree=degree, dx=dx, dy=dy):
                    np.testing.assert_allclose(
                        potential.evaluate(self.jax.numpy.asarray(0.71), np.array([0.12, 2.4]), np.array([1.1, 3.7]), dx=dx, dy=dy),
                        Potential.evaluate(potential, 0.71, np.array([0.12, 2.4]), np.array([1.1, 3.7]), dx=dx, dy=dy),
                        rtol=2e-10, atol=2e-10,
                    )

    def test_periodic_seams_and_knots(self) -> None:
        grid = self.potential.grid
        x = np.concatenate((grid.x, [grid.xmin - 1e-12, grid.xmin,
                                     grid.xmin + grid.period, grid.xmin + grid.period + 1e-12]))
        y = np.full_like(x, grid.ymin)
        for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 2)):
            np.testing.assert_allclose(
                self.potential.evaluate(self.jax.numpy.asarray(0.4), x, y, dx=dx, dy=dy),
                Potential.evaluate(self.potential, 0.4, x, y, dx=dx, dy=dy), rtol=3e-12, atol=3e-12,
            )
        np.testing.assert_allclose(
            self.potential.evaluate(self.jax.numpy.asarray(0.4), x + 3 * grid.period, y - 2 * grid.period),
            Potential.evaluate(self.potential, 0.4, x, y), rtol=3e-12, atol=3e-12,
        )

    def test_scalars_broadcasting_grid_and_empty_points(self) -> None:
        for time, x, y in ((0.2, 0.3, 0.4),
                           (np.array([[0.1], [0.2]]), np.array([[0.3, 0.4]]), np.array([[0.5, 0.6]])),
                           (0.2, np.empty(0), np.empty(0))):
            actual = self.potential.evaluate(self.jax.numpy.asarray(time), x, y)
            expected = Potential.evaluate(self.potential, time, x, y)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
        for time in (0.3, np.array([0.2, 0.3]), np.array([[0.2], [0.3]])):
            for dt in (0, 1, 2):
                self.assertIsInstance(self.potential.evaluate_grid(time, dt=dt), np.ndarray)
                with self.assertRaisesRegex(TypeError, "NumPy time"):
                    self.potential.evaluate_grid(self.jax.numpy.asarray(time), dt=dt)
        x, y = np.array([0.1, 0.2]), np.array([0.3, 0.4])
        np.testing.assert_allclose(self.potential.electric_field(self.jax.numpy.asarray(0.2), x, y),
                                   self.potential.electric_field(0.2, x, y), rtol=3e-12, atol=3e-12)
        self.assertIsInstance(self.potential.electric_field(0.2)[0], self.jax.Array)
        self.assertIsInstance(self.potential.electric_field(self.jax.numpy.asarray(0.2))[0], self.jax.Array)

    def test_no_modes_and_gyroaveraged_potential(self) -> None:
        for potential in (JaxPotential(self.potential.grid, mean=self.potential.mean),
                          JaxPotential(self.potential.grid), self.potential.gyroaverage(0.3)):
            for dt in (0, 1, 2):
                np.testing.assert_allclose(potential.evaluate(self.jax.numpy.asarray(0.3), 0.2, 0.4, dt=dt),
                                           Potential.evaluate(potential, 0.3, 0.2, 0.4, dt=dt), rtol=3e-12, atol=3e-12)

    def test_jit_and_automatic_spatial_derivative(self) -> None:
        compiled = self.jax.jit(lambda t, x, y: self.potential.evaluate(t, x, y, dx=1))
        np.testing.assert_allclose(compiled(0.2, 0.3, 0.4),
                                   self.potential.evaluate(0.2, 0.3, 0.4, dx=1), atol=3e-12)
        derivative = self.jax.grad(lambda x: self.potential.evaluate(0.2, x, 0.4))(0.3)
        np.testing.assert_allclose(derivative, compiled(0.2, 0.3, 0.4), atol=3e-12)

    def test_hdf5_loaded_field_uses_the_same_splines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.h5"
            axis = np.arange(8) * 0.01
            x, y = np.meshgrid(axis, axis, indexing="ij")
            with h5py.File(path, "w") as h5:
                h5["Rcells"], h5["Zcells"] = axis, axis
                h5["freqs"] = np.array([0., 2., 3.])
                h5["fields"] = np.stack((x + y, np.sin(x + y) + 1j * np.cos(y), np.cos(x) + 1j * y))
            potential = JaxPotential.load(path, characteristic_frequency=2.0, indx=(1, 2)).gyroaverage(0.3)
            self.assertIsInstance(potential, JaxPotential)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
                np.testing.assert_allclose(potential.evaluate(self.jax.numpy.asarray(0.3), 0.4, 0.5, dx=dx, dy=dy),
                                           Potential.evaluate(potential, 0.3, 0.4, 0.5, dx=dx, dy=dy), rtol=1e-11, atol=1e-10)

    def test_invalid_requests_on_both_backends(self) -> None:
        for evaluate in (self.potential.evaluate,
                         lambda *args, **kwargs: Potential.evaluate(self.potential, *args, **kwargs)):
            for time in (0., self.jax.numpy.asarray(0.)):
                for kwargs in ({"dx": True}, {"dy": -1}, {"dx": 3}, {"dt": 3}, {"dt": 0.5}):
                    with self.subTest(evaluate=evaluate, derivative=kwargs), self.assertRaises(ValueError):
                        evaluate(time, 0., 0., **kwargs)
                with self.assertRaisesRegex(ValueError, "same shape"):
                    evaluate(time, np.zeros(2), np.zeros(3))
                with self.assertRaisesRegex(ValueError, "provided together"):
                    self.potential.electric_field(time, x=0.)

    def test_float32_is_rejected_without_changing_global_configuration(self) -> None:
        self.jax.config.update("jax_enable_x64", False)
        try:
            with self.assertRaisesRegex(RuntimeError, "requires float64"):
                self.potential.evaluate(self.jax.numpy.asarray(0.), 0., 0.)
            self.assertFalse(self.jax.config.x64_enabled)
        finally:
            self.jax.config.update("jax_enable_x64", True)

    def test_gpu_values_when_available(self) -> None:
        try:
            devices = self.jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            self.skipTest("No JAX GPU device is available")
        x, y = np.linspace(-1., 7., 256), np.linspace(7., -1., 256)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
            actual = self.potential.evaluate(0.2, self.jax.device_put(x, devices[0]), self.jax.device_put(y, devices[0]), dx=dx, dy=dy)
            self.assertEqual(actual.devices(), {devices[0]})
            np.testing.assert_allclose(actual, Potential.evaluate(self.potential, 0.2, x, y, dx=dx, dy=dy),
                                       rtol=3e-12, atol=3e-12)


if __name__ == "__main__":
    unittest.main()
