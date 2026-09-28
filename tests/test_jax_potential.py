"""Check optional JAX evaluation against the canonical SciPy interpolants."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from potential import Grid, JaxPotentialEvaluator, Potential, load_gc2d_h5_potential


def _potential(degree: int = 3) -> Potential:
    """Make a non-square, shifted periodic field with two complex modes."""
    grid = Grid(-0.4, 0.7, 2 * np.pi / 12, 2 * np.pi / 14, 12, 14, 2 * np.pi)
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    return Potential(
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
from contracts.execution import Execution
from potential import Grid, Potential, JaxPotentialEvaluator
p = Potential(Grid.periodic(8, 8))
assert p.evaluate(0., 0., 0.) == 0.
try:
    p.evaluate(0., 0., 0., execution=Execution(backend='jax'))
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
        cls.evaluator = JaxPotentialEvaluator(cls.potential)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.jax.config.update("jax_enable_x64", cls.previous_x64)

    def test_values_gradients_hessians_and_time_derivatives(self) -> None:
        rng = np.random.default_rng(27)
        x, y = rng.uniform(-20, 20, (2, 256))
        for dx, dy, dt in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0),
                           (1, 1, 0), (0, 2, 0), (0, 0, 1), (0, 0, 2), (1, 1, 2)):
            with self.subTest(dx=dx, dy=dy, dt=dt):
                actual = self.evaluator.evaluate(0.23, x, y, dx=dx, dy=dy, dt=dt)
                expected = self.potential.evaluate(0.23, x, y, dx=dx, dy=dy, dt=dt)
                self.assertIsInstance(actual, self.jax.Array)
                self.assertEqual(actual.dtype, np.dtype("float64"))
                self.assertEqual(actual.devices(), {self.evaluator.device})
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def test_all_supported_degrees_and_high_derivatives(self) -> None:
        for degree in (2, 3, 4, 5):
            potential = _potential(degree)
            evaluator = JaxPotentialEvaluator(potential)
            for dx, dy in ((0, 0), (1, 1), (degree - 1, 0), (0, degree - 1)):
                with self.subTest(degree=degree, dx=dx, dy=dy):
                    np.testing.assert_allclose(
                        evaluator.evaluate(0.71, np.array([0.12, 2.4]), np.array([1.1, 3.7]), dx=dx, dy=dy),
                        potential.evaluate(0.71, np.array([0.12, 2.4]), np.array([1.1, 3.7]), dx=dx, dy=dy),
                        rtol=2e-10, atol=2e-10,
                    )

    def test_periodic_seams_and_knots(self) -> None:
        grid = self.potential.grid
        x = np.concatenate((grid.x, [grid.xmin - 1e-12, grid.xmin,
                                     grid.xmin + grid.period, grid.xmin + grid.period + 1e-12]))
        y = np.full_like(x, grid.ymin)
        for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 2)):
            np.testing.assert_allclose(
                self.evaluator.evaluate(0.4, x, y, dx=dx, dy=dy),
                self.potential.evaluate(0.4, x, y, dx=dx, dy=dy), rtol=3e-12, atol=3e-12,
            )
        np.testing.assert_allclose(
            self.evaluator.evaluate(0.4, x + 3 * grid.period, y - 2 * grid.period),
            self.evaluator.evaluate(0.4, x, y), rtol=3e-12, atol=3e-12,
        )

    def test_scalars_broadcasting_grid_and_empty_points(self) -> None:
        for time, x, y in ((0.2, 0.3, 0.4),
                           (np.array([[0.1], [0.2]]), np.array([[0.3, 0.4]]), np.array([[0.5, 0.6]])),
                           (0.2, np.empty(0), np.empty(0))):
            actual = self.evaluator.evaluate(time, x, y)
            expected = self.potential.evaluate(time, x, y)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
        for time in (0.3, np.array([0.2, 0.3]), np.array([[0.2], [0.3]])):
            for dt in (0, 1, 2):
                np.testing.assert_allclose(self.evaluator.evaluate_grid(time, dt=dt),
                                           self.potential.evaluate_grid(time, dt=dt), rtol=3e-12, atol=3e-12)
        for x, y in ((None, None), (np.array([0.1, 0.2]), np.array([0.3, 0.4]))):
            np.testing.assert_allclose(self.evaluator.electric_field(0.2, x, y),
                                       self.potential.electric_field(0.2, x, y), rtol=3e-12, atol=3e-12)

    def test_no_modes_and_gyroaveraged_potential(self) -> None:
        for potential in (Potential(self.potential.grid, mean=self.potential.mean),
                          Potential(self.potential.grid), self.potential.gyroaverage(0.3)):
            evaluator = JaxPotentialEvaluator(potential)
            for dt in (0, 1, 2):
                np.testing.assert_allclose(evaluator.evaluate(0.3, 0.2, 0.4, dt=dt),
                                           potential.evaluate(0.3, 0.2, 0.4, dt=dt), rtol=3e-12, atol=3e-12)
                np.testing.assert_allclose(evaluator.evaluate_grid(np.array([0., 0.2]), dt=dt),
                                           potential.evaluate_grid(np.array([0., 0.2]), dt=dt), rtol=3e-12, atol=3e-12)

    def test_jit_and_automatic_spatial_derivative(self) -> None:
        compiled = self.jax.jit(lambda t, x, y: self.evaluator.evaluate(t, x, y, dx=1))
        np.testing.assert_allclose(compiled(0.2, 0.3, 0.4),
                                   self.potential.evaluate(0.2, 0.3, 0.4, dx=1), atol=3e-12)
        derivative = self.jax.grad(lambda x: self.evaluator.evaluate(0.2, x, 0.4))(0.3)
        np.testing.assert_allclose(derivative, compiled(0.2, 0.3, 0.4), atol=3e-12)

    def test_hdf5_loaded_field_uses_the_same_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.h5"
            axis = np.arange(8) * 0.01
            x, y = np.meshgrid(axis, axis, indexing="ij")
            with h5py.File(path, "w") as h5:
                h5["Rcells"], h5["Zcells"] = axis, axis
                h5["freqs"] = np.array([0., 2., 3.])
                h5["fields"] = np.stack((x + y, np.sin(x + y) + 1j * np.cos(y), np.cos(x) + 1j * y))
            potential = load_gc2d_h5_potential(path).gyroaverage(0.3)
            evaluator = JaxPotentialEvaluator(potential)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
                np.testing.assert_allclose(evaluator.evaluate(0.3, 0.4, 0.5, dx=dx, dy=dy),
                                           potential.evaluate(0.3, 0.4, 0.5, dx=dx, dy=dy), rtol=1e-11, atol=1e-10)

    def test_invalid_requests_and_explicit_device_failure(self) -> None:
        for kwargs in ({"dx": True}, {"dy": -1}, {"dx": 3}, {"dt": 3}, {"dt": 0.5}):
            with self.assertRaises(ValueError):
                self.evaluator.evaluate(0., 0., 0., **kwargs)
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(0., np.zeros(2), np.zeros(3))
        with self.assertRaises(ValueError):
            self.evaluator.electric_field(0., x=0.)
        for kwargs in ({"device": "tpu"}, {"device_index": -1}, {"device_index": True}):
            with self.assertRaises(ValueError):
                JaxPotentialEvaluator(self.potential, **kwargs)
        with patch.object(self.jax, "devices", side_effect=RuntimeError("Unavailable")):
            with self.assertRaisesRegex(RuntimeError, "gpu.*unavailable"):
                JaxPotentialEvaluator(self.potential, device="gpu")
        with patch.object(self.jax, "devices", return_value=[]):
            with self.assertRaisesRegex(ValueError, "index 0 is unavailable"):
                JaxPotentialEvaluator(self.potential)

    def test_float32_is_rejected_without_changing_global_configuration(self) -> None:
        self.jax.config.update("jax_enable_x64", False)
        try:
            with self.assertRaisesRegex(RuntimeError, "requires float64"):
                JaxPotentialEvaluator(self.potential)
            with self.assertRaisesRegex(RuntimeError, "requires float64"):
                self.evaluator.evaluate(0., 0., 0.)
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
        evaluator = JaxPotentialEvaluator(self.potential, device="gpu")
        x, y = np.linspace(-1., 7., 256), np.linspace(7., -1., 256)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
            actual = evaluator.evaluate(0.2, x, y, dx=dx, dy=dy)
            self.assertEqual(actual.devices(), {devices[0]})
            np.testing.assert_allclose(actual, self.potential.evaluate(0.2, x, y, dx=dx, dy=dy),
                                       rtol=3e-12, atol=3e-12)


if __name__ == "__main__":
    unittest.main()
