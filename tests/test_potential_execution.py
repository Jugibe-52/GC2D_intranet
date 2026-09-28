"""Public execution selection, shared preparation, and device-cache lifetime."""

from dataclasses import FrozenInstanceError
import importlib.util
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution import Execution
from potential import Grid, Potential, ScipyPotentialEvaluator


def _field() -> Potential:
    """Prepare a small field whose time reconstruction is known independently."""
    grid = Grid.periodic(12, 14)
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    return Potential(grid, mean=np.sin(x) * np.cos(y),
                     modes=(0.2 * np.cos(x + y) + 0.1j * np.sin(x - y))[None],
                     frequencies=np.array([0.7]))


class ExecutionTests(unittest.TestCase):
    """Configuration is immutable, hashable, and independent of optional JAX."""

    def test_defaults_and_valid_choices(self) -> None:
        self.assertEqual(Execution(), Execution(backend="scipy", device="cpu"))
        for device in ("cpu", "gpu"):
            execution = Execution(backend="jax", device=device, device_index=np.int64(2))
            self.assertEqual(execution, pickle.loads(pickle.dumps(execution)))
            self.assertEqual(hash(execution), hash(Execution(backend="jax", device=device, device_index=2)))
            self.assertIs(type(execution.device_index), int)
        with self.assertRaises(FrozenInstanceError):
            Execution().device = "gpu"

    def test_invalid_choices(self) -> None:
        for options in ({"backend": "numpy"}, {"device": "tpu"}, {"device": "gpu"},
                        {"device_index": 1}, {"device_index": -1}, {"device_index": True},
                        {"device_index": np.bool_(False)}, {"device_index": 0.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Execution(**options)


class ScipyExecutionTests(unittest.TestCase):
    """The default public API preserves its CPU behavior and read-only data."""

    def test_default_and_explicit_scipy_share_preparation(self) -> None:
        potential = _field()
        direct = ScipyPotentialEvaluator(potential.prepared)
        expected = direct.evaluate(0.3, 0.4, 0.5, dx=1)
        for execution in (None, Execution()):
            result = potential.evaluate(0.3, 0.4, 0.5, dx=1, execution=execution)
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(potential.electric_field(0.3, execution=execution),
                                          direct.electric_field(0.3))
            np.testing.assert_array_equal(potential.evaluate_grid(0.3, execution=execution),
                                          direct.evaluate_grid(0.3))

    def test_prepared_arrays_and_public_physical_attributes_are_readonly(self) -> None:
        potential = _field()
        data = potential.prepared.spline_data
        for value in (potential.mean, potential.modes, potential.frequencies,
                      potential.prepared.samples, data.knots_x, data.knots_y, data.coefficients):
            self.assertFalse(value.flags.writeable)
        with self.assertRaises(AttributeError):
            potential.mean = np.zeros(potential.grid.shape)
        with self.assertRaises(FrozenInstanceError):
            potential.prepared.interpolation_order = 2

    def test_execution_type_validation_is_consistent(self) -> None:
        potential = _field()
        for call in (lambda: potential.evaluate(0., 0., 0., execution="jax"),
                     lambda: potential.evaluate_grid(0., execution="jax"),
                     lambda: potential.electric_field(0., execution="jax")):
            with self.assertRaisesRegex(TypeError, "Execution instance"):
                call()


@unittest.skipUnless(importlib.util.find_spec("jax"), "Optional JAX is not installed")
class JaxExecutionTests(unittest.TestCase):
    """Reuse native evaluators through Potential without changing its defaults."""

    @classmethod
    def setUpClass(cls) -> None:
        import jax

        cls.jax = jax
        cls.previous_x64 = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.jax.config.update("jax_enable_x64", cls.previous_x64)

    def test_equivalent_configurations_reuse_one_evaluator_for_all_operations(self) -> None:
        from potential.jax_evaluator import JaxPotentialEvaluator

        potential = _field()
        with patch("potential.jax_evaluator.JaxPotentialEvaluator", wraps=JaxPotentialEvaluator) as constructor:
            result = potential.evaluate(0.3, 0.4, 0.5, execution=Execution(backend="jax"))
            field = potential.electric_field(0.3, 0.4, 0.5, execution=Execution(backend="jax"))
            grid = potential.evaluate_grid(0.3, execution=Execution(backend="jax"))
            self.assertEqual(constructor.call_count, 1)
        for value in (result, *field, grid):
            self.assertIsInstance(value, self.jax.Array)
            self.assertEqual(value.dtype, np.dtype("float64"))
        default = potential.evaluate(0.3, 0.4, 0.5)
        self.assertIsInstance(default, np.ndarray)
        np.testing.assert_allclose(result, default, atol=3e-12)

    def test_shared_reconstruction_against_independent_harmonic_formula(self) -> None:
        potential = _field()
        time = np.array([0.2, 0.7])
        omega = 2 * np.pi * potential.frequencies[0]
        for dt in (0, 1, 2):
            expected = 2 * np.real(potential.modes[0, ..., None] * np.exp(1j * omega * time) * (1j * omega)**dt)
            if dt == 0:
                expected += potential.mean[..., None]
            for execution in (Execution(), Execution(backend="jax")):
                np.testing.assert_allclose(potential.evaluate_grid(time, dt=dt, execution=execution),
                                           expected, rtol=3e-12, atol=3e-12)

    def test_first_call_inside_jit_does_not_cache_tracers(self) -> None:
        potential = _field()
        execution = Execution(backend="jax")
        with self.jax.checking_leaks():
            compiled = self.jax.jit(lambda t, x, y: potential.evaluate(t, x, y, dx=1, execution=execution))
            first = compiled(0.3, 0.4, 0.5)
            first.block_until_ready()
        # A later untraced call must be usable after the first trace has ended.
        np.testing.assert_allclose(first, potential.evaluate(0.3, 0.4, 0.5, dx=1, execution=execution), atol=3e-12)
        np.testing.assert_allclose(first, potential.evaluate(0.3, 0.4, 0.5, dx=1), atol=3e-12)

    def test_serialization_excludes_device_cache_and_rebuilds_readonly_data(self) -> None:
        from potential.jax_evaluator import JaxPotentialEvaluator

        potential = _field()
        execution = Execution(backend="jax")
        expected = potential.evaluate(0.3, 0.4, 0.5, execution=execution)
        restored = pickle.loads(pickle.dumps(potential))
        self.assertFalse(restored.mean.flags.writeable)
        self.assertIsInstance(restored.evaluate(0.3, 0.4, 0.5), np.ndarray)
        with patch("potential.jax_evaluator.JaxPotentialEvaluator", wraps=JaxPotentialEvaluator) as constructor:
            actual = restored.evaluate(0.3, 0.4, 0.5, execution=execution)
            self.assertEqual(constructor.call_count, 1)
        np.testing.assert_array_equal(expected, actual)

    def test_cached_evaluation_rejects_disabled_float64_and_does_not_poison_default(self) -> None:
        potential = _field()
        execution = Execution(backend="jax")
        potential.evaluate(0.3, 0.4, 0.5, execution=execution).block_until_ready()
        self.jax.config.update("jax_enable_x64", False)
        try:
            with self.assertRaisesRegex(RuntimeError, "requires float64"):
                potential.evaluate(0.3, 0.4, 0.5, execution=execution)
            self.assertIsInstance(potential.evaluate(0.3, 0.4, 0.5), np.ndarray)
        finally:
            self.jax.config.update("jax_enable_x64", True)

    def test_gpu_request_fails_explicitly_without_affecting_scipy(self) -> None:
        potential = _field()
        with patch.object(self.jax, "devices", side_effect=RuntimeError("Unavailable")):
            with self.assertRaisesRegex(RuntimeError, "gpu.*unavailable"):
                potential.evaluate(0.3, 0.4, 0.5, execution=Execution(backend="jax", device="gpu"))
        self.assertIsInstance(potential.evaluate(0.3, 0.4, 0.5), np.ndarray)


if __name__ == "__main__":
    unittest.main()
