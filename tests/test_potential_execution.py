"""Simulation execution selection and backend-independent potential data."""

from dataclasses import FrozenInstanceError
import importlib.util
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from potential import Grid, JaxPotential, Potential


def _field(cls: type[Potential] = Potential) -> Potential:
    """Prepare a small field whose time reconstruction is known independently."""
    grid = Grid.periodic(12, 14)
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    return cls(grid, mean=np.sin(x) * np.cos(y),
                     modes=(0.2 * np.cos(x + y) + 0.1j * np.sin(x - y))[None],
                     frequencies=np.array([0.7]))


class ExecutionTests(unittest.TestCase):
    """Configuration is immutable, hashable, and independent of optional JAX."""

    def test_defaults_and_valid_choices(self) -> None:
        self.assertEqual(ExecutionOptions(), ExecutionOptions(backend="scipy", device="cpu"))
        for device in ("cpu", "gpu"):
            execution = ExecutionOptions(backend="jax", device=device, device_index=np.int64(2))
            self.assertEqual(execution, pickle.loads(pickle.dumps(execution)))
            self.assertEqual(hash(execution), hash(ExecutionOptions(backend="jax", device=device, device_index=2)))
            self.assertIs(type(execution.device_index), int)
        with self.assertRaises(FrozenInstanceError):
            ExecutionOptions().device = "gpu"

    def test_invalid_choices(self) -> None:
        for options in ({"backend": "numpy"}, {"device": "tpu"}, {"device": "gpu"},
                        {"device_index": 1}, {"device_index": -1}, {"device_index": True},
                        {"device_index": np.bool_(False)}, {"device_index": 0.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ExecutionOptions(**options)


class ScipyExecutionTests(unittest.TestCase):
    """Host arguments retain SciPy behavior and immutable physical data."""

    def test_host_arguments_and_grid_outputs_are_numpy(self) -> None:
        potential = _field()
        for coordinates in ((0.4, 0.5), ([0.4], [0.5]),
                            (np.array([0.4]), np.array([0.5]))):
            with self.subTest(coordinates=coordinates):
                result = potential.evaluate(0.3, *coordinates, dx=1)
                self.assertIsInstance(result, np.ndarray)
        for value in (*potential.electric_field(0.3), potential.evaluate_grid(0.3)):
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.shape, potential.grid.shape)

    def test_physical_arrays_and_spline_exports_are_readonly(self) -> None:
        potential = _field()
        for value in (potential.mean, potential.modes, potential.frequencies,
                      *potential._spline_data):
            self.assertFalse(value.flags.writeable)
        for name, value in (("mean", np.zeros(potential.grid.shape)),
                            ("modes", np.empty((0, *potential.grid.shape))),
                            ("frequencies", np.empty(0)), ("grid", potential.grid),
                            ("interpolation_order", 2)):
            with self.subTest(attribute=name), self.assertRaises(AttributeError):
                setattr(potential, name, value)

    def test_grid_reconstruction_against_independent_harmonic_formula(self) -> None:
        potential = _field()
        time = np.array([0.2, 0.7])
        omega = 2 * np.pi * potential.frequencies[0]
        for dt in (0, 1, 2):
            expected = 2 * np.real(potential.modes[0, ..., None] * np.exp(1j * omega * time) * (1j * omega)**dt)
            if dt == 0:
                expected += potential.mean[..., None]
            np.testing.assert_allclose(potential.evaluate_grid(time, dt=dt),
                                       expected, rtol=3e-12, atol=3e-12)

    def test_potential_does_not_accept_execution_options(self) -> None:
        potential = _field()
        for call in (lambda: potential.evaluate(0., 0., 0., execution="jax"),
                     lambda: potential.evaluate_grid(0., execution="jax"),
                     lambda: potential.electric_field(0., execution="jax")):
            with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                call()


@unittest.skipUnless(importlib.util.find_spec("jax"), "Optional JAX is not installed")
class JaxExecutionTests(unittest.TestCase):
    """Explicit JAX evaluation reuses concrete coefficients without refitting."""

    @classmethod
    def setUpClass(cls) -> None:
        import jax

        cls.jax = jax
        cls.previous_x64 = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.jax.config.update("jax_enable_x64", cls.previous_x64)

    def test_class_selects_backend_for_scalar_numpy_and_jax_inputs(self) -> None:
        potential = _field(JaxPotential)
        expected = Potential.evaluate(potential, 0.3, 0.4, 0.5)
        arguments = (0.3, 0.4, 0.5)
        for position in range(3):
            mixed = list(arguments)
            mixed[position] = self.jax.numpy.asarray(mixed[position])
            with self.subTest(position=position):
                actual = potential.evaluate(*mixed)
                self.assertIsInstance(actual, self.jax.Array)
                self.assertEqual(actual.dtype, np.dtype("float64"))
                np.testing.assert_allclose(actual, expected, atol=3e-12)
        scalar_result = potential.evaluate(*arguments)
        self.assertIsInstance(scalar_result, self.jax.Array)
        self.assertEqual(set(potential._jax_buffers), scalar_result.devices())
        host = _field()
        self.assertIsInstance(host.evaluate(*mixed), np.ndarray)

    def test_factories_and_gyroaverage_preserve_the_subclass(self) -> None:
        potential = JaxPotential.random(A=0.2, M=3, nx=12, ny=14)
        self.assertIsInstance(potential, JaxPotential)
        self.assertIs(potential.gyroaverage(0), potential)
        averaged = potential.gyroaverage(0.3)
        self.assertIsInstance(averaged, JaxPotential)
        self.assertIsInstance(averaged.evaluate(0.2, 0.4, 0.5), self.jax.Array)
        x, y = np.meshgrid(potential.grid.x, potential.grid.y, indexing="ij")
        for t in (0.2, self.jax.numpy.asarray(0.2)):
            field = potential.electric_field(t)
            self.assertIsInstance(field[0], self.jax.Array)
            np.testing.assert_allclose(field[0], -Potential.evaluate(potential, np.asarray(t), x, y, dx=1), atol=3e-12)
        compiled = self.jax.jit(potential.electric_field)(0.2)
        np.testing.assert_allclose(compiled, potential.electric_field(0.2), atol=3e-12)

    def test_preparation_shares_splines_and_preserves_original_dynamics(self) -> None:
        from dynamics.gc import GuidingCenterDynamics
        from dynamics.fc import FullCyclotronDynamics
        from execution._jax import prepare_dynamics

        potential = _field()
        for source in (GuidingCenterDynamics(potential, rho=0.3),
                       FullCyclotronDynamics(potential, rho=0.3, eta=-0.4)):
            with patch("potential.potential._build_periodic_spline", side_effect=AssertionError("Refitted spline")):
                prepared = prepare_dynamics(source)
                self.assertIs(prepare_dynamics(source), prepared)
            self.assertIs(source.potential, potential)
            self.assertIsInstance(prepared.potential, JaxPotential)
            self.assertIs(prepared.potential._splines, potential._splines)
            self.assertIs(prepared.potential.mean, potential.mean)
            self.assertIsNot(prepared.potential._jax_buffers, potential._jax_buffers)
            if isinstance(source, GuidingCenterDynamics):
                self.assertIs(prepared.effective_potential._splines, source.effective_potential._splines)
                self.assertEqual(prepared.rho, source.rho)
            else:
                self.assertEqual((prepared.rho, prepared.eta), (source.rho, source.eta))
        explicit = GuidingCenterDynamics(_field(JaxPotential), rho=0.3)
        self.assertIs(prepare_dynamics(explicit), explicit)
        class CustomPotential(Potential):
            def evaluate(self, *args, **kwargs):
                return 0.0
        with self.assertRaisesRegex(TypeError, "subclass overrides"):
            prepare_dynamics(GuidingCenterDynamics(CustomPotential(potential.grid)))

    def test_repeated_device_calls_share_coefficients_without_refitting(self) -> None:
        from dynamics.gc import GuidingCenterDynamics
        from execution._jax import require_builtin_dynamics, resolve_device

        potential = _field(JaxPotential)
        dynamics = GuidingCenterDynamics(potential)
        self.assertIs(require_builtin_dynamics(dynamics), dynamics)
        device = resolve_device(ExecutionOptions(backend="jax"))
        xd, yd = self.jax.device_put((np.array([0.4, 0.7]), np.array([0.5, 0.2])), device)
        with patch("potential.potential._build_periodic_spline", side_effect=AssertionError("Refitted spline")):
            result = potential.evaluate(0.3, xd, yd)
            buffers = potential._jax_buffers[device]
            field = potential.electric_field(0.3, xd, yd)
            repeated = potential.evaluate(0.3, xd, yd)
            self.assertIs(potential._jax_buffers[device], buffers)
            self.assertEqual(len(potential._jax_buffers), 1)
        for value in (result, *field, repeated):
            self.assertIsInstance(value, self.jax.Array)
            self.assertEqual(value.dtype, np.dtype("float64"))
            self.assertEqual(value.devices(), {device})
        np.testing.assert_allclose(result, Potential.evaluate(potential, 0.3, np.asarray(xd), np.asarray(yd)), atol=3e-12)
        np.testing.assert_array_equal(result, repeated)

    def test_first_use_inside_jit_does_not_leak_or_cache_tracers(self) -> None:
        potential = _field(JaxPotential)
        with self.jax.checking_leaks():
            compiled = self.jax.jit(lambda t, x, y: potential.evaluate(t, x, y, dx=1))
            first = compiled(0.3, 0.4, 0.5)
            first.block_until_ready()
        self.assertEqual(potential._jax_buffers, {})
        np.testing.assert_allclose(first, Potential.evaluate(potential, 0.3, 0.4, 0.5, dx=1), atol=3e-12)
        # A subsequent eager JAX call must remain usable and cache only concrete buffers.
        eager = potential.evaluate(0.3, self.jax.numpy.asarray(0.4), 0.5, dx=1)
        np.testing.assert_allclose(first, eager, atol=3e-12)
        self.assertEqual(len(potential._jax_buffers), 1)
        for buffers in potential._jax_buffers.values():
            self.assertFalse(any(isinstance(value, self.jax.core.Tracer) for value in buffers))

    def test_traced_time_and_vectorization_preserve_the_jax_path(self) -> None:
        potential = _field(JaxPotential)
        x, y = np.array([0.4, 0.7]), np.array([0.5, 0.2])
        time_derivative = self.jax.jit(self.jax.grad(lambda t: self.jax.numpy.sum(potential.evaluate(t, x, y))))
        np.testing.assert_allclose(time_derivative(0.3), np.sum(Potential.evaluate(potential, 0.3, x, y, dt=1)), atol=3e-12)
        times = self.jax.numpy.asarray([0.2, 0.7])
        batched = self.jax.jit(self.jax.vmap(lambda t: potential.evaluate(t, x, y)))(times)
        expected = np.stack([Potential.evaluate(potential, t, x, y) for t in np.asarray(times)])
        np.testing.assert_allclose(batched, expected, atol=3e-12)
        self.assertEqual(potential._jax_buffers, {})

    def test_closed_over_jax_arrays_do_not_leak_tracers_on_first_compilation(self) -> None:
        potential = _field(JaxPotential)
        t, x, y = (self.jax.numpy.asarray(value) for value in (0.3, 0.4, 0.5))
        with self.jax.checking_leaks():
            compiled = self.jax.jit(lambda: potential.evaluate(t, x, y))
            first = compiled()
            first.block_until_ready()
        for buffers in potential._jax_buffers.values():
            self.assertFalse(any(isinstance(value, self.jax.core.Tracer) for value in buffers))
        eager = potential.evaluate(t, x, y)
        np.testing.assert_allclose(first, eager, atol=3e-12)
        np.testing.assert_allclose(first, Potential.evaluate(potential, 0.3, 0.4, 0.5), atol=3e-12)

    def test_grid_methods_reject_jax_time_including_tracers(self) -> None:
        potential = _field(JaxPotential)
        for call in (potential.evaluate_grid,):
            with self.subTest(method=call.__name__):
                with self.assertRaisesRegex(TypeError, "requires NumPy time"):
                    call(self.jax.numpy.asarray(0.3))
                with self.assertRaisesRegex(TypeError, "requires NumPy time"):
                    self.jax.jit(call)(0.3)
                host_value = call(np.asarray(self.jax.numpy.asarray(0.3)))
                for value in host_value if isinstance(host_value, tuple) else (host_value,):
                    self.assertIsInstance(value, np.ndarray)

    def test_serialization_rebuilds_splines_and_empty_device_caches(self) -> None:
        potential = _field(JaxPotential)
        x = self.jax.numpy.asarray(0.4)
        expected = potential.evaluate(0.3, x, 0.5)
        self.assertTrue(potential._jax_buffers)
        restored = pickle.loads(pickle.dumps(potential))
        self.assertFalse(restored.mean.flags.writeable)
        self.assertEqual(restored._jax_buffers, {})
        self.assertNotIn("_spline_data", restored.__dict__)
        self.assertIsInstance(restored, JaxPotential)
        self.assertIsInstance(restored.evaluate(0.3, 0.4, 0.5), self.jax.Array)
        actual = restored.evaluate(0.3, x, 0.5)
        np.testing.assert_array_equal(expected, actual)

    def test_cached_evaluation_rejects_disabled_float64_and_preserves_numpy(self) -> None:
        potential = _field(JaxPotential)
        x = self.jax.numpy.asarray(0.4)
        potential.evaluate(0.3, x, 0.5).block_until_ready()
        self.jax.config.update("jax_enable_x64", False)
        try:
            with self.assertRaisesRegex(RuntimeError, "requires float64"):
                potential.evaluate(0.3, x, 0.5)
            self.assertIsInstance(Potential.evaluate(potential, 0.3, 0.4, 0.5), np.ndarray)
        finally:
            self.jax.config.update("jax_enable_x64", True)

    def test_gpu_selection_fails_explicitly_without_affecting_scipy(self) -> None:
        from execution._jax import resolve_device

        potential = _field(JaxPotential)
        with patch.object(self.jax, "devices", side_effect=RuntimeError("Unavailable")):
            with self.assertRaisesRegex(RuntimeError, "gpu.*unavailable"):
                resolve_device(ExecutionOptions(backend="jax", device="gpu"))
        self.assertIsInstance(Potential.evaluate(potential, 0.3, 0.4, 0.5), np.ndarray)


if __name__ == "__main__":
    unittest.main()
