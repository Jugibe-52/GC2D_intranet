"""Check that one prepared Hamiltonian generates every GC derivative."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from dynamics import GuidingCenterDynamics
from potential import Potential


def potential() -> Potential:
    """Use a small reproducible periodic field with time dependence."""
    return Potential.random(A=0.08, M=3, nx=16, ny=18, seed=27)


class GCHamiltonianTests(unittest.TestCase):
    def test_derivatives_match_finite_differences_of_the_hamiltonian(self) -> None:
        source = potential()
        # Three particles include coordinates outside the primary periodic cell.
        state = np.array([-0.3, 1.1, 7.2, 0.4, 2.3, -1.4])
        time, delta = 0.31, 1e-5
        for magnetic_field in (1.0, 1.7, -2.3):
            for radius in (0.0, 0.3):
                with self.subTest(B=magnetic_field, rho=radius):
                    dynamics = GuidingCenterDynamics(source, B=magnetic_field, rho=radius)
                    gradients = []
                    jacobian_columns = []
                    for component in range(2):
                        offset = np.zeros_like(state)
                        offset[component * 3:(component + 1) * 3] = delta
                        gradients.append((dynamics.hamiltonian(time, state + offset)
                                          - dynamics.hamiltonian(time, state - offset)) / (2 * delta))
                        jacobian_columns.append(((dynamics.vector_field(time, state + offset)
                                                  - dynamics.vector_field(time, state - offset))
                                                 / (2 * delta)).reshape(2, 3).T)
                    expected_drift = np.concatenate((-gradients[1], gradients[0]))
                    np.testing.assert_allclose(dynamics.vector_field(time, state), expected_drift,
                                               rtol=2e-7, atol=2e-10)
                    np.testing.assert_allclose(dynamics.particle_vector_field_jacobians(time, state),
                                               np.stack(jacobian_columns, axis=-1), rtol=2e-7, atol=2e-10)
                    time_difference = -(dynamics.hamiltonian(time + delta, state)
                                        - dynamics.hamiltonian(time - delta, state)) / (2 * delta)
                    np.testing.assert_allclose(dynamics.extended_momentum_derivative(time, state),
                                               time_difference, rtol=2e-7, atol=2e-10)

    def test_histories_reuse_prepared_splines_without_an_electric_field_call(self) -> None:
        dynamics = GuidingCenterDynamics(potential(), rho=0.3, B=-1.7)
        times = np.array([0.13, 0.31, 0.72])
        history = np.array([[0.4, 0.5, 0.7], [1.1, 1.2, 1.3],
                            [0.6, 0.7, 0.8], [2.2, 2.3, 2.4]])
        splines = dynamics.hamiltonian_potential._splines
        with patch.object(Potential, "electric_field", side_effect=AssertionError("GC used electric_field")), \
                patch.object(Potential, "gyroaverage", side_effect=AssertionError("Repeated gyroaverage")), \
                patch("potential.potential._build_periodic_spline", side_effect=AssertionError("Refitted spline")):
            for name in ("vector_field", "hamiltonian", "extended_momentum_derivative"):
                evaluate = getattr(dynamics, name)
                expected = np.stack([evaluate(t, state) for t, state in zip(times, history.T)], axis=-1)
                np.testing.assert_allclose(evaluate(times, history), expected, rtol=1e-14, atol=1e-14)
            dynamics.particle_vector_field_jacobians(times[0], history[:, 0])
        self.assertIs(dynamics.hamiltonian_potential._splines, splines)


@unittest.skipUnless(importlib.util.find_spec("jax"), "Optional JAX is not installed")
class JaxGCHamiltonianTests(unittest.TestCase):
    def test_compiled_derivatives_and_histories_agree_with_numpy(self) -> None:
        import jax

        previous_x64 = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        self.addCleanup(jax.config.update, "jax_enable_x64", previous_x64)
        times = np.array([0.13, 0.31, 0.72])
        state = np.array([-0.3, 1.1, 7.2, 0.4, 2.3, -1.4])
        history = np.repeat(state[:, None], times.size, axis=1)
        for magnetic_field, radius in ((1.0, 0.0), (-1.7, 0.3)):
            with self.subTest(B=magnetic_field, rho=radius):
                source = GuidingCenterDynamics(potential(), rho=radius, B=magnetic_field)
                prepared = source._as_jax()
                self.assertIs(prepared.hamiltonian_potential._splines, source.hamiltonian_potential._splines)
                with patch.object(Potential, "electric_field", side_effect=AssertionError("GC used electric_field")):
                    for name in ("vector_field", "hamiltonian", "extended_momentum_derivative"):
                        actual = jax.jit(getattr(prepared, name))(times, history)
                        np.testing.assert_allclose(actual, getattr(source, name)(times, history),
                                                   rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(jax.jit(prepared.particle_vector_field_jacobians)(0.31, state),
                                               source.particle_vector_field_jacobians(0.31, state),
                                               rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
