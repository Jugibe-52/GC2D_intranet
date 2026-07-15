import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import ArrayLike

from classes import Potential, PotentialSystem
from pyhamsys import HamSys, OdeSolution, solve_ivp_symp, solve_ivp_sympext
from pyhamsys.solvers import _step_count


def identity_flow(h: float, t: float, y: np.ndarray) -> np.ndarray:
    return y


class SolveIvpSympTests(unittest.TestCase):
    def test_accepts_array_like_initial_state(self) -> None:
        sol = solve_ivp_symp(
            identity_flow,
            identity_flow,
            (0.0, 0.1),
            [1.0, 0.0],
            step=0.01,
        )

        self.assertIsInstance(sol, OdeSolution)
        self.assertEqual(sol.y.shape, (2, 11))
        self.assertTrue(np.issubdtype(sol.y.dtype, np.floating))
        np.testing.assert_array_equal(sol.y[:, -1], [1.0, 0.0])

    def test_samples_nonuniform_times_without_including_t0(self) -> None:
        requested = np.array([0.07, 0.23, 0.4])
        sol = solve_ivp_symp(
            identity_flow,
            identity_flow,
            (0.0, 0.4),
            [1.0, 0.0],
            step=0.03,
            t_eval=requested,
        )

        np.testing.assert_array_equal(sol.t, requested)
        self.assertEqual(sol.y.shape, (2, 3))
        self.assertLessEqual(sol.step, 0.03)

    def test_accepts_one_requested_time(self) -> None:
        sol = solve_ivp_symp(
            identity_flow,
            identity_flow,
            (0.0, 1.0),
            [1.0],
            step=0.1,
            t_eval=[0.5],
        )

        np.testing.assert_array_equal(sol.t, [0.5])
        self.assertEqual(sol.y.shape, (1, 1))

    def test_preserves_positional_t_eval_compatibility(self) -> None:
        sol = solve_ivp_symp(
            identity_flow,
            identity_flow,
            (0.0, 1.0),
            [1.0],
            0.1,
            [0.0, 0.5, 1.0],
        )

        np.testing.assert_array_equal(sol.t, [0.0, 0.5, 1.0])

    def test_zero_length_span_returns_initial_state(self) -> None:
        sol = solve_ivp_symp(
            identity_flow,
            identity_flow,
            (2.0, 2.0),
            [3.0, 4.0],
            step=0.1,
        )

        np.testing.assert_array_equal(sol.t, [2.0])
        np.testing.assert_array_equal(sol.y[:, 0], [3.0, 4.0])
        self.assertEqual(sol.step, 0.1)

    def test_callback_runs_after_each_internal_step(self) -> None:
        callback_times: list[float] = []
        solve_ivp_symp(
            identity_flow,
            identity_flow,
            (0.0, 0.1),
            [1.0],
            step=0.03,
            command=lambda t, y: callback_times.append(t),
        )

        np.testing.assert_allclose(callback_times, [0.03, 0.06, 0.09, 0.1])

    def test_save_step_controls_storage_independently(self) -> None:
        sol = solve_ivp_symp(
            identity_flow,
            identity_flow,
            (0.0, 0.25),
            [1.0],
            step=0.03,
            save_step=0.1,
        )

        np.testing.assert_allclose(sol.t, [0.0, 0.1, 0.2, 0.25])
        self.assertEqual(sol.y.shape, (1, 4))
        self.assertEqual(sol.n_steps, 10)
        self.assertEqual(sol.requested_step, 0.03)
        self.assertLessEqual(sol.max_step, sol.requested_step)
        self.assertLessEqual(sol.min_step, sol.max_step)

    def test_save_step_matches_an_equivalent_regular_t_eval(self) -> None:
        def drift(h: float, t: float, y: np.ndarray) -> np.ndarray:
            q, p = y
            return np.array([q + h * (p + 0.1 * t), p])

        def kick(h: float, t: float, y: np.ndarray) -> np.ndarray:
            q, p = y
            return np.array([q, p - h * (q - 0.05 * t)])

        requested = 2 * np.pi * np.arange(4)
        by_save_step = solve_ivp_symp(
            drift,
            kick,
            (0.0, requested[-1]),
            [1.0, 0.25],
            step=0.1,
            save_step=2 * np.pi,
        )
        by_t_eval = solve_ivp_symp(
            drift,
            kick,
            (0.0, requested[-1]),
            [1.0, 0.25],
            step=0.1,
            t_eval=requested,
        )

        np.testing.assert_array_equal(by_save_step.t, requested)
        np.testing.assert_array_equal(by_save_step.t, by_t_eval.t)
        np.testing.assert_array_equal(by_save_step.y, by_t_eval.y)
        self.assertEqual(by_save_step.n_steps, by_t_eval.n_steps)

    def test_save_step_and_t_eval_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            solve_ivp_symp(
                identity_flow,
                identity_flow,
                (0.0, 1.0),
                [1.0],
                step=0.1,
                save_step=0.2,
                t_eval=[0.0, 1.0],
            )

    def test_integration_reaches_tf_after_last_requested_sample(self) -> None:
        callback_times: list[float] = []
        sol = solve_ivp_symp(
            identity_flow,
            identity_flow,
            (0.0, 1.0),
            [1.0],
            step=0.25,
            t_eval=[0.0, 0.2],
            command=lambda t, y: callback_times.append(t),
        )

        np.testing.assert_array_equal(sol.t, [0.0, 0.2])
        self.assertAlmostEqual(callback_times[-1], 1.0)
        self.assertEqual(sol.n_steps, 5)

    def test_step_count_uses_float_precision_not_relative_tolerance(self) -> None:
        self.assertEqual(_step_count(1_000_000_000_000.4, 1.0), 1_000_000_000_001)

    def test_rejects_invalid_steps(self) -> None:
        for step in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    solve_ivp_symp(
                        identity_flow,
                        identity_flow,
                        (0.0, 1.0),
                        [1.0],
                        step=step,
                    )

    def test_rejects_invalid_save_steps(self) -> None:
        for save_step in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(save_step=save_step):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    solve_ivp_symp(
                        identity_flow,
                        identity_flow,
                        (0.0, 1.0),
                        [1.0],
                        step=0.1,
                        save_step=save_step,
                    )

    def test_rejects_nonfinite_or_nonnumeric_initial_state(self) -> None:
        for y0 in ([np.nan], [np.inf], ["not-a-number"]):
            with self.subTest(y0=y0):
                with self.assertRaises(ValueError):
                    solve_ivp_symp(
                        identity_flow,
                        identity_flow,
                        (0.0, 1.0),
                        y0,
                        step=0.1,
                    )

    def test_rejects_invalid_evaluation_times(self) -> None:
        invalid_values: tuple[ArrayLike, ...] = (
            [],
            0.5,
            [0.2, 0.1],
            [-0.1, 0.2],
            [0.1, np.nan],
        )
        for t_eval in invalid_values:
            with self.subTest(t_eval=t_eval):
                with self.assertRaises(ValueError):
                    solve_ivp_symp(
                        identity_flow,
                        identity_flow,
                        (0.0, 1.0),
                        [1.0],
                        step=0.1,
                        t_eval=t_eval,
                    )

    def test_rejects_a_flow_that_changes_state_shape(self) -> None:
        def wrong_shape(h: float, t: float, y: np.ndarray) -> np.ndarray:
            return np.append(y, 0.0)

        with self.assertRaisesRegex(ValueError, "preserve the shape"):
            solve_ivp_symp(
                wrong_shape,
                wrong_shape,
                (0.0, 0.1),
                [1.0],
                step=0.1,
            )

    def test_extended_solver_uses_refactored_solver(self) -> None:
        system = HamSys(1)
        setattr(system, "y_dot", lambda t, y: np.array([y[1], -y[0]]))

        sol = solve_ivp_sympext(
            system,
            (0.0, 0.1),
            np.array([1.0, 0.0]),
            step=0.01,
            t_eval=[0.05, 0.1],
        )

        np.testing.assert_array_equal(sol.t, [0.05, 0.1])
        self.assertEqual(sol.y.shape, (2, 2))


class PotentialSystemAreaTests(unittest.TestCase):
    @staticmethod
    def make_system(*, periodic: bool = False) -> PotentialSystem:
        period = 2 * np.pi
        coordinates = np.linspace(0.0, period, 8, endpoint=not periodic)
        potential = Potential(
            coordinates,
            coordinates,
            [np.zeros((8, 8)), None],
            freqs=[],
            xy_period=period if periodic else None,
            k=3,
        )
        return PotentialSystem(potential, {'type': 'gc', 'rho': 0.0, 'eta': 0.0})

    def test_square_initial_conditions_are_centred_and_counter_clockwise(self) -> None:
        system = self.make_system()

        initial = system.guiding_center_square_initial_conditions(side=1.0)
        x, y = system.get_positions(initial)

        centre_x = (system.xmin + system.xmax) / 2
        centre_y = (system.ymin + system.ymax) / 2
        np.testing.assert_allclose(x, [centre_x, centre_x + 1, centre_x + 1, centre_x])
        np.testing.assert_allclose(y, [centre_y, centre_y, centre_y + 1, centre_y + 1])

    def test_hamiltonian_preserves_trajectory_and_time_axes(self) -> None:
        coordinates = np.linspace(0.0, 2 * np.pi, 8)
        system = PotentialSystem(
            Potential(
                coordinates,
                coordinates,
                [np.ones((8, 8)), None],
                freqs=[],
                k=3,
            ),
            {'type': 'gc', 'rho': 0.0, 'eta': 0.0},
        )
        states = np.array([
            [1.0, 1.2, 1.4],
            [2.0, 2.2, 2.4],
            [3.0, 3.2, 3.4],
            [4.0, 4.2, 4.4],
        ])

        energy = system.hamiltonian(np.array([0.0, 0.5, 1.0]), states)

        self.assertEqual(energy.shape, (2, 3))

    def test_area_element_is_preserved_by_shear(self) -> None:
        system = self.make_system()
        solution = SimpleNamespace(
            t=np.array([0.0, 0.5, 1.0]),
            y=np.array([
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [1.0, 1.1, 1.2],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.2, 1.2, 1.2],
            ]),
        )

        area = system.guiding_center_area_element(solution)

        np.testing.assert_allclose(area, 0.02)

    def test_area_element_uses_minimum_periodic_displacement(self) -> None:
        system = self.make_system(periodic=True)
        self.assertIsNotNone(system.xy_period)
        period = cast(float, system.xy_period)
        solution = SimpleNamespace(
            t=np.array([0.0, 1.0]),
            y=np.array([
                [period - 0.05, period - 0.05],
                [0.05, 0.05],
                [period - 0.05, period - 0.05],
                [period - 0.05, period - 0.05],
                [period - 0.05, period - 0.05],
                [0.15, 0.15],
            ]),
        )

        area = system.guiding_center_area_element(solution)

        np.testing.assert_allclose(area, 0.02)

    def test_polygon_area_is_preserved_by_shear(self) -> None:
        system = self.make_system()
        solution = SimpleNamespace(
            t=np.array([0.0, 0.5, 1.0]),
            y=np.array([
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.25, 2.5],
                [1.0, 1.25, 1.5],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]),
        )

        area = system.guiding_center_polygon_area(solution)

        np.testing.assert_allclose(area, 1.0)

    def test_square_boundary_can_use_more_than_four_points(self) -> None:
        system = self.make_system()

        initial = system.guiding_center_square_initial_conditions(
            side=1.0,
            lower_left=(1.0, 1.0),
            points_per_side=3,
        )
        x, _ = system.get_positions(initial)

        self.assertEqual(x.size, 12)

    def test_area_animation_is_created(self) -> None:
        system = self.make_system()
        solution = SimpleNamespace(
            t=np.array([0.0, 0.5, 1.0]),
            y=np.array([
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [1.0, 1.1, 1.2],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.2, 1.2, 1.2],
            ]),
        )

        animation = system.animate_electric_psi_area_conservation(
            solution,
            frame_stride=2,
            step=1,
            repeat=False,
        )
        setattr(animation, '_draw_was_started', True)

        self.assertIsInstance(animation, FuncAnimation)

    def test_polygon_area_animation_is_created(self) -> None:
        system = self.make_system()
        initial = system.guiding_center_square_initial_conditions(
            side=1.0,
            lower_left=(1.0, 1.0),
            points_per_side=2,
        )
        solution = SimpleNamespace(
            t=np.array([0.0, 1.0]),
            y=np.column_stack((initial, initial)),
        )

        animation = system.animate_electric_psi_area_conservation(
            solution,
            step=1,
            repeat=False,
        )
        setattr(animation, '_draw_was_started', True)

        self.assertIsInstance(animation, FuncAnimation)

    def test_area_animation_rejects_zero_initial_area(self) -> None:
        system = self.make_system()
        solution = SimpleNamespace(
            t=np.array([0.0, 1.0]),
            y=np.array([
                [1.0, 1.0],
                [1.1, 1.1],
                [1.2, 1.2],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
        )

        with self.assertRaisesRegex(ValueError, 'must be non-zero'):
            system.animate_electric_psi_area_conservation(solution)


if __name__ == "__main__":
    unittest.main()
