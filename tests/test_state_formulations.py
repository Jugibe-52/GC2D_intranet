"""Four state layouts, passive energy and unchanged physical solver decisions."""
import unittest
import numpy as np
from dynamics import FullCyclotronDynamics
from initial_conditions import GCInitialConfiguration, FCInitialConfiguration
from simulation import (
    ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42, DOP853, Radau,
    ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Midpoint, BM4Implicit,
    InitialValueProblem, SimulationRequest, simulate, PhysicalFormulation, DoubledFormulation,
)
from tests.test_method_integration import _problem

CLASSICAL = (ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42, DOP853, Radau)
DUPLICATED = (ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Midpoint, BM4Implicit)


class RotatingHamiltonian:
    """Time-only energy term stresses the auxiliary without altering spatial rotation."""
    state_dimension = 2

    def vector_field(self, time, state):
        x, y = np.split(state, 2)
        return np.r_[-y, x]

    def hamiltonian(self, time, state):
        x, y = np.split(state, 2, axis=0)
        return (x*x + y*y)/2 + 1e6*np.asarray(time)**5

    def extended_momentum_derivative(self, time, state):
        return np.full(state.size // 2, -5e6*time**4)


class FourStateFormulationTests(unittest.TestCase):
    def test_all_methods_have_actual_dimensions_and_passive_energy(self):
        source = _problem()
        request = SimulationRequest.uniform(t_span=(.3, .38), max_step=.02, sample_count=9)
        for particles in (1, 3):
            problem = InitialValueProblem(source.dynamics, GCInitialConfiguration.from_components(
                x=np.linspace(1., 1.2, particles), y=np.linspace(1.2, 1.4, particles)))
            for cls in CLASSICAL + DUPLICATED:
                with self.subTest(method=cls.__name__, particles=particles):
                    plain = simulate(problem, cls(), request)
                    events = []
                    method = cls(track_energy=True, step_observer=events.append)
                    run = method.new_run(problem, request)
                    spatial = (4 if cls in DUPLICATED else 2)*particles
                    self.assertEqual(run.initial_state.size, spatial + 2 * particles)
                    np.testing.assert_array_equal(run.state_formulation.time(run.initial_state), np.full(particles, .3))
                    self.assertEqual(plain.diagnostics['accepted_internal_state_dimension'], spatial)
                    tracked = simulate(problem, method, request)
                    np.testing.assert_array_equal(plain.states, tracked.states)
                    for key in ('step_times', 'nonlinear_iterations', 'residual_evaluations',
                                'nonlinear_residual_norms', 'nonlinear_tolerances',
                                'function_evaluations', 'jacobian_evaluations', 'lu_decompositions'):
                        if key in plain.diagnostics:
                            np.testing.assert_array_equal(plain.diagnostics[key], tracked.diagnostics[key])
                    np.testing.assert_array_equal(tracked.diagnostics['extended_time'], np.broadcast_to(tracked.t, (particles, 9)))
                    self.assertEqual(tracked.diagnostics['accepted_internal_state_dimension'], spatial + 2 * particles)
                    self.assertEqual(tracked.diagnostics['extended_momentum'].shape, (particles, 9))
                    self.assertEqual(tracked.diagnostics['extended_momentum_normalization'], 'physical_kappa')
                    balance = tracked.diagnostics['physical_hamiltonian'] + tracked.diagnostics['extended_momentum']
                    np.testing.assert_array_equal(balance-balance[:, :1], tracked.diagnostics['generalized_energy_error'])
                    for event in events:
                        self.assertEqual(event.state_before.shape, (2*particles,))
                        if hasattr(event, 'multiplier'):
                            self.assertEqual(event.multiplier.shape, (2*particles,))
                    if cls not in (DOP853, Radau):
                        step = run.advance(.3, run.initial_state, .02).state
                        self.assertEqual(step.size, spatial + 2 * particles)
                        np.testing.assert_array_equal(run.state_formulation.time(step), np.full(particles, .3 + .02))
                        if cls in DUPLICATED:
                            np.testing.assert_array_equal(step[:2*particles], step[2*particles:4*particles])

    def test_adaptive_auxiliary_does_not_change_grid_jacobian_or_work(self):
        problem = InitialValueProblem(RotatingHamiltonian(), GCInitialConfiguration([1., .2]))
        for cls in (DOP853, Radau):
            options = {'jacobian': lambda t,z: np.array([[0.,-1.],[1.,0.]])} if cls is Radau else {}
            for count in (2, 29):
                request = SimulationRequest.uniform(t_span=(.3, 1.), max_step=.13, sample_count=count)
                plain = simulate(problem, cls(**options), request)
                tracked = simulate(problem, cls(track_energy=True, **options), request)
                np.testing.assert_array_equal(plain.states, tracked.states)
                for key in ('step_times','function_evaluations','jacobian_evaluations','lu_decompositions'):
                    np.testing.assert_array_equal(plain.diagnostics[key], tracked.diagnostics[key])
                exact = -1e6*(tracked.t**5-.3**5)
                np.testing.assert_allclose(tracked.diagnostics['extended_momentum'][0], exact, rtol=2e-15, atol=5e-10)

    def test_no_projected_time_or_momentum_mode_is_silently_reinterpreted(self):
        for cls in DUPLICATED[:-1]:
            with self.subTest(method=cls.__name__):
                with self.assertRaisesRegex(ValueError, 'track_energy=True'):
                    cls(state_extension='fully_extended')

    def test_formulation_objects_expose_four_distinct_named_configurations(self):
        problem = _problem()
        expected = ((PhysicalFormulation,False,2,'physical'),
                    (PhysicalFormulation,True,4,'physical_with_energy'),
                    (DoubledFormulation,False,4,'duplicated'),
                    (DoubledFormulation,True,6,'duplicated_with_energy'))
        for cls, tracking, size, name in expected:
            form = cls(problem,.3,tracking)
            self.assertEqual(form.dimension,size)
            self.assertEqual(form.metadata()['state_formulation'],name)
            self.assertEqual(form.initial_state.size,size)

    def test_uniform_components_keep_particle_coordinates_together(self):
        source = _problem()
        problem = InitialValueProblem(source.dynamics, GCInitialConfiguration.from_components(
            x=[1., 2., 3.], y=[4., 5., 6.]))
        for cls in (PhysicalFormulation, DoubledFormulation):
            for tracking in (False, True):
                form = cls(problem, .3, tracking)
                state = form.pack(problem.initial_state, .4, np.array([7., 8., 9.]))
                components = form.components(state)
                expected = np.tile([[1., 2., 3.], [4., 5., 6.]], (form.copies, 1))
                if tracking:
                    expected = np.vstack((expected, [.4, .4, .4], [7., 8., 9.]))
                np.testing.assert_array_equal(components, expected)

    def test_formulation_aligns_roundoff_and_rejects_a_divergent_particle_time(self):
        source = _problem()
        problem = InitialValueProblem(source.dynamics, GCInitialConfiguration.from_components(
            x=[1., 1.1, 1.2], y=[1.2, 1.3, 1.4]))
        times = np.array([.3, .32, .34])
        for cls in (PhysicalFormulation, DoubledFormulation):
            form = cls(problem, .3, True)
            physical = np.repeat(problem.initial_state[:, None], times.size, axis=1)
            history = form.pack(physical, np.nextafter(times, np.inf), np.zeros((3, 3)))
            before = history.copy()
            _, diagnostics = form.extract_history(times, history)
            np.testing.assert_array_equal(diagnostics['extended_time'], np.broadcast_to(times, (3, 3)))
            np.testing.assert_array_equal(history, before)
            form.time(history)[1, 1] += 1e-4
            with self.assertRaisesRegex(ValueError, 'Each particle time'):
                form.extract_history(times, history)

    def test_full_cyclotron_keeps_six_coordinates_per_particle_with_energy(self):
        dynamics = FullCyclotronDynamics(_problem().dynamics.potential, rho=.2, eta=.1)
        problem = InitialValueProblem(dynamics, FCInitialConfiguration.from_components(
            x=[1., 1.1, 1.2], y=[1.2, 1.3, 1.4], vx=[.1, .2, .3], vy=[-.1, -.2, -.3]))
        request = SimulationRequest.uniform(t_span=(.3, .34), max_step=.01, sample_count=5)
        run = RK4(track_energy=True).new_run(problem, request)
        self.assertEqual(run.state_formulation.components(run.initial_state).shape, (6, 3))
        tracked = simulate(problem, RK4(track_energy=True), request)
        plain = simulate(problem, RK4(), request)
        np.testing.assert_array_equal(tracked.states, plain.states)
        self.assertEqual(tracked.diagnostics['accepted_internal_state_dimension'], 18)
        np.testing.assert_array_equal(tracked.diagnostics['extended_time'], np.broadcast_to(tracked.t, (3, 5)))

    def test_bm4_tracking_agrees_with_independent_stage_observer(self):
        from diagnostics import GCGeneralizedEnergyObserver
        problem = _problem()
        request = SimulationRequest.uniform(t_span=(.3,.4),max_step=(.4-.3)/5,sample_count=6)
        for solver in ('newton','broyden'):
            events=[]
            method=BM4Implicit(track_energy=True, nonlinear_solver=solver, coupling_frequency=.4, step_observer=events.append)
            result=simulate(problem,method,request)
            # The independent diagnostic reconstructs accepted spatial stages.
            obs=GCGeneralizedEnergyObserver(problem.dynamics, initial_state=problem.initial_state, initial_time=.3)
            for event in events: obs(event)
            np.testing.assert_allclose(result.diagnostics['extended_momentum'][0], [r.kappa for r in obs.records], atol=1e-14, rtol=1e-12)
