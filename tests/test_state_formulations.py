"""Four state layouts, passive energy and unchanged physical solver decisions."""

from dataclasses import replace

from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
import unittest
from types import SimpleNamespace
import numpy as np
from contracts.dynamics.protocols import DynamicalSystem, HamiltonianSystem
from dynamics import FullCyclotronDynamics

from simulation import (
    ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42, DOP853, Radau,
    ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Midpoint, BM4Implicit,
    InitialValueProblem, SimulationRequest, simulate, SingleFormulation, DoubleFormulation,
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
    def test_hamiltonian_contract_requires_dynamics_and_both_energy_operations(self):
        source = RotatingHamiltonian()
        members = {name: getattr(source, name) for name in (
            'state_dimension', 'vector_field', 'hamiltonian', 'extended_momentum_derivative')}
        self.assertIsInstance(source, DynamicalSystem)
        self.assertIsInstance(source, HamiltonianSystem)
        for missing in members:
            with self.subTest(missing=missing):
                partial = SimpleNamespace(**{name: value for name, value in members.items()
                                             if name != missing})
                self.assertNotIsInstance(partial, HamiltonianSystem)

    def test_partial_energy_capabilities_are_accepted_only_without_tracking(self):
        source = _problem()
        request = SimulationRequest.uniform(t_span=(.3, .32),  sample_count=2)

        def unused_energy(time, state):
            """Fail if an untracked formulation evaluates either energy operation."""
            raise AssertionError('Untracked integration must not evaluate energy.')

        for energy_members in ({}, {'hamiltonian': unused_energy},
                               {'extended_momentum_derivative': unused_energy}):
            dynamics = SimpleNamespace(
                state_dimension=source.dynamics.state_dimension,
                vector_field=source.dynamics.vector_field,
                particle_vector_field_jacobians=source.dynamics.particle_vector_field_jacobians,
                **energy_members)
            problem = InitialValueProblem(dynamics, source.initial_state, layout=GuidingCenterDynamics.coordinates)
            self.assertIsInstance(dynamics, DynamicalSystem)
            self.assertNotIsInstance(dynamics, HamiltonianSystem)
            for cls in CLASSICAL + DUPLICATED:
                with self.subTest(method=cls.__name__, energy_members=tuple(energy_members)):
                    plain = simulate(problem, cls(track_energy=False, **({"max_step": 0.02} if cls in (DOP853, Radau) else {"step": 0.02})), request)
                    self.assertEqual(plain.states.shape, (2, 2))
                    self.assertNotIn('physical_hamiltonian', plain.diagnostics)
                    with self.assertRaisesRegex(TypeError, 'HamiltonianSystem'):
                        cls(track_energy=True, **({"max_step": 0.02} if cls in (DOP853, Radau) else {"step": 0.02})).new_run(problem, request)

    def test_all_methods_have_actual_dimensions_and_passive_energy(self):
        source = _problem()
        request = SimulationRequest.uniform(t_span=(.3, .38),  sample_count=9)
        for particles in (1, 3):
            problem = InitialValueProblem(source.dynamics, GuidingCenterDynamics.coordinates.pack_components(np.linspace(1., 1.2, particles), np.linspace(1.2, 1.4, particles)), layout=GuidingCenterDynamics.coordinates)
            for cls in CLASSICAL + DUPLICATED:
                with self.subTest(method=cls.__name__, particles=particles):
                    plain = simulate(problem, cls(**({"max_step": .02} if cls in (DOP853, Radau) else {"step": .02})), request)
                    events = []
                    method = cls(track_energy=True, step_observer=events.append)
                    run = replace(method, **({"max_step": .02} if isinstance(method, (DOP853, Radau)) else {"step": .02})).new_run(problem, request)
                    spatial = (4 if cls in DUPLICATED else 2)*particles
                    self.assertEqual(run.initial_state.size, 2 * particles)
                    np.testing.assert_array_equal(run.initial_momentum, np.zeros(particles))
                    self.assertEqual(plain.diagnostics['accepted_internal_state_dimension'], 2 * particles)
                    self.assertEqual(plain.diagnostics['workspace_state_dimension'], spatial)
                    tracked = simulate(problem, replace(method, **({"max_step": .02} if isinstance(method, (DOP853, Radau)) else {"step": .02})), request)
                    np.testing.assert_array_equal(plain.states, tracked.states)
                    for key in ('step_times', 'nonlinear_iterations', 'residual_evaluations',
                                'nonlinear_residual_norms', 'nonlinear_tolerances',
                                'function_evaluations', 'jacobian_evaluations', 'lu_decompositions'):
                        if key in plain.diagnostics:
                            np.testing.assert_array_equal(plain.diagnostics[key], tracked.diagnostics[key])
                    np.testing.assert_array_equal(tracked.diagnostics['extended_time'], np.broadcast_to(tracked.t, (particles, 9)))
                    self.assertEqual(tracked.diagnostics['accepted_internal_state_dimension'], 2 * particles)
                    self.assertEqual(tracked.diagnostics['momentum_state_dimension'], particles)
                    self.assertEqual(tracked.diagnostics['extended_momentum'].shape, (particles, 9))
                    self.assertEqual(tracked.diagnostics['extended_momentum_normalization'], 'physical_kappa')
                    balance = tracked.diagnostics['physical_hamiltonian'] + tracked.diagnostics['extended_momentum']
                    np.testing.assert_array_equal(balance-balance[:, :1], tracked.diagnostics['generalized_energy_error'])
                    for event in events:
                        self.assertEqual(event.state_before.shape, (2*particles,))
                        if hasattr(event, 'multiplier'):
                            self.assertEqual(event.multiplier.shape, (2*particles,))
                    if cls not in (DOP853, Radau):
                        step = run.advance(.3, run.initial_state, .02, momentum=run.initial_momentum)
                        self.assertEqual(step.state.size, 2 * particles)
                        self.assertEqual(step.momentum.shape, (particles,))

    def test_adaptive_auxiliary_does_not_change_grid_jacobian_or_work(self):
        problem = InitialValueProblem(RotatingHamiltonian(), np.array([1., .2], dtype=float, copy=True), layout=GuidingCenterDynamics.coordinates)
        for cls in (DOP853, Radau):
            options = {'jacobian': lambda t,z: np.array([[0.,-1.],[1.,0.]])} if cls is Radau else {}
            for count in (2, 29):
                request = SimulationRequest.uniform(t_span=(.3, 1.),  sample_count=count)
                plain = simulate(problem, cls(max_step=.13, **options), request)
                tracked = simulate(problem, cls(max_step=.13, track_energy=True, **options), request)
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

    def test_full_cyclotron_keeps_four_physical_coordinates_and_separate_momentum(self):
        dynamics = FullCyclotronDynamics(_problem().dynamics.potential, rho=.2, eta=.1)
        problem = InitialValueProblem(dynamics, FullCyclotronDynamics.coordinates.pack_components([1., 1.1, 1.2], [1.2, 1.3, 1.4], [.1, .2, .3], [-.1, -.2, -.3]), layout=FullCyclotronDynamics.coordinates)
        request = SimulationRequest.uniform(t_span=(.3, .34),  sample_count=5)
        run = RK4(step=0.01, track_energy=True).new_run(problem, request)
        self.assertEqual(run.initial_state.shape, (12,))
        self.assertEqual(run.initial_momentum.shape, (3,))
        tracked = simulate(problem, RK4(step=0.01, track_energy=True), request)
        plain = simulate(problem, RK4(step=0.01), request)
        np.testing.assert_array_equal(tracked.states, plain.states)
        self.assertEqual(tracked.diagnostics['accepted_internal_state_dimension'], 12)
        np.testing.assert_array_equal(tracked.diagnostics['extended_time'], np.broadcast_to(tracked.t, (3, 5)))

    def test_bm4_tracking_agrees_with_independent_stage_observer(self):
        from diagnostics import GCGeneralizedEnergyObserver
        problem = _problem()
        request = SimulationRequest.uniform(t_span=(.3,.4),sample_count=6)
        for solver in ('newton','broyden'):
            events=[]
            method=BM4Implicit(track_energy=True, nonlinear_solver=solver, step_observer=events.append)
            result=simulate(problem,replace(method, step=(0.4 - 0.3) / 5),request)
            # The independent diagnostic reconstructs accepted spatial stages.
            obs=GCGeneralizedEnergyObserver(problem.dynamics, initial_state=problem.initial_state, initial_time=.3)
            for event in events: obs(event)
            np.testing.assert_allclose(result.diagnostics['extended_momentum'][0], [r.kappa for r in obs.records], atol=1e-14, rtol=1e-12)
