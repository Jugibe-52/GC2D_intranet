"""Order, backend, diagnostic reduction and sampling contracts of the campaign."""
import unittest
import numpy as np
from states.initial import InitialConditions
from methods.runge_kutta.rk2 import RK2
from methods.runge_kutta.implicit_midpoint import ImplicitMidpoint
from execution._jax_step import prepare_step
from studies.main_campaign import CampaignConfig, METHODS, OPTIONS, prepare_case, validate_case
from diagnostics.studies.main_campaign import compact_diagnostics
from states.history import States
from dynamics.gc import GuidingCenterDynamics


class MainCampaignTests(unittest.TestCase):
    def test_requested_rho_reaches_gyroaverage_without_rescaling(self):
        from scipy.special import j0
        from potential.grid import Grid
        from potential.load import GC2DH5Metadata
        from potential.potential import Potential

        grid = Grid.periodic(16, 16, period=1.)
        mean = np.broadcast_to(np.cos(2*np.pi*grid.x[:, None]), (16, 16))
        for scale in (.18, .5):
            provenance = GC2DH5Metadata(source_field_indices=np.array([15]),
                source_x=scale*grid.x, source_y=scale*grid.y,
                source_frequencies=np.array([1.]), characteristic_length=None,
                characteristic_period=None, normalization_factor=1., attributes={},
                spatial_scale=scale, spatial_rescaling=True, temporal_rescaling=True)
            field = Potential(grid, mean=mean, modes=np.zeros((1, 16, 16)),
                frequencies=np.array([1.]), spatial_extension='reflected', metadata=provenance)
            particles = {'particles': [{'position_m': [scale/8, scale/2]} for _ in range(84)]}
            for rho in (0., .3, .5):
                with self.subTest(spatial_scale=scale, rho=rho):
                    problem, _, metadata = prepare_case(CampaignConfig(cycles=1), field,
                        particles, rho, 'RK2', code_sha256='test')
                    # The analytic gyroaverage of cos(2*pi*x) is J0(2*pi*rho)*cos(2*pi*x).
                    np.testing.assert_allclose(problem.dynamics.hamiltonian_potential.mean,
                        j0(2*np.pi*rho)*mean/1.5, atol=1e-14, rtol=1e-13)
                    self.assertEqual(problem.dynamics.rho, rho)
                    self.assertEqual(metadata['rho'], rho)
                    self.assertEqual(metadata['rho_runtime'], rho)
                    self.assertAlmostEqual(metadata['rho_m'], rho*scale)

    def test_changed_code_cannot_resume_a_prepared_campaign(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from studies.main_campaign_io import run_campaign

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'campaign.json').write_text(json.dumps({'code_sha256': 'legacy'}))
            with patch('studies.main_campaign_io.code_digest', return_value='corrected'), \
                 patch('studies.main_campaign_io.load_field') as load, \
                 patch('studies.main_campaign_io.Execution_Modal') as executor:
                with self.assertRaisesRegex(ValueError, 'Code fingerprint changed'):
                    run_campaign(directory, directory, publish=False)
                load.assert_not_called()
                executor.assert_not_called()

    def test_named_recipes_order_and_jax(self):
        import jax
        import jax.numpy as jnp
        initial = InitialConditions.gc(x=[.3, -.4], y=[.6, .2])
        with jax.experimental.enable_x64():
            for recipe, stages in ((RK2, 2), (ImplicitMidpoint, 1)):
                kwargs = {'jacobian': lambda t, z: np.broadcast_to(-np.eye(2), (2, 2, 2))} if recipe is ImplicitMidpoint else {}
                run = recipe().new_run(lambda t, z: -z, initial, **kwargs)
                self.assertEqual(run.metadata['stage_count'], stages)
                errors = []
                for n in (10, 20, 40):
                    state = initial.values
                    for i in range(n):
                        state = run.advance(i/n, state, 1/n).state
                    errors.append(np.max(abs(state - initial.values*np.exp(-1))))
                self.assertTrue(all(3.7 < a/b < 4.4 for a,b in zip(errors, errors[1:])))
                device_kwargs = {'jacobian': lambda t, z: jnp.broadcast_to(-jnp.eye(2), (2, 2, 2))} if recipe is ImplicitMidpoint else {}
                device = recipe().new_run(lambda t, z: -z, initial, **device_kwargs)
                kernel = prepare_step(device, jax.devices('cpu')[0], initial_conditions=initial)
                actual = jax.jit(kernel)(.3, jnp.asarray(initial.values), .1)
                np.testing.assert_allclose(actual[0], run.advance(.3, initial.values, .1).state, atol=2e-14, rtol=0)
                self.assertTrue(actual[2])

    def test_midpoint_rotation_conserves_quadratic_energy(self):
        initial = InitialConditions.gc(x=[1.], y=[0.])
        matrix = np.array([[0., -1.], [1., 0.]])
        run = ImplicitMidpoint().new_run(lambda t,z: matrix@z, initial, jacobian=lambda t,z: matrix[None])
        state = initial.values
        for i in range(100):
            state = run.advance(.2*i, state, .2).state
        self.assertAlmostEqual(float(state@state), 1., places=12)

    def test_direct_energy_and_signed_projection_reduction_for_any_stride(self):
        for stride in (1, 3, 4):
            with self.subTest(stride=stride):
                count = 2*stride
                times = np.arange(count+1)*.025
                mu = np.arange(count*4).reshape(count, 4)*.01-1
                energy = np.column_stack((np.sin(times[::stride]), np.cos(times[::stride]))).T
                derivative = np.column_stack((np.cos(times[::stride]), -np.sin(times[::stride]))).T
                # No endpoint rate arrays are required: only direct sampled histories.
                diagnostics = dict(step_count=count, output_interpolation_count=0,
                    physical_hamiltonian=energy, hamiltonian_time_derivative=derivative,
                    projection_multipliers=mu, energy_integral=np.ones_like(energy),
                    energy_balance=np.ones_like(energy), energy_quadrature_difference=np.ones_like(energy),
                    energy_quadrature='legacy')
                solution = States(layout=GuidingCenterDynamics.coordinates,
                    t=times[::stride], states=np.zeros((3, 4)), diagnostics=diagnostics)
                compact = compact_diagnostics(solution, stride).diagnostics
                np.testing.assert_array_equal(compact['physical_hamiltonian'], energy)
                np.testing.assert_array_equal(compact['hamiltonian_time_derivative'], derivative)
                np.testing.assert_array_equal(compact['energy_drift'], energy-energy[:, :1])
                np.testing.assert_array_equal(compact['projection_multipliers_sampled'],
                    mu[stride-1::stride].reshape(2, 2, 2))
                norms = np.abs(mu.reshape(count, 2, 2)).max(axis=1)
                np.testing.assert_array_equal(compact['projection_particle_norms'], norms[stride-1::stride])
                np.testing.assert_array_equal(compact['projection_particle_block_max'],
                    norms.reshape(2, stride, 2).max(axis=1))
                for key in ('projection_multipliers', 'energy_integral', 'energy_balance',
                            'energy_quadrature_difference', 'energy_quadrature'):
                    self.assertNotIn(key, compact)
                for invalid in (0, -1, True, 1.5):
                    with self.assertRaises(ValueError):
                        compact_diagnostics(solution, invalid)

    def test_all_methods_record_the_same_direct_physical_energy_function(self):
        import jax
        from diagnostics.methods.energy import EnergyObserver
        from potential.grid import Grid
        from potential.load import GC2DH5Metadata
        from potential.potential import Potential
        from simulation.runner import simulate

        grid = Grid.periodic(8, 8, period=1.)
        shape = np.cos(2*np.pi*grid.x[:, None])*np.cos(2*np.pi*grid.y[None, :])
        provenance = GC2DH5Metadata(source_field_indices=np.array([15]),
            source_x=.18*grid.x, source_y=.18*grid.y,
            source_frequencies=np.array([1.]), characteristic_length=None,
            characteristic_period=None, normalization_factor=1., attributes={},
            spatial_scale=.18, spatial_rescaling=True, temporal_rescaling=True)
        field = Potential(grid, mean=.002*shape, modes=(.001+.0004j)*shape[None],
            frequencies=np.array([1.]), spatial_extension='reflected', metadata=provenance)
        particles = {'particles': [{'position_m': [.18*(.15+.6*i/84), .18*.35]} for i in range(84)]}
        config = CampaignConfig(cycles=1, steps_per_cycle=6, samples_per_cycle=2)
        expected_times = np.array([0., .5, 1.])
        for name in METHODS:
            with self.subTest(method=name), jax.experimental.enable_x64():
                problem, method, metadata = prepare_case(config, field, particles, .3,
                    name, code_sha256='test')
                raw = simulate(problem, method, options=OPTIONS, observers=(EnergyObserver(),))
                solution = compact_diagnostics(raw, config.stride)
                validate_case(solution, metadata)
                np.testing.assert_array_equal(solution.t, expected_times)
                expected = problem.dynamics.hamiltonian(solution.t, solution.states.T)
                derivative = problem.dynamics.hamiltonian_time_derivative(solution.t, solution.states.T)
                np.testing.assert_allclose(solution.diagnostics['physical_hamiltonian'], expected,
                    atol=2e-14, rtol=1e-12)
                np.testing.assert_allclose(solution.diagnostics['hamiltonian_time_derivative'], derivative,
                    atol=2e-14, rtol=1e-12)
                self.assertEqual(metadata['schema_version'], 2)
                self.assertEqual(metadata['energy_evaluation'], 'physical_hamiltonian_at_saved_states')
                for key in ('energy_integral', 'energy_balance', 'energy_quadrature_difference', 'energy_quadrature'):
                    self.assertNotIn(key, solution.diagnostics)
                    self.assertNotIn(key, metadata)
                # New records cannot claim direct-only evaluation while retaining an old balance.
                legacy = solution.with_diagnostics({'energy_balance': np.zeros_like(expected)})
                with self.assertRaisesRegex(ValueError, 'without quadrature'):
                    validate_case(legacy, metadata)
                shifted = States(layout=solution.layout, t=solution.t+1e-4,
                    states=solution.states, diagnostics=solution.diagnostics)
                with self.assertRaisesRegex(ValueError, 'common requested times'):
                    validate_case(shifted, metadata)

    def test_poincare_uses_only_whole_cycles_and_period_two(self):
        from visualization.main_campaign import poincare_returns
        c = CampaignConfig(cycles=2)
        states = np.zeros((21,168))
        states[:,0] = np.arange(21)*.15
        metadata = dict(config=dict(cycles=2,steps_per_cycle=40,samples_per_cycle=10),
            method='RK2', initial_positions=np.zeros((84,2)).tolist(), cycle_duration=1., spatial_period=2.)
        diagnostics = dict(step_count=80, output_interpolation_count=0, execution_backend='jax')
        for key in ('physical_hamiltonian','hamiltonian_time_derivative','energy_integral','energy_balance','energy_quadrature_difference'):
            diagnostics[key] = np.zeros((84,21))
        solution = States(layout=GuidingCenterDynamics.coordinates, t=np.arange(21)/10,states=states,diagnostics=diagnostics)
        returns = poincare_returns(solution,metadata)
        self.assertEqual(returns.shape,(3,84,2))
        np.testing.assert_allclose(returns[:,0,0],[0.,1.5,1.])

    def test_recording_controls(self):
        self.assertEqual(CampaignConfig().stride,4)
        self.assertEqual(CampaignConfig(steps_per_cycle=30).stride, 3)
        self.assertEqual(CampaignConfig(steps_per_cycle=10).stride, 1)
        for kwargs in ({'cycles':0}, {'cycles':True}, {'steps_per_cycle':41}, {'samples_per_cycle':80}):
            with self.assertRaises(ValueError):
                CampaignConfig(**kwargs)

    def test_finalization_requires_complete_cases_and_keeps_billing_unknown(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from studies.main_campaign_io import finalize_campaign

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = dict(rhos=[0.], methods=['RK2'], code_sha256='same', bucket='unused')
            (directory/'campaign.json').write_text(json.dumps(record))
            report = dict(status='running', cases={})
            (directory/'progress.json').write_text(json.dumps(report))
            with patch('studies.main_campaign_io.ArtifactStore') as store, \
                 patch('visualization.main_campaign.export_campaign', return_value=directory/'site') as export:
                with self.assertRaises(ValueError):
                    finalize_campaign(directory)
                store.assert_not_called()
                export.assert_not_called()
                report.update(status='complete', cases={'rho-0.00-RK2': dict(status='complete',
                    rho=0., method='RK2', call_id='original', worker_started_unix_seconds=10.,
                    worker_finished_unix_seconds=16., worker_wall_seconds=6.)},
                    cost_estimated_requested_resources_usd=.001,
                    cost_estimated_limit_resources_usd=.002, cost_billed_status='unavailable')
                (directory/'progress.json').write_text(json.dumps(report))
                self.assertEqual(finalize_campaign(directory), directory/'site')
                metrics = json.loads((directory/'execution-metrics.json').read_text())
                self.assertEqual(metrics['worker_span_seconds'], 6.)
                self.assertEqual(metrics['sum_worker_wall_seconds'], 6.)
                self.assertIsNone(metrics['cost_billed_usd'])
                store.return_value.put.assert_called_once()
                export.assert_called_once()

    def test_local_poll_timeout_recovers_original_call_and_saved_archive(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from studies.main_campaign_io import run_campaign

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = dict(config=dict(cycles=2, steps_per_cycle=40, samples_per_cycle=10),
                particles={}, rhos=[0.], methods=['RK2'], code_sha256='same', app_name='test')
            (directory/'campaign.json').write_text(json.dumps(record))
            receipt = directory/'receipt.json'
            solution = SimpleNamespace(diagnostics=dict(worker_wall_seconds=2.,
                cost_estimated_requested_resources_usd=.001, execution_call_id='original-call'))
            submitted, recovered = MagicMock(), MagicMock()
            submitted.last_record_path = receipt
            recovered.run.return_value = solution

            def submit(*args, **kwargs):
                receipt.write_text('{"call_id":"original-call"}')
                raise TimeoutError('Local polling expired; remote execution continues.')

            def save(solution, path, **kwargs):
                path.mkdir(parents=True)
                (path/'manifest.json').write_text('{}')

            submitted.run.side_effect = submit
            with patch('studies.main_campaign_io.code_digest', return_value='same'), \
                 patch('studies.main_campaign_io.load_field'), \
                 patch('studies.main_campaign_io.prepare_case', return_value=(object(), object(), {})), \
                 patch('studies.main_campaign_io.validate_case'), \
                 patch('studies.main_campaign_io.save_solution', side_effect=save), \
                 patch('studies.main_campaign_io.load_solution', return_value=SimpleNamespace(solution=solution, metadata={})), \
                 patch('studies.main_campaign_io.Execution_Modal', return_value=submitted) as executor:
                executor.from_record.return_value = recovered
                report = run_campaign(directory, directory, max_workers=1, publish=False)
                self.assertEqual(report['status'], 'saved_local')
                self.assertEqual(report['cases']['rho-0.00-RK2']['call_id'], 'original-call')
                executor.assert_called_once()
                executor.from_record.assert_called_once_with(receipt)
                recovered.run.assert_called_once()
                executor.reset_mock()
                second = run_campaign(directory, directory, max_workers=1, publish=False)
                executor.assert_not_called()
                executor.from_record.assert_not_called()
                self.assertEqual(second['cost_estimated_requested_resources_usd'], .001)

if __name__ == '__main__':
    unittest.main()
