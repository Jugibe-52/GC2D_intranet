"""Order, backend, diagnostic reduction and sampling contracts of the campaign."""
import unittest
import numpy as np
from states.initial import InitialConditions
from methods.runge_kutta.rk2 import RK2
from methods.runge_kutta.implicit_midpoint import ImplicitMidpoint
from execution._jax_step import prepare_step
from studies.main_campaign import CampaignConfig
from diagnostics.studies.main_campaign import compact_diagnostics
from states.history import States
from dynamics.gc import GuidingCenterDynamics


class MainCampaignTests(unittest.TestCase):
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

    def test_simpson_and_signed_per_particle_block_reduction(self):
        times = np.arange(9)*.025
        rates = np.column_stack((times**3, 2*times**3))
        mu = np.arange(32).reshape(8,4)*.01-1
        energy = np.column_stack((times[::4]**4/4, times[::4]**4/2)).T
        diagnostics = dict(step_count=8, step_sizes=np.diff(times), output_interpolation_count=0,
            energy_time_derivative_start=rates[:-1], energy_time_derivative_end=rates[1:],
            physical_hamiltonian=energy, projection_multipliers=mu)
        solution = States(layout=GuidingCenterDynamics.coordinates, t=times[::4], states=np.zeros((3,4)), diagnostics=diagnostics)
        compact = compact_diagnostics(solution, 4).diagnostics
        np.testing.assert_allclose(compact['energy_integral'], energy, atol=1e-19, rtol=1e-14)
        np.testing.assert_allclose(compact['energy_balance'], 0., atol=1e-19)
        np.testing.assert_array_equal(compact['projection_multipliers_sampled'], mu[[3,7]].reshape(2,2,2))
        norms = np.abs(mu.reshape(8,2,2)).max(axis=1)
        np.testing.assert_array_equal(compact['projection_particle_block_max'], norms.reshape(2,4,2).max(axis=1))
        self.assertNotIn('projection_multipliers', compact)

    def test_quadrature_convergence(self):
        # Known time-only H: any residual is quadrature error, not orbit error.
        errors = []
        for steps in (40, 80, 160):
            times = np.arange(steps+1)/steps
            stride = steps//10
            rates = np.cos(times)[:,None]
            energy = np.sin(times[::stride])[None,:]
            solution = States(layout=GuidingCenterDynamics.coordinates, t=times[::stride], states=np.zeros((11,2)),
                diagnostics=dict(step_count=steps, step_sizes=np.diff(times), output_interpolation_count=0,
                energy_time_derivative_start=rates[:-1], energy_time_derivative_end=rates[1:], physical_hamiltonian=energy))
            errors.append(abs(compact_diagnostics(solution,stride).diagnostics['energy_balance'][0,-1]))
        self.assertTrue(all(15 < a/b < 17 for a,b in zip(errors,errors[1:])))

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
        for kwargs in ({'cycles':0}, {'cycles':True}, {'steps_per_cycle':41}, {'steps_per_cycle':30}):
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
