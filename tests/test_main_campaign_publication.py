"""Explicit partial publication preserves verified data and visible failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics.io.persistence import load_solution, save_solution
from dynamics.gc import GuidingCenterDynamics
from states.history import States
from studies.main_campaign_io import finalize_campaign, publication_cases
from visualization.main_campaign import export_campaign


class CampaignPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        particles = dict(particles=[dict(id=i+1, color='#335577') for i in range(84)], families=[])
        self.record = dict(config=dict(cycles=2, steps_per_cycle=40, samples_per_cycle=10),
            rhos=[0.], methods=['RK2', 'ImplicitMidpoint'], particles=particles,
            code_sha256='frozen-numerical-code', source_sha256='source', bucket='unused')
        self.report = dict(status='failed', cost_billed_status='unavailable', cases={
            'rho-0.00-RK2': dict(status='complete', rho=0., method='RK2',
                worker_started_unix_seconds=10., worker_finished_unix_seconds=16., worker_wall_seconds=6.,
                cost_estimated_requested_resources_usd=.001, cost_estimated_limit_resources_usd=.002),
            'rho-0.00-ImplicitMidpoint': dict(status='failed', rho=0., method='ImplicitMidpoint',
                error='Nonlinear solver did not converge.')})
        self.write_records()
        metadata = dict(config=self.record['config'], rho=0., method='RK2',
            code_sha256=self.record['code_sha256'], particle_record=particles,
            initial_positions=np.zeros((84, 2)).tolist(), cycle_duration=1., spatial_period=2.)
        states = np.zeros((21, 168))
        states[:, 0] = np.arange(21)*.15
        diagnostics = dict(step_count=80, output_interpolation_count=0, execution_backend='jax')
        for key in ('physical_hamiltonian', 'hamiltonian_time_derivative', 'energy_integral',
                    'energy_balance', 'energy_quadrature_difference'):
            diagnostics[key] = np.zeros((84, 21))
        solution = States(layout=GuidingCenterDynamics.coordinates, t=np.arange(21)/10,
            states=states, diagnostics=diagnostics)
        save_solution(solution, self.directory/'runs/rho-0.00-RK2', metadata=metadata)

    def write_records(self):
        (self.directory/'campaign.json').write_text(json.dumps(self.record))
        (self.directory/'progress.json').write_text(json.dumps(self.report))

    def test_partial_export_keeps_missing_panel_and_whole_cycle_data(self):
        with self.assertRaisesRegex(ValueError, 'Missing case'):
            export_campaign(self.directory)
        output = export_campaign(self.directory, allow_partial=True)
        data = json.loads((output/'data.json').read_text())
        self.assertEqual((data['publicationStatus'], data['availableCases'], data['expectedCases']), ('partial', 1, 2))
        self.assertEqual(data['methods'], ['RK2', 'ImplicitMidpoint'])
        missing = data['cases']['rho-0.00-ImplicitMidpoint']
        self.assertEqual(missing['status'], 'unavailable')
        self.assertIn('converge', missing['reason'])
        self.assertNotIn('asset', missing)
        item = data['cases']['rho-0.00-RK2']
        self.assertEqual(item['shape'], [3, 84, 2])
        xy = np.fromfile(output/item['asset'], dtype='<f8').reshape(item['shape'])
        np.testing.assert_allclose(xy[:, 0, 0], [0., 1.5, 1.])
        self.assertEqual(len(list((output/'assets').glob('*.bin'))), 1)
        self.assertEqual(data['rhoConvention'], 'legacy_physical_scale')
        self.assertIn('Historical radius convention', (output/'index.html').read_text())
        self.assertIn('0.06 / (2π)', (output/'index.html').read_text())
        self.assertNotIn('ρ is measured in base-cell widths', (output/'index.html').read_text())

    def test_direct_radius_export_uses_the_recorded_base_cell_width(self):
        from types import SimpleNamespace
        stored = load_solution(self.directory/'runs/rho-0.00-RK2')
        metadata = dict(stored.metadata, schema_version=2, spatial_scale_m=.18,
            rho_runtime=0., rho_m=0., energy_evaluation='physical_hamiltonian_at_saved_states')
        diagnostics = {key: value for key, value in stored.solution.diagnostics.items()
            if key not in ('energy_integral', 'energy_balance', 'energy_quadrature_difference')}
        diagnostics.update(energy_drift=np.zeros((84, 21)),
            energy_evaluation='physical_hamiltonian_at_saved_states')
        current = SimpleNamespace(metadata=metadata,
            solution=stored.solution.with_diagnostics(diagnostics, replace=True))
        with patch('visualization.main_campaign.load_solution', return_value=current):
            output = export_campaign(self.directory, allow_partial=True)
        data = json.loads((output/'data.json').read_text())
        html = (output/'index.html').read_text()
        self.assertEqual(data['rhoConvention'], 'base_cell_width')
        self.assertEqual(data['spatialScaleM'], .18)
        self.assertIn('ρ is measured in base-cell widths', html)
        self.assertIn('ρ<sub>m</sub> = ρ × L, L = 0.18 m.', html)
        self.assertNotIn('0.06 / (2π)', html)
        self.assertNotIn('__RHO_UNITS__', html)

    def test_pending_or_saved_unverified_cases_cannot_be_omitted(self):
        row = self.report['cases']['rho-0.00-ImplicitMidpoint']
        for status in ('running', 'saved_local', 'uploaded_pending_readback'):
            with self.subTest(status=status):
                row['status'] = status
                self.write_records()
                with self.assertRaisesRegex(ValueError, 'not verified'):
                    export_campaign(self.directory, allow_partial=True)
        row['status'] = 'failed'
        self.write_records()
        path = self.directory/'runs/rho-0.00-ImplicitMidpoint'
        path.mkdir()
        (path/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'not verified'):
            export_campaign(self.directory, allow_partial=True)

    def test_partial_export_does_not_hide_corrupt_or_missing_verified_archive(self):
        path = self.directory/'runs/rho-0.00-RK2/solution.npz'
        path.write_bytes(b'corrupt archive')
        with self.assertRaises(ValueError):
            export_campaign(self.directory, allow_partial=True)
        (path.parent/'manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing case'):
            export_campaign(self.directory, allow_partial=True)

    def test_finalization_requires_opt_in_and_records_only_available_costs(self):
        with patch('studies.main_campaign_io.ArtifactStore') as store:
            with self.assertRaises(ValueError):
                finalize_campaign(self.directory)
            store.assert_not_called()
            output = finalize_campaign(self.directory, allow_partial=True)
            metrics = json.loads((self.directory/'execution-metrics-partial-1.json').read_text())
            self.assertEqual(metrics['publication_status'], 'partial')
            self.assertEqual(metrics['available_cases'], 1)
            self.assertEqual(metrics['expected_cases'], 2)
            self.assertEqual(metrics['worker_span_seconds'], 6.)
            self.assertEqual(metrics['cost_estimated_requested_resources_usd'], .001)
            self.assertIsNone(metrics['cost_billed_usd'])
            self.assertEqual(len(metrics['missing_cases']), 1)
            self.assertTrue((output/'index.html').is_file())
            store.return_value.put.assert_called_once()

    def test_partial_publication_rejects_unknown_or_empty_case_sets(self):
        del self.report['cases']['rho-0.00-ImplicitMidpoint']
        self.write_records()
        with self.assertRaises(ValueError):
            publication_cases(self.directory, self.record, allow_partial=True)
        self.report['cases']['rho-0.00-ImplicitMidpoint'] = dict(status='failed', rho=0., method='ImplicitMidpoint')
        self.report['cases']['rho-0.00-RK2']['status'] = 'failed'
        (self.directory/'runs/rho-0.00-RK2/manifest.json').unlink()
        self.write_records()
        with self.assertRaisesRegex(ValueError, 'at least one'):
            publication_cases(self.directory, self.record, allow_partial=True)


if __name__ == '__main__':
    unittest.main()
