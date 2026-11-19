"""Direct-radius backgrounds and exact case selection never reintegrate returns."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics.io.persistence import save_solution
from dynamics.gc import GuidingCenterDynamics
from potential.grid import Grid
from potential.load import GC2DH5Metadata
from potential.potential import Potential
from states.history import States
from studies.main_campaign import CampaignConfig, SOURCE_SHA256, campaign_case_key, prepare_case
from visualization.main_campaign import export_campaign
from visualization.main_campaign_background import sample_campaign_background


def field_and_case(rho=.0025):
    grid = Grid.periodic(16, 16, period=1.)
    pattern = np.cos(2*np.pi*grid.x[:, None])*np.cos(2*np.pi*grid.y[None, :])
    provenance = GC2DH5Metadata(source_field_indices=np.array([15]),
        source_x=.18*grid.x, source_y=.18*grid.y,
        source_frequencies=np.array([1.]), characteristic_length=None,
        characteristic_period=None, normalization_factor=1., attributes={},
        spatial_scale=.18, spatial_rescaling=True, temporal_rescaling=True)
    field = Potential(grid, mean=.002*pattern, modes=(.001+.0004j)*pattern[None],
        frequencies=np.array([1.]), spatial_extension='reflected', metadata=provenance)
    particles = dict(particles=[dict(id=i+1, position_m=[.036, .054], color='#335577')
        for i in range(84)], families=[])
    problem, _, metadata = prepare_case(CampaignConfig(cycles=2), field, particles,
        rho, 'RK2', code_sha256='frozen-numerical-code')
    return field, problem, metadata


class CampaignBackgroundTests(unittest.TestCase):
    def test_all_phases_match_the_actual_hamiltonian_and_drift(self):
        field, problem, metadata = field_and_case()
        with patch('simulation.runner.simulate', side_effect=AssertionError('No trajectory integration')):
            sampled = sample_campaign_background(field, metadata, grid_size=8, vector_grid_size=5)
        self.assertEqual(sampled['potential'].shape, (51, 8, 8))
        self.assertEqual(sampled['velocity'].shape, (51, 5, 5, 2))
        self.assertEqual(sampled['trajectoryIntegrations'], 0)
        q = 2*(np.arange(8)+.5)/8
        xx, yy = np.meshgrid(q, q)
        vq = 2*(np.arange(5)+.5)/5
        vx, vy = np.meshgrid(vq, vq)
        state = np.r_[vx.ravel(), vy.ravel()]
        for phase in (0, 7, 25, 49, 50):
            t = phase/50
            np.testing.assert_allclose(sampled['potential'][phase],
                problem.dynamics.hamiltonian_potential.evaluate(t, xx, yy), atol=1e-16)
            np.testing.assert_allclose(sampled['velocity'][phase],
                problem.dynamics.vector_field(t, state).reshape(2, 5, 5).transpose(1, 2, 0), atol=1e-16)
        for name in ('potential', 'velocity'):
            np.testing.assert_array_equal(sampled[name][-1], sampled[name][0])
        self.assertEqual(sampled['bounds'], dict(x=0., y=0., span=2.))
        self.assertEqual(sampled['minimum'], float(sampled['potential'].min()))
        self.assertEqual(sampled['maximum'], float(sampled['potential'].max()))
        self.assertEqual(sampled['maxSpeed'], float(np.linalg.norm(sampled['velocity'], axis=-1).max()))
        self.assertEqual(sampled['rhoRuntime'], .0025)

    def test_reflected_tiles_keep_the_actual_velocity_orientation(self):
        field, _, metadata = field_and_case()
        sampled = sample_campaign_background(field, metadata, grid_size=8, vector_grid_size=8)
        h, v = sampled['potential'], sampled['velocity']
        np.testing.assert_allclose(h[:, :, ::-1], h, atol=1e-16)
        np.testing.assert_allclose(v[:, :, ::-1, 0], v[:, :, :, 0], atol=1e-15)
        np.testing.assert_allclose(v[:, :, ::-1, 1], -v[:, :, :, 1], atol=1e-15)
        _, _, changed_metadata = field_and_case(.025)
        changed = sample_campaign_background(field, changed_metadata, grid_size=8, vector_grid_size=8)
        self.assertFalse(np.array_equal(changed['potential'], h))
        self.assertFalse(np.array_equal(changed['velocity'], v))

    def test_old_radius_and_changed_physics_are_rejected(self):
        field, _, metadata = field_and_case()
        for changes in [dict(schema_version=1), dict(rho_runtime=.0001), dict(rho_m=.0001),
                dict(magnetic_field=2.), dict(spatial_period=1.), dict(time_scale_seconds=2.),
                dict(source_field_indices=[1])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                sample_campaign_background(field, dict(metadata, **changes), grid_size=8, vector_grid_size=5)

    def test_export_keeps_exact_small_radius_keys_and_lazy_hashed_fields(self):
        rhos = [.0025, .005, .0175]
        field, _, metadata = field_and_case()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = dict(config=metadata['config'], rhos=rhos, methods=['RK2'],
                particles=metadata['particle_record'], source_sha256=SOURCE_SHA256,
                code_sha256=metadata['code_sha256'])
            (directory/'campaign.json').write_text(json.dumps(record))
            for rho in rhos:
                _, problem, case_metadata = field_and_case(rho)
                states = np.repeat(problem.initial_state[None], 21, axis=0)
                diagnostics = dict(step_count=80, output_interpolation_count=0,
                    execution_backend='jax', energy_evaluation='physical_hamiltonian_at_saved_states',
                    physical_hamiltonian=np.zeros((84, 21)), hamiltonian_time_derivative=np.zeros((84, 21)),
                    energy_drift=np.zeros((84, 21)))
                solution = States(layout=GuidingCenterDynamics.coordinates,
                    t=np.arange(21)/10, states=states, diagnostics=diagnostics)
                save_solution(solution, directory/'runs'/campaign_case_key(rho, 'RK2'), metadata=case_metadata)
            with patch('visualization.main_campaign.load_field', return_value=field), \
                    patch('simulation.runner.simulate', side_effect=AssertionError('No integration')):
                output = export_campaign(directory, source=Path('verified-source.h5'),
                    field_grid_size=8, vector_grid_size=5)
            data = json.loads((output/'data.json').read_text())
            self.assertEqual([option['label'] for option in data['rhoOptions']], ['0.0025', '0.005', '0.0175'])
            for option in data['rhoOptions']:
                self.assertEqual(option['cases']['RK2'], campaign_case_key(option['value'], 'RK2'))
                background = data['fields'][option['fieldKey']]
                payload = (output/background['asset']).read_bytes()
                self.assertEqual(len(payload), background['bytes'])
                self.assertEqual(sha256(payload).hexdigest(), background['sha256'])
                values = np.frombuffer(payload, dtype='<f8')
                self.assertEqual(values.size, 51*8*8 + 51*5*5*2)
                self.assertEqual(background['potentialValues'], 51*8*8)
                self.assertLess(background['bytes'], 25*1024*1024)
                self.assertTrue(background['phaseEndpointsVerified'])
            html = (output/'index.html').read_text()
            self.assertIn('option.cases[panel.method]', html)
            self.assertNotIn('toFixed(2)', html)
            self.assertIn('Potentiel effectif Hρ et champ GC', html)
            self.assertIn('Cycle initial', html)
            self.assertIn('ctx.arc(sx,sy,5', html)
            self.assertEqual(len(list((output/'assets').glob('*.bin'))), 6)


if __name__ == '__main__':
    unittest.main()
