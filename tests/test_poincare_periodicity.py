"""Analytical periodic maps and saved-data integrity checks."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from studies.poincare_periodicity import (
    analyze_poincare_periodicity, analyze_short_periodicity,
    load_saved_poincare_section, section_distances,
)
from visualization.poincare_periodicity import (
    export_interval_recurrence_viewer, export_phase_class_selector,
)


class PoincarePeriodicityTests(unittest.TestCase):
    def test_exact_period_four_and_stationary_particle(self):
        n = np.arange(201)
        z = np.zeros((201, 2, 2))
        z[:, 0, 0] = (n / 4) % 1
        z[:, 1] = .3
        result = analyze_poincare_periodicity(z, [7, 1], max_lag=20)
        self.assertEqual([r['best_q'] for r in result.summary], [4, 1])
        self.assertEqual(result.summary[0]['max_over_L'], 0)
        self.assertEqual(result.summary[0]['peak_frequency'], .25)
        self.assertIsNone(result.summary[1]['peak_frequency'])
        self.assertEqual([r['smallest_q'] for r in result.thresholds], [4, 4, 4, 1, 1, 1])

    def test_minimum_image_across_boundary(self):
        a = np.array([[[.99, .1], [.2, .99]]])
        b = np.array([[[.01, .1], [.2, .01]]])
        np.testing.assert_allclose(section_distances(a, b), [[.02, .02]], atol=1e-15)

    def test_detuned_rotation_has_no_exact_period(self):
        # For uniform torus translation the q-lag defect is known analytically.
        alpha = np.sqrt(2) / 10
        z = np.zeros((301, 1, 2))
        z[:, 0, 0] = (np.arange(301) * alpha) % 1
        result = analyze_poincare_periodicity(z, [3], max_lag=20,
                                              threshold_fractions=(1e-6,))
        expected = np.abs((result.lags * alpha + .5) % 1 - .5)
        np.testing.assert_allclose(result.metrics[:, 0, 0], expected, atol=1e-14)
        np.testing.assert_allclose(result.metrics[:, 0, 2], expected, atol=1e-14)
        self.assertIsNone(result.thresholds[0]['smallest_q'])

    def test_initial_return_does_not_imply_sequence_repetition(self):
        z = np.zeros((101, 1, 2))
        z[6:, 0, 0] = (np.arange(95) * .137) % 1
        result = analyze_poincare_periodicity(z, [5], max_lag=10,
                                              threshold_fractions=(1e-4,))
        self.assertEqual(result.summary[0]['closest_initial_distance_over_L'], 0)
        self.assertIsNone(result.thresholds[0]['smallest_q'])
        self.assertEqual(result.summary[0]['overlap_pairs'], 101 - result.summary[0]['best_q'])

    def test_search_cap_and_invalid_controls(self):
        z = np.zeros((101, 1, 2))
        self.assertEqual(analyze_poincare_periodicity(z, [1], max_lag=1000).lags[-1], 20)
        for kwargs in ({'minimum_repetitions': 2}, {'max_lag': 1.5},
                       {'threshold_fractions': (0,)}, {'threshold_fractions': (np.nan,)},
                       {'threshold_fractions': (.01, .01)}):
            with self.assertRaises(ValueError):
                analyze_poincare_periodicity(z, [1], **kwargs)

    def test_short_rhythm_and_fixed_pattern_are_distinct(self):
        n = np.arange(4001)
        z = np.empty((len(n), 2, 2))
        # Both recur near eight cycles, but only one repeats at integer multiples.
        for j, period in enumerate((8., 7.8)):
            phase = 2*np.pi*n/period
            z[:, j] = .5 + .08*np.column_stack((np.cos(phase), np.sin(phase)))
        result = analyze_short_periodicity(z, [5, 1], block_cycles=400)
        locked, detuned = result.summary
        self.assertEqual([r['short_q'] for r in result.summary], [8, 8])
        self.assertGreater(locked['heldout_template_score'], .999999)
        self.assertLess(detuned['heldout_template_score'], .02)
        self.assertAlmostEqual(locked['mean_rotation_cycles'], 8., places=5)
        self.assertAlmostEqual(detuned['mean_rotation_cycles'], 7.8, places=3)
        repeated = [r for r in result.multiples if r['particle'] == 5]
        self.assertLess(max(r['max_over_L'] for r in repeated), 1e-12)
        other = [r for r in result.multiples if r['particle'] == 1]
        self.assertGreater(other[9]['rms_over_L'], 5*other[0]['rms_over_L'])

    def test_short_clock_crosses_cell_boundary_and_handles_stationary_data(self):
        n = np.arange(801)
        phase = 2*np.pi*n/8
        z = np.zeros((len(n), 2, 2))
        z[:, 0] = (.98 + .08*np.column_stack((np.cos(phase), np.sin(phase)))) % 1
        z[:, 1] = .5
        result = analyze_short_periodicity(z, [7, 3], block_cycles=200)
        self.assertEqual(result.summary[0]['short_q'], 8)
        self.assertAlmostEqual(result.summary[0]['mean_rotation_cycles'], 8., places=5)
        self.assertGreater(result.summary[0]['heldout_template_score'], .999999)
        self.assertIsNone(result.summary[1]['short_q'])
        self.assertIsNone(result.summary[1]['mean_rotation_cycles'])

    def test_short_period_controls_and_disjoint_blocks(self):
        n = np.arange(401)
        phase = 2*np.pi*n/7
        z = (.5 + .1*np.column_stack((np.cos(phase), np.sin(phase))))[:, None]
        result = analyze_short_periodicity(z, [1], block_cycles=100)
        self.assertEqual([b['start_cycle'] for b in result.blocks], [0, 100, 200, 300])
        self.assertEqual([b['stop_cycle_exclusive'] for b in result.blocks], [100, 200, 300, 400])
        self.assertEqual(result.summary[0]['short_q'], 7)
        for kwargs in ({'candidate_max_lag': 80}, {'max_lag': 400},
                       {'peak_height': float('nan')}, {'block_cycles': 500},
                       {'threshold_fractions': (0,)}):
            with self.assertRaises(ValueError):
                analyze_short_periodicity(z, [1], **kwargs)

    def test_phase_class_selector_embeds_all_panels_and_samples(self):
        n = np.arange(101)
        z = np.empty((len(n), 2, 2))
        for j, period in enumerate((8, 7)):
            phase = 2*np.pi*n/period
            z[:, j] = .5 + .08*np.column_stack((np.cos(phase), np.sin(phase)))
        result = analyze_short_periodicity(z, [1, 3], max_lag=40,
                                           candidate_max_lag=10, block_cycles=50)
        section = type('Section', (), {'positions': z, 'particle_ids': (1, 3)})()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_phase_class_selector(Path(tmp) / 'selector.html', section, result,
                                               title='Test <selector>')
            html = path.read_text()
        self.assertNotIn('__PHASE_CLASS_DATA__', html)
        self.assertIn('Test \\u003cselector>', html)
        self.assertIn('"particle":1,"q":8,"samples":101', html)
        self.assertIn('"particle":3,"q":7,"samples":101', html)
        self.assertIn('Show all classes', html)
        self.assertIn('Visible classes:', html)

    def test_phase_class_selector_rejects_missing_candidate(self):
        z = np.full((101, 1, 2), .5)
        result = analyze_short_periodicity(z, [1], max_lag=40,
                                           candidate_max_lag=10, block_cycles=50)
        section = type('Section', (), {'positions': z, 'particle_ids': (1,)})()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'no integer short-period candidate'):
                export_phase_class_selector(Path(tmp) / 'selector.html', section, result,
                                            title='No candidate')

    def test_interval_viewer_embeds_linked_particle_panels(self):
        n = np.arange(101)
        z = np.empty((len(n), 2, 2))
        z[:, 0] = .5 + .1*np.column_stack((np.cos(n), np.sin(n)))
        z[:, 1] = .5 + .2*np.column_stack((np.cos(n/2), np.sin(n/2)))
        result = analyze_poincare_periodicity(z, [1, 3], max_lag=20)
        section = type('Section', (), {
            'positions': z, 'particle_ids': (1, 3), 'colors': ('#112233', '#445566'),
        })()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_interval_recurrence_viewer(
                Path(tmp) / 'interval.html', section, result, title='Long <returns>',
            )
            html = path.read_text()
        self.assertNotIn('__POINCARE_INTERVAL_DATA__', html)
        self.assertIn('Long \\u003creturns>', html)
        self.assertIn('"particle":1,"color":"#112233","samples":101', html)
        self.assertIn('"particle":3,"color":"#445566","samples":101', html)
        self.assertIn('Start cycle', html)
        self.assertIn('End cycle', html)
        self.assertIn("source.endsWith('Range')", html)
        self.assertIn('cursor: grab', html)
        self.assertIn('spatial points total', html)

    def test_loader_preserves_original_ids_and_rejects_missing_or_corrupt_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'assets').mkdir()
            run = root / 'resultados' / 'test'
            run.mkdir(parents=True)
            original = dict(particles=[dict(particle=p, x=p / 10, y=.5,
                                            radius_over_L=p / 10, color='#112233') for p in (1, 7)])
            manifest = root / 'assets/original_particles.json'
            manifest.write_text(json.dumps(original))
            meta = dict(run_id='test', particle_ids=[1, 7], cycles=12, t0=0, cycle_duration=1,
                        original_particles_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        field_provenance={'grid': {'period': 1}})
            (run / 'metadata.json').write_text(json.dumps(meta))
            rows = [dict(particle=p, cycle=n, time_normalized=n, x_over_L=p / 10,
                         y_over_L=.5, x_wrapped=p / 10, y_wrapped=.5,
                         initial_radius_over_L=p / 10, color='#112233')
                    for n in range(1, 13) for p in (1, 7)]

            def write_records(records):
                with (run / 'positions_after_each_cycle.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(records)
                complete = dict(run_id='test', sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                    for name in ('metadata.json', 'positions_after_each_cycle.csv')})
                (run / 'COMPLETE.json').write_text(json.dumps(complete))

            write_records(rows[::-1])
            section = load_saved_poincare_section(root, 'test', [7, 1])
            self.assertEqual(section.particle_ids, (7, 1))
            np.testing.assert_allclose(section.positions[0, :, 0], [.7, .1])
            np.testing.assert_allclose(section.positions[-1, :, 0], [.7, .1])
            write_records(rows[:-1])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                load_saved_poincare_section(root, 'test', [7, 1])
            write_records(rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                load_saved_poincare_section(root, 'test', [7, 1])
            write_records(rows)
            (run / 'positions_after_each_cycle.csv').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                load_saved_poincare_section(root, 'test', [7, 1])


if __name__ == '__main__':
    unittest.main()
