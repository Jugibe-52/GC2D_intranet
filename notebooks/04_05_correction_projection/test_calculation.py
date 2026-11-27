"""Focused scientific-reduction and no-duplicate-submission regression tests."""
from contextlib import nullcontext
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import calculation_runtime as runtime


def arrays():
    """Use signed (−3,4) multipliers to distinguish Euclidean/infinity norms."""
    mu = np.empty((10, 2, 84), dtype=np.float64)
    mu[:, 0], mu[:, 1] = -3., 4.
    ends = np.arange(1, 41, dtype=np.float64) / 40
    return dict(t=np.r_[0., ends[3::4]], states=np.zeros((11, 168)),
        projection_multipliers_sampled=mu, particle_ids=np.arange(1, 85),
        step_start_times=np.r_[0., ends[:-1]], step_times=ends, step_sizes=np.full(40, .025),
        nonlinear_iterations=np.arange(40) % 4, nonlinear_residual_norms=np.full(40, 5e-11),
        nonlinear_tolerances=np.full(40, 1e-10), projection_multiplier_norms=np.full(40, 4.))


class ScientificReductionTests(unittest.TestCase):
    def test_euclidean_samples_and_every_joint_newton_solve(self):
        summary = runtime.summarize_arrays(arrays(), 1)
        self.assertEqual(summary, dict(sampled_median=5., sampled_maximum=5.,
            newton_mean_iterations=1.5, newton_max_iterations=3, max_residual_to_tolerance=.5))

    def test_reject_unconverged_unsampled_step(self):
        values = arrays()
        values['nonlinear_residual_norms'][0] = 2e-10
        with self.assertRaisesRegex(ValueError, 'Newton'):
            runtime.summarize_arrays(values, 1)

    def test_reject_changed_sampling_link(self):
        values = arrays()
        values['projection_multiplier_norms'][3] = 5.
        with self.assertRaises(AssertionError):
            runtime.summarize_arrays(values, 1)

    def test_launch_settings_reject_ambiguous_radius_selection(self):
        for indices in ([True], [11], [0, 0], []):
            with self.assertRaises(ValueError):
                runtime.check_settings(1, 1, 'test', indices)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.settings = runtime.check_settings(1, 1, 'test', [6])
        self.case = dict(case_key='rho-0.015-ABBA2Implicit', rho=.015, method='ABBA2Implicit', reference_status='complete')
        self.protocol = dict(cases=[self.case])
        self.count = 0
        def spawn(**kwargs):
            self.count += 1
            return SimpleNamespace(object_id='fc-test123')
        self.worker = SimpleNamespace(app=SimpleNamespace(run=lambda **kwargs: nullcontext()),
            calculate=SimpleNamespace(spawn=spawn))
        self.modal = patch.dict('sys.modules', modal=SimpleNamespace(enable_output=lambda: nullcontext()))
        self.modal.start()

    def tearDown(self):
        self.modal.stop()
        self.temp.cleanup()

    def launch(self, **options):
        return runtime.launch_or_recover(self.worker, self.directory, self.settings,
            self.protocol, 'a' * 64, **options)

    def test_launch_is_explicit_and_confirmed_call_is_reused(self):
        with self.assertRaises(RuntimeError):
            self.launch()
        self.assertEqual(self.count, 0)
        first = self.launch(allow_launch=True)
        self.assertEqual(first, self.launch(allow_launch=True))
        self.assertEqual(self.count, 1)
        self.settings['cycles'] = 2
        with self.assertRaisesRegex(ValueError, 'different'):
            self.launch(allow_launch=True)

    def test_ambiguous_submission_never_retries_and_explicit_recovery_works(self):
        def broken(**kwargs):
            self.count += 1
            raise ConnectionError('A submission may have reached the service.')
        self.worker.calculate.spawn = broken
        with self.assertRaises(ConnectionError):
            self.launch(allow_launch=True)
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous'):
            self.launch(allow_launch=True)
        self.assertEqual(self.count, 1)
        receipt = self.launch(resume_calls={self.case['case_key']: 'fc-recovered'})
        self.assertEqual(receipt[self.case['case_key']]['call_id'], 'fc-recovered')
        self.assertEqual(self.count, 1)


class ResultArchiveTests(unittest.TestCase):
    """Exercise malformed payloads using the independently run local smoke case."""
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path('/tmp/gc2d-colab45-validation')
        path = cls.workspace / 'validation-one-cycle/rho-0.015-BM4Implicit.zip'
        if not path.exists():
            raise unittest.SkipTest('Run validate_calculation.py first for the actual integration fixture.')
        cls.data = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(cls.data)) as archive:
            cls.files = {name: archive.read(name) for name in archive.namelist() if name != 'manifest.json'}
        cls.receipt = json.loads(cls.files['result.json'])
        cls.protocol = json.loads(cls.files['protocol.json'])

    def verify(self, data):
        return runtime.verify_case_result(data, self.receipt['case'], self.receipt['settings'],
            self.protocol, self.receipt['protocol_sha256'])

    def test_valid_real_result(self):
        self.assertEqual(self.verify(self.data)['status'], 'complete')

    def test_reject_array_corruption_even_with_new_zip_hashes(self):
        files = dict(self.files)
        with np.load(io.BytesIO(files['arrays.npz']), allow_pickle=False) as values:
            changed = {name: values[name].copy() for name in values.files}
        changed['nonlinear_residual_norms'][0] = 1.
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **changed)
        files['arrays.npz'] = buffer.getvalue()
        with self.assertRaisesRegex(ValueError, 'Newton'):
            self.verify(runtime.pack_files(files))

    def test_explicit_failure_has_no_complete_arrays_or_summary(self):
        files = dict(self.files)
        files.pop('arrays.npz')
        result = dict(self.receipt, status='failed', reason='nonlinear_convergence_failure', first_failed_step=1)
        result.pop('summary')
        files['result.json'] = json.dumps(result).encode()
        self.assertEqual(self.verify(runtime.pack_files(files))['status'], 'failed')
        result['numerical_seconds'] += 1
        files['result.json'] = json.dumps(result).encode()
        with self.assertRaisesRegex(ValueError, 'timing'):
            self.verify(runtime.pack_files(files))

    def test_checksum_change_is_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(self.data)) as source, zipfile.ZipFile(buffer, 'w') as output:
            for name in source.namelist():
                data = source.read(name) + (b' ' if name == 'result.json' else b'')
                output.writestr(name, data)
        with self.assertRaisesRegex(ValueError, 'Damaged'):
            self.verify(buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
