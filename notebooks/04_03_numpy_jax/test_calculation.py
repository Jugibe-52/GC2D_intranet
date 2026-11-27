"""Offline safety checks for input integrity and duplicate-submission recovery."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import calculation_runtime as runtime


class CalculationTests(unittest.TestCase):
    def test_modified_bundle_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'kit'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                runtime.prepare_kit(b'wrong', '0' * 64, destination)
            self.assertFalse(destination.exists())

    def test_unsafe_archive_cannot_escape(self):
        for name, kind in [('../outside', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                ('a/../outside', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w') as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = '/tmp/outside' if kind == tarfile.SYMTYPE else ''
                    archive.addfile(member)
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    runtime.unpack_tar(stream.getvalue(), Path(directory) / 'kit')

    def test_settings_reject_invalid_or_unsafe_runs(self):
        for values in [([], 3, 'ok'), ([91], 3, 'ok'), ([True], 3, 'ok'), ([1, 1], 3, 'ok'),
                ([1], 2, 'ok'), ([1], 3, '../outside'), (1, 3, 'ok')]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                runtime.check_settings(*values)
        self.assertEqual(runtime.check_settings((10, 30, 90), 3, 'reader-run')['cycles'], [10, 30, 90])

    def test_confirmed_receipt_reuses_call_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            settings = runtime.check_settings([1], 3, 'test')
            receipt = dict(submission='confirmed', call_id='fc-test', settings={**settings, 'kit_sha256': 'abc'})
            runtime.write_json(path, receipt)
            worker = Mock()
            self.assertEqual(runtime.launch_or_recover(worker, path, settings, 'abc'), receipt)
            worker.calculate.spawn.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'different settings'):
                runtime.launch_or_recover(worker, path, {**settings, 'cycles': [2]}, 'abc')
            with self.assertRaisesRegex(ValueError, 'different settings'):
                runtime.launch_or_recover(worker, path, settings, 'changed-kit')
            with self.assertRaisesRegex(ValueError, 'differs'):
                runtime.launch_or_recover(worker, path, settings, 'abc', 'fc-other')

    def test_interrupted_submission_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            settings = runtime.check_settings([1], 3, 'test')
            app = SimpleNamespace(run=lambda **kwargs: nullcontext())
            worker = SimpleNamespace(app=app, calculate=SimpleNamespace(spawn=Mock(side_effect=ConnectionError('Interrupted'))))
            fake_modal = SimpleNamespace(enable_output=lambda: nullcontext())
            with patch.dict('sys.modules', {'modal': fake_modal}):
                with self.assertRaises(ConnectionError):
                    runtime.launch_or_recover(worker, path, settings, 'abc')
            self.assertEqual(json.loads(path.read_text())['submission'], 'unconfirmed')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                runtime.launch_or_recover(worker, path, settings, 'abc')
            self.assertEqual(worker.calculate.spawn.call_count, 1)

    def test_resume_records_call_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            worker = Mock()
            receipt = runtime.launch_or_recover(worker, path, runtime.check_settings([1], 3, 'test'), 'abc', 'fc-existing')
            self.assertEqual(receipt['call_id'], 'fc-existing')
            worker.calculate.spawn.assert_not_called()

    def test_invalid_resume_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            with self.assertRaisesRegex(ValueError, 'identifier'):
                runtime.launch_or_recover(Mock(), path, runtime.check_settings([1], 3, 'test'), 'abc', 'wrong')
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
