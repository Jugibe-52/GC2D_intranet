"""Time rows at recording, solver and versioned archive boundaries."""

from hashlib import sha256
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from diagnostics.io.persistence import load_solution, save_solution
from dynamics.coordinates import GCCoordinates, FCCoordinates
from states import States


class TimeRowStatesTests(unittest.TestCase):
	def test_rows_follow_times_for_both_models_and_selection(self):
		for layout in (GCCoordinates(), FCCoordinates()):
			with self.subTest(layout=type(layout).__name__):
				times = np.array([.1, .3, .8])
				rows = np.arange(3 * 2 * layout.state_dimension, dtype=float).reshape(3, -1)
				result = States.prepare(times, rows[0], layout=layout)
				for index, (time, row) in enumerate(zip(times, rows, strict=True)):
					result.record(time, row)
					self.assertEqual(result.states.shape, (index + 1, rows.shape[1]))
					for actual_time, actual_row in zip(result.t, result.states, strict=True):
						np.testing.assert_array_equal(actual_row, rows[np.flatnonzero(times == actual_time)[0]])
				result.finalize()
				self.assertIs(result.states, result.values)
				self.assertTrue(result.states.flags.c_contiguous)
				np.testing.assert_array_equal(result.select([0, 2]).states, rows[[0, 2]])
				np.testing.assert_array_equal(result.select(1).states, rows[1:2])
				np.testing.assert_array_equal(result.positions()[0], rows[:, :2].T)
				self.assertEqual(result.particle_count, 2)
				for restored in (result.with_diagnostics({'tag': 'rows'}), pickle.loads(pickle.dumps(result))):
					np.testing.assert_array_equal(restored.states, rows)
					self.assertFalse(restored.states.flags.writeable)

	def test_legacy_pickle_restores_component_columns_as_time_rows(self):
		from states.history import _restore_states
		rows = np.array([[1., 2.], [3., 4.]])
		class LegacyState:
			def __reduce__(self):
				return (_restore_states, (np.array([0., 1.]), rows.T, {}, GCCoordinates(), None))
		loaded = pickle.loads(pickle.dumps(LegacyState()))
		np.testing.assert_array_equal(loaded.states, rows)
		self.assertFalse(loaded.states.flags.writeable)

	def test_constructor_and_bulk_recording_do_not_infer_orientation(self):
		rows = np.array([[1., 2.], [3., 4.], [5., 6.]])
		times = np.array([0., .5, 1.])
		result = States(times, rows, layout=GCCoordinates())
		np.testing.assert_array_equal(result.states[1], [3., 4.])
		with self.assertRaises(ValueError):
			States(times, rows.T, layout=GCCoordinates())
		for invalid in (rows.T, np.full_like(rows, np.nan)):
			result = States.prepare(times, rows[0], layout=GCCoordinates())
			with self.assertRaises(ValueError): result.record_history(invalid)
			self.assertEqual(result.states.shape, (0, 2))
			result.record_history(rows)
			result.finalize()
			np.testing.assert_array_equal(result.states, rows)

	def test_new_and_legacy_archives_use_version_even_for_square_arrays(self):
		for dimension in (2, 4):
			for samples in (1, dimension, dimension + 1):
				with self.subTest(dimension=dimension, samples=samples), tempfile.TemporaryDirectory() as tmp:
					layout = GCCoordinates() if dimension == 2 else FCCoordinates()
					rows = np.arange(samples * dimension, dtype=float).reshape(samples, dimension)
					result = States(np.arange(samples, dtype=float), rows, layout=layout)
					path = Path(tmp) / 'archive'
					save_solution(result, path, metadata={'case': 'row-order'})
					with np.load(path / 'solution.npz') as saved:
						arrays = {key: saved[key] for key in saved.files}
						np.testing.assert_array_equal(saved['states'], rows)
					metadata = json.loads((path / 'metadata.json').read_text())
					manifest = json.loads((path / 'manifest.json').read_text())
					self.assertEqual(metadata['schema_version'], 2)
					np.testing.assert_array_equal(load_solution(path).solution.states, rows)
					# Reproduce a schema-1 writer, including its checksums.
					arrays['states'] = rows.T
					np.savez_compressed(path / 'solution.npz', **arrays)
					metadata['schema_version'] = manifest['schema_version'] = 1
					(path / 'metadata.json').write_text(json.dumps(metadata))
					for name in manifest['files']:
						payload = (path / name).read_bytes()
						manifest['files'][name] = {'bytes': len(payload), 'sha256': sha256(payload).hexdigest()}
					(path / 'manifest.json').write_text(json.dumps(manifest))
					loaded = load_solution(path).solution
					np.testing.assert_array_equal(loaded.states, rows)
					self.assertIsNone(loaded.initial_state)
					manifest['schema_version'] = 2
					(path / 'manifest.json').write_text(json.dumps(manifest))
					with self.assertRaisesRegex(ValueError, 'versions disagree'): load_solution(path)
