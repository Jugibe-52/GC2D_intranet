"""Numerical reductions stay independent of immutable archive records."""

import subprocess
import sys
import unittest

import numpy as np

from contracts.comparison import AdaptiveReference, EnergyAccuracySeries
from contracts.study_results import ParallelBM4RecurrenceConfig, ParallelBM4RecurrenceResult
from diagnostics.studies.comparison import (
	time_integrated_rms_floor, final_rms_floor, rms_error,
	maximum_absolute_error, running_maximum_absolute_error,
)
from diagnostics.studies.bm4_recurrence import recurrence_distances
from studies._validation import require_uncoupled_bm4


class ComparisonDiagnosticsTests(unittest.TestCase):
	"""Use known values and nonuniform time weights to check the moved operations."""

	def test_reference_floor_uses_saved_time_weights(self) -> None:
		reference = AdaptiveReference(
			np.array([0., 1., 3.]), np.zeros((4, 3)), np.zeros((4, 3)),
			np.array([[3., 0., 6.], [4., 0., 8.]]), 1., 1., 1, 1,
		)
		self.assertAlmostEqual(time_integrated_rms_floor(reference), np.sqrt(18.75))
		self.assertAlmostEqual(final_rms_floor(reference), np.sqrt(50.))
		self.assertFalse(reference.audit_distances.flags.writeable)
		np.testing.assert_array_equal(reference.audit_distances, [[3., 0., 6.], [4., 0., 8.]])

	def test_signed_energy_errors_reduce_over_particles_without_mutation(self) -> None:
		series = EnergyAccuracySeries('RK4', np.array([[3., 0., -6.], [-4., 0., 8.]]))
		rms = rms_error(series.errors)
		np.testing.assert_allclose(rms, [np.sqrt(12.5), 0., np.sqrt(50.)])
		np.testing.assert_array_equal(maximum_absolute_error(series.errors), [4., 0., 8.])
		np.testing.assert_array_equal(running_maximum_absolute_error(series.errors), [4., 4., 8.])
		rms[0] = -1.
		np.testing.assert_array_equal(series.errors, [[3., 0., -6.], [-4., 0., 8.]])
		self.assertFalse(series.errors.flags.writeable)

	def test_periodic_recurrence_keeps_independent_particles_and_read_only_output(self) -> None:
		config = ParallelBM4RecurrenceConfig(
			particle_count=2, t_span=(0., 2.), steps_per_cycle=1,
			saved_samples_per_cycle=1, worker_count=1, coupling_frequency=.3,
		)
		result = ParallelBM4RecurrenceResult(
			config, np.array([0., 1., 2.]), np.array([[9., 1.], [2., 2.]]),
			np.array([[[9., 1., 8.], [1., 1., 3.]], [[2., 2., 12.], [2., 7., 2.]]]),
			np.ones(2), np.zeros(2, dtype=int), np.zeros(2), np.zeros(2, dtype=int),
			np.zeros(2), 1.,
		)
		distances = recurrence_distances(result, period=10.)
		np.testing.assert_allclose(distances, [[0., 2., np.sqrt(5.)], [0., 5., 0.]])
		self.assertFalse(distances.flags.writeable)
		self.assertFalse(np.shares_memory(distances, result.positions))
		for period in (0., -1., float('inf'), float('nan'), True):
			with self.subTest(period=period), self.assertRaises(ValueError):
				recurrence_distances(result, period=period)
		# Legacy metadata remains readable, but new coupled work is still rejected.
		with self.assertRaisesRegex(ValueError, 'no longer supports'):
			require_uncoupled_bm4(config.coupling_frequency)
		for rate in (None, 0.):
			require_uncoupled_bm4(rate)

	def test_record_imports_do_not_require_diagnostics_or_studies(self) -> None:
		script = """
import builtins
original = builtins.__import__
def without_analysis(name, *args, **kwargs):
    if name.split('.')[0] in {'diagnostics', 'studies', 'simulation'}:
        raise ImportError('Records must not require analysis: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = without_analysis
from contracts.comparison import AdaptiveReference, EnergyAccuracySeries
from contracts.study_results import ParallelBM4RecurrenceResult
assert not hasattr(AdaptiveReference, 'final_rms_floor')
assert not hasattr(EnergyAccuracySeries, 'rms_error')
assert not hasattr(ParallelBM4RecurrenceResult, 'recurrence_distances')
"""
		completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
		self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
	unittest.main()
