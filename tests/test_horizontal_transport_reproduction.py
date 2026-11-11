"""Historical Poincare provenance and exact geometry reconstruction checks."""

from dataclasses import replace
from hashlib import file_digest
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from potential import Grid, Potential
from studies.dimensional_h5_midpoint import DimensionalH5Field
from studies.horizontal_transport_reproduction import prepare_horizontal_reproduction
from studies.poincare_gap_probes import GapProbeSeeds, build_gap_probes
from studies.poincare_rho_sweep import RhoStarConfig, build_rho_star


class HorizontalReproductionTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		self.source = Path(self.directory.name) / 'field.h5'
		n = 8
		x = .000703125 + np.arange(n) * .18 / n
		y = .0003515625 + np.arange(n) * np.nextafter(.18 / n, np.inf)
		xx, yy = np.meshgrid(np.arange(n) * 2*np.pi/n, np.arange(n) * 2*np.pi/n, indexing='ij')
		fields = np.zeros((16, n, n), dtype=complex)
		fields[0] = .031 * np.cos(xx) * np.cos(yy)
		fields[15] = .007 * np.exp(1j * (xx + yy))
		frequencies = np.arange(16, dtype=float) * 200.
		with h5py.File(self.source, 'w') as archive:
			archive['Rcells'], archive['Zcells'] = x, y
			archive['freqs'], archive['fields'] = frequencies, fields
		with self.source.open('rb') as stream:
			self.digest = file_digest(stream, 'sha256').hexdigest()
		grid = Grid(x[0], y[0], x[1]-x[0], y[1]-y[0], n, n, n*(x[1]-x[0]))
		divisor = frequencies[15] * .06**2 * 1.5 / (2*np.pi)**2
		clock = 2*np.pi/frequencies[15]
		mean = (fields[0].real / divisor) * divisor
		modes = (fields[15:16] / divisor) * divisor
		raw = Potential(grid, mean=mean, modes=modes, frequencies=np.ones(1))
		stream = Potential(grid, mean=mean*(clock/1.5), modes=modes*(clock/1.5), frequencies=np.ones(1))
		self.field = DimensionalH5Field(raw, stream, clock, 1.5, (15,))
		self.config = RhoStarConfig(rho_hat=.3, particles=40, arms=8, method='RK4',
			cycles=5000, steps_per_cycle=50, hamiltonian_convention='radial')

	def test_restores_original_field_and_geometry_and_limits_validation_span(self):
		original = build_rho_star(self.field, self.config, source_sha256=self.digest)
		record = dict(original.metadata)
		record['config'] = {**record['config'], 'source_selection': [0, 1]}
		restored = prepare_horizontal_reproduction(self.source, {'experiment': record}, cycles=2, steps_per_cycle=100)
		np.testing.assert_array_equal(restored.problem.initial_state, original.problem.initial_state)
		np.testing.assert_array_equal(restored.problem.dynamics.potential.mean, original.problem.dynamics.potential.mean)
		np.testing.assert_array_equal(restored.problem.dynamics.potential.modes, original.problem.dynamics.potential.modes)
		self.assertEqual(restored.request.t_span, (0., 2.))
		np.testing.assert_array_equal(restored.request.output_times, (0., 1., 2.))
		self.assertEqual(restored.config.steps_per_cycle, 100)
		self.assertEqual(restored.metadata['historical_reproduction']['original_config']['cycles'], 5000)
		self.assertFalse(restored.metadata['historical_reproduction']['bitwise_trajectory_reproduction_claimed'])

	def test_preserves_literal_probe_ids_and_seeds(self):
		seeds = GapProbeSeeds(fractions=((.4407,.5817),(.4417,.5896),(.6073,.5675),(.603,.5753),
			(.4317,.3856),(.4376,.3814),(.5749,.3911),(.579,.3918)),
			particle_ids=tuple(range(41,49)), gap_names=('nw','nw','ne','ne','sw','sw','se','se'))
		original = build_gap_probes(self.field, replace(self.config, particles=8),
			source_sha256=self.digest, seeds=seeds)
		restored = prepare_horizontal_reproduction(self.source, original.metadata)
		np.testing.assert_array_equal(restored.problem.initial_state, original.problem.initial_state)
		np.testing.assert_array_equal(restored.metadata['particle_id'], range(41,49))
		self.assertEqual(restored.metadata['historical_reproduction']['grouping'], 'eight_probes')

	def test_rejects_changed_source_clock_or_even_tiny_geometry_change(self):
		original = build_rho_star(self.field, self.config, source_sha256=self.digest)
		for key, value, message in (
			('source_sha256', 'wrong', 'fingerprint'),
			('time_unit_seconds', self.field.cycle_seconds * 2, 'clock'),
			('initial_positions', np.nextafter(original.metadata['initial_positions'], np.inf), 'initial_positions'),
		):
			with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
				prepare_horizontal_reproduction(self.source, {**original.metadata, key: value})


if __name__ == '__main__':
	unittest.main()
