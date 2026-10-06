"""Regression contracts for shared study validation and its intentional differences."""

from __future__ import annotations

from dataclasses import replace
import unittest
from typing import Any

import numpy as np

from studies._gauss_legendre4_common import AdaptiveReference
from studies.abba4_configuration_comparison import ABBA4ConfigurationComparisonConfig
from studies.abba_implicit_iterations import ImplicitABBAIterationStudyConfig
from studies.area_comparison import AreaComparisonConfig, AreaStep
from studies.poincare_periodicity import analyze_poincare_periodicity, analyze_short_periodicity
from studies.poincare_probe import RK4ProbeSettings
from studies.poincare_region_probe import RegionProbeSettings
from studies.poincare_rho_sweep import RhoStarConfig
from studies.single_particle_recurrence import SingleParticleRecurrenceConfig
from studies.trajectory_symplecticity import TrajectorySymplecticityConfig


class StudyValidationContractTests(unittest.TestCase):
	"""Keep distinct input policies and owned result arrays stable after extraction."""

	def test_span_conversion_policies_remain_distinct(self) -> None:
		"""Array conversion and iterable unpacking retain their original exception types."""
		invalid: Any = object()
		with self.assertRaises(TypeError):
			ABBA4ConfigurationComparisonConfig(t_span=invalid)
		with self.assertRaisesRegex(ValueError, "two finite increasing times"):
			AreaComparisonConfig(
				steps=(AreaStep("coarse", 0.1), AreaStep("fine", 0.05)),
				t_span=invalid, save_interval=0.1,
			)

	def test_array_spans_are_copied_and_normalized(self) -> None:
		"""Subsequent caller mutations must not change the frozen study interval."""
		span: Any = np.array([0.0, 1.0])
		config = ABBA4ConfigurationComparisonConfig(t_span=span)
		span[1] = 3.0
		self.assertEqual(config.t_span, (0.0, 1.0))

	def test_integer_valued_float_acceptance_is_study_specific(self) -> None:
		"""Recurrence accepts integral numeric values; iteration studies require integer types."""
		floating_count: Any = 2.0
		self.assertEqual(SingleParticleRecurrenceConfig(samples_per_cycle=floating_count).samples_per_cycle, 2.0)
		with self.assertRaisesRegex(ValueError, "integer of at least two"):
			ImplicitABBAIterationStudyConfig(sample_count=floating_count)

	def test_coupling_boolean_policy_is_not_broadened(self) -> None:
		"""Historical float-converting area controls differ from strict trajectory controls."""
		steps = (AreaStep("coarse", 0.1), AreaStep("fine", 0.05))
		area = AreaComparisonConfig(steps=steps, t_span=(0.0, 1.0), save_interval=0.1, coupling_frequency=True)
		self.assertEqual(area.coupling_frequency, 1.0)
		with self.assertRaisesRegex(ValueError, "finite and non-negative"):
			TrajectorySymplecticityConfig(
				steps=steps, t_span=(0.0, 1.0), save_interval=0.1,
				rho=0.0, coupling_frequency=True,
			)

	def test_numpy_integer_particle_contracts_remain_distinct(self) -> None:
		"""Rho stars accept NumPy integers while both probe APIs retain Python integer checks."""
		count: Any = np.int64(8)
		self.assertEqual(RhoStarConfig(rho_hat=0.0, particles=count).particles, 8)
		with self.assertRaisesRegex(ValueError, "particle_id must be a positive integer"):
			RK4ProbeSettings(count, (0.1, 0.2), "#aabbcc", 2, 10, 10, 0.0)
		with self.assertRaisesRegex(ValueError, "particle_id must be a positive integer"):
			RegionProbeSettings("RK4", (count,), ((0.1, 0.2),), ("#aabbcc",), 2, 10, 10, 0.0)

	def test_periodicity_threshold_uniqueness_is_scan_specific(self) -> None:
		"""Short-rhythm scans retain repeated thresholds, unlike long-lag scans."""
		positions = np.zeros((101, 1, 2))
		thresholds = (0.01, 0.01)
		with self.assertRaisesRegex(ValueError, "distinct fractions"):
			analyze_poincare_periodicity(positions, (1,), threshold_fractions=thresholds)
		result = analyze_short_periodicity(
			positions, (1,), max_lag=8, candidate_max_lag=2,
			multiple_count=2, block_cycles=20, threshold_fractions=thresholds,
		)
		self.assertEqual(len(result.summary), 1)

	def test_reference_arrays_are_owned_and_immutable(self) -> None:
		"""Shared array validation must preserve ownership as well as shape checks."""
		times = np.array([0.0, 1.0])
		states = np.zeros((2, 2))
		distances = np.zeros((1, 2))
		result = AdaptiveReference(times, states, states, distances, 1.0, 1.0, 1, 1)
		states[:] = 7.0
		self.assertTrue(np.array_equal(result.states, np.zeros((2, 2))))
		for value in (result.times, result.states, result.audit_states, result.audit_distances):
			self.assertFalse(value.flags.writeable)
		with self.assertRaisesRegex(ValueError, "invalid or misaligned"):
			replace(result, audit_states=np.zeros((2, 3)))


if __name__ == "__main__":
	unittest.main()
