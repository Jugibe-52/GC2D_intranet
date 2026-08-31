"""Sixteen-configuration ABBA4 comparison on separate trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from diagnostics import ReferenceTrajectoryPaths, StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from studies.abba4_configuration_comparison import (
	ABBA4_CONFIGURATION_KEYS,
	ABBA4_CONFIGURATION_PARTICLE_COUNT,
	ABBA4_CONFIGURATION_VARIANTS,
	ABBA4ConfigurationComparisonConfig,
	ABBA4ConfigurationComparisonResult,
	run_abba4_configuration_comparison,
)
from studies.reference_trajectory import potential_fingerprint


def _potential() -> Potential:
	"""Return a compact smooth non-autonomous field for the 160 smoke runs."""
	return Potential.random(
		A=0.02,
		M=2,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)


def _configuration(*, particle_count: int = 10) -> GCInitialConfiguration:
	"""Place deterministic independent guiding centers inside the base cell."""
	return GCInitialConfiguration.from_components(
		x=np.linspace(0.8, 1.4, particle_count),
		y=np.linspace(1.0, 1.6, particle_count),
	)


def _reference(
	potential: Potential,
	configuration: GCInitialConfiguration,
	*,
	distance_convention: str = "euclidean",
	fingerprint: str | None = None,
) -> StoredReferenceTrajectory:
	"""Construct an in-memory DOP853 reference containing a longer prefix."""
	initial_state = configuration.initial_state
	assert initial_state is not None
	particle_count = configuration.layout.particle_count(initial_state)
	dynamics = GuidingCenterDynamics(potential, rho=0.0)
	times = np.linspace(0.0, 0.04, 5)
	solve = solve_ivp(
		dynamics.vector_field,
		(float(times[0]), float(times[-1])),
		initial_state,
		method="DOP853",
		t_eval=times,
		rtol=1e-12,
		atol=1e-14,
		max_step=0.001,
	)
	if not solve.success:
		raise RuntimeError(solve.message)
	states = np.asarray(solve.y, dtype=float)
	states[:, 0] = initial_state
	directory = Path("/tmp/abba4-configuration-comparison-reference")
	metadata = {
		"config": {
			"distance_convention": distance_convention,
			"rho": 0.0,
			"t_span": (0.0, 0.04),
			"save_interval": 0.01,
		},
		"particle_count": particle_count,
	}
	if fingerprint is not None:
		metadata["dynamics_fingerprint_sha256"] = fingerprint
	return StoredReferenceTrajectory(
		times=times,
		states=states,
		initial_state=initial_state,
		audit_states=states,
		audit_distances=np.zeros((particle_count, times.size)),
		metadata=metadata,
		paths=ReferenceTrajectoryPaths(
			directory=directory,
			trajectory=directory / "trajectory.npz",
			metadata=directory / "metadata.json",
			readme=directory / "README.md",
		),
	)


class ABBA4ConfigurationComparisonTests(unittest.TestCase):
	"""Verify orchestration, metrics, dimensions, and reference semantics."""

	potential: ClassVar[Potential]
	configuration: ClassVar[GCInitialConfiguration]
	reference: ClassVar[StoredReferenceTrajectory]
	config: ClassVar[ABBA4ConfigurationComparisonConfig]
	result: ClassVar[ABBA4ConfigurationComparisonResult]

	@classmethod
	def setUpClass(cls) -> None:
		"""Run all 160 short simulations once for shared assertions."""
		cls.potential = _potential()
		cls.configuration = _configuration()
		fingerprint = potential_fingerprint(
			GuidingCenterDynamics(cls.potential, rho=0.0).effective_potential
		)
		cls.reference = _reference(
			cls.potential,
			cls.configuration,
			fingerprint=fingerprint,
		)
		cls.config = ABBA4ConfigurationComparisonConfig(
			t_span=(0.0, 0.02),
			integration_step=0.01,
			save_interval=0.01,
			rho=0.0,
			absolute_tolerance=1e-12,
			relative_tolerance=1e-12,
			max_iterations=40,
			progress=False,
		)
		cls.result = run_abba4_configuration_comparison(
			cls.potential,
			cls.configuration,
			cls.reference,
			config=cls.config,
		)

	def test_runs_exactly_sixteen_by_ten_aligned_trajectories(self) -> None:
		"""Keep every one-particle run instead of vectorizing extended states."""
		self.assertEqual(self.config.particle_count, ABBA4_CONFIGURATION_PARTICLE_COUNT)
		self.assertEqual(len(ABBA4_CONFIGURATION_VARIANTS), 16)
		self.assertEqual(tuple(self.result.solutions), ABBA4_CONFIGURATION_KEYS)
		self.assertEqual(tuple(self.result.runtimes), ABBA4_CONFIGURATION_KEYS)
		np.testing.assert_array_equal(self.result.times, [0.0, 0.01, 0.02])
		np.testing.assert_array_equal(self.result.reference_sample_indices, [0, 1, 2])
		self.assertFalse(self.result.reference_sample_indices.flags.writeable)
		for key in ABBA4_CONFIGURATION_KEYS:
			self.assertEqual(len(self.result.solutions[key]), 10)
			self.assertEqual(self.result.runtimes[key].shape, (10,))
			self.assertFalse(self.result.runtimes[key].flags.writeable)
			for solution in self.result.solutions[key]:
				self.assertEqual(solution.states.shape, (2, 3))

	def test_summaries_contain_exactly_five_finite_metrics(self) -> None:
		"""Reduce particle-time errors, runtime, solver work, and energy."""
		rows = self.result.summaries()
		self.assertEqual(len(rows), 16)
		self.assertEqual(tuple(row.key for row in rows), ABBA4_CONFIGURATION_KEYS)
		metric_names = (
			"mean_trajectory_error",
			"final_trajectory_error",
			"total_runtime_seconds",
			"mean_iterations_per_solve",
			"mean_relative_energy_error",
		)
		for row in rows:
			metrics = np.asarray(
				[getattr(row, name) for name in metric_names],
				dtype=float,
			)
			self.assertTrue(np.all(np.isfinite(metrics)))
			self.assertTrue(np.all(metrics >= 0.0))
			self.assertGreater(row.total_runtime_seconds, 0.0)

	def test_trajectory_metrics_are_global_and_final_particle_rms_errors(
		self,
	) -> None:
		"""Evaluate both RMS definitions directly from reference coordinates."""
		rows = {row.key: row for row in self.result.summaries()}
		reference = self.reference.states[:, self.result.reference_sample_indices]
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			squared_distances = np.asarray(
				[
					(
						solution.states[0] - reference[particle]
					) ** 2
					+ (
						solution.states[1]
						- reference[self.config.particle_count + particle]
					) ** 2
					for particle, solution in enumerate(
						self.result.solutions[variant.key]
					)
				],
				dtype=float,
			)
			expected_mean = float(np.sqrt(np.mean(squared_distances)))
			expected_final = float(
				np.sqrt(np.mean(squared_distances[:, -1]))
			)
			with self.subTest(configuration=variant.key):
				np.testing.assert_allclose(
					rows[variant.key].mean_trajectory_error,
					expected_mean,
					rtol=1e-14,
					atol=1e-18,
				)
				np.testing.assert_allclose(
					rows[variant.key].final_trajectory_error,
					expected_final,
					rtol=1e-14,
					atol=1e-18,
				)

	def test_runtime_and_iteration_metrics_use_sum_and_per_solve_mean(
		self,
	) -> None:
		"""Sum all timings and count every composed nonlinear solve once."""
		rows = {row.key: row for row in self.result.summaries()}
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			expected_runtime = sum(
				float(value) for value in self.result.runtimes[variant.key]
			)
			iteration_total = 0
			solve_total = 0
			for solution in self.result.solutions[variant.key]:
				substep_iterations = np.asarray(
					solution.diagnostics["substep_nonlinear_iterations"],
					dtype=int,
				)
				self.assertEqual(substep_iterations.ndim, 2)
				self.assertEqual(
					substep_iterations.shape[1],
					int(solution.diagnostics["nonlinear_solves_per_step"]),
				)
				iteration_total += int(np.sum(substep_iterations))
				solve_total += int(substep_iterations.size)
			expected_iterations = float(iteration_total) / float(solve_total)
			with self.subTest(configuration=variant.key):
				np.testing.assert_allclose(
					rows[variant.key].total_runtime_seconds,
					expected_runtime,
					rtol=1e-14,
					atol=1e-15,
				)
				self.assertEqual(
					rows[variant.key].mean_iterations_per_solve,
					expected_iterations,
				)

	def test_shared_time_energy_metric_uses_hamiltonian_plus_kappa(self) -> None:
		"""Reconstruct the shared-time generalized energy independently."""
		rows = {row.key: row for row in self.result.summaries()}
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			if variant.state_extension != "shared_time":
				continue
			relative_errors: list[np.ndarray] = []
			for solution in self.result.solutions[variant.key]:
				diagnostics = solution.diagnostics
				times = np.asarray(diagnostics["extended_time"], dtype=float)
				hamiltonian = np.asarray(
					self.result.dynamics.hamiltonian(times, solution.states),
					dtype=float,
				).reshape(-1)
				kappa = np.asarray(
					diagnostics["extended_kappa"],
					dtype=float,
				).reshape(-1)
				generalized_energy = hamiltonian + kappa
				scale = max(
					abs(float(generalized_energy[0])),
					float(np.finfo(float).eps),
				)
				relative_errors.append(
					np.abs(generalized_energy - generalized_energy[0]) / scale
				)
			expected = float(np.mean(np.asarray(relative_errors)))
			with self.subTest(configuration=variant.key):
				np.testing.assert_allclose(
					rows[variant.key].mean_relative_energy_error,
					expected,
					rtol=1e-14,
					atol=1e-18,
				)

	def test_fully_extended_energy_metric_uses_hamiltonian_plus_k(self) -> None:
		"""Reconstruct direct-k generalized energy without stored energy arrays."""
		rows = {row.key: row for row in self.result.summaries()}
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			if variant.state_extension != "fully_extended":
				continue
			relative_errors: list[np.ndarray] = []
			for solution in self.result.solutions[variant.key]:
				diagnostics = solution.diagnostics
				times = np.asarray(diagnostics["extended_time"], dtype=float)
				hamiltonian = np.asarray(
					self.result.dynamics.hamiltonian(times, solution.states),
					dtype=float,
				).reshape(-1)
				momentum = np.asarray(
					diagnostics["extended_momentum"],
					dtype=float,
				).reshape(-1)
				generalized_energy = hamiltonian + momentum
				scale = max(
					abs(float(generalized_energy[0])),
					float(np.finfo(float).eps),
				)
				relative_errors.append(
					np.abs(generalized_energy - generalized_energy[0]) / scale
				)
			expected = float(np.mean(np.asarray(relative_errors)))
			with self.subTest(configuration=variant.key):
				np.testing.assert_allclose(
					rows[variant.key].mean_relative_energy_error,
					expected,
					rtol=1e-14,
					atol=1e-18,
				)

	def test_every_variant_reports_its_literal_r6_or_r8_dimensions(self) -> None:
		"""Distinguish splitting dimensions from nonlinear workspace sizes."""
		for variant in ABBA4_CONFIGURATION_VARIANTS:
			diagnostics = self.result.solutions[variant.key][0].diagnostics
			if variant.state_extension == "shared_time":
				expected = (
					4,
					6,
					2
					if variant.projection_formulation == "reduced_multiplier"
					else 6,
				)
			else:
				expected = (
					4,
					8,
					4
					if variant.projection_formulation == "reduced_multiplier"
					else 12,
				)
			actual = tuple(
				int(diagnostics[name])
				for name in (
					"accepted_internal_state_dimension",
					"base_splitting_state_dimension",
					"nonlinear_unknown_dimension",
				)
			)
			self.assertEqual(actual, expected)

	def test_rejects_non_euclidean_references_before_simulation(self) -> None:
		"""Do not apply periodic minimum-image errors to the clipped HDF5 model."""
		periodic = _reference(
			self.potential,
			self.configuration,
			distance_convention="periodic",
		)
		with self.assertRaisesRegex(ValueError, "Euclidean"):
			run_abba4_configuration_comparison(
				self.potential,
				self.configuration,
				periodic,
				config=self.config,
			)

	def test_rejects_a_nonprefix_reference_interval(self) -> None:
		"""Do not pair the stored initial state with a later reference time."""
		shifted = ABBA4ConfigurationComparisonConfig(
			t_span=(0.01, 0.03),
			integration_step=0.01,
			save_interval=0.01,
			absolute_tolerance=1e-12,
			relative_tolerance=1e-12,
		)
		with self.assertRaisesRegex(ValueError, "reference initial time"):
			run_abba4_configuration_comparison(
				self.potential,
				self.configuration,
				self.reference,
				config=shifted,
			)

	def test_rejects_wrong_particle_count_and_fingerprint(self) -> None:
		"""Require the configured state count and certified interpolated dynamics."""
		with self.assertRaisesRegex(ValueError, "particle_count=10"):
			run_abba4_configuration_comparison(
				self.potential,
				_configuration(particle_count=9),
				self.reference,
				config=self.config,
			)
		wrong_fingerprint = _reference(
			self.potential,
			self.configuration,
			fingerprint="not-the-effective-potential",
		)
		with self.assertRaisesRegex(ValueError, "dynamics differs"):
			run_abba4_configuration_comparison(
				self.potential,
				self.configuration,
				wrong_fingerprint,
				config=self.config,
			)

	def test_particle_count_three_runs_sixteen_separate_triplets(self) -> None:
		"""Allow a quick 16-by-3 study without changing the ten-path default."""
		configuration = _configuration(particle_count=3)
		fingerprint = potential_fingerprint(
			GuidingCenterDynamics(self.potential, rho=0.0).effective_potential
		)
		reference = _reference(
			self.potential,
			configuration,
			fingerprint=fingerprint,
		)
		config = ABBA4ConfigurationComparisonConfig(
			t_span=(0.0, 0.01),
			integration_step=0.01,
			save_interval=0.01,
			rho=0.0,
			absolute_tolerance=1e-12,
			relative_tolerance=1e-12,
			max_iterations=40,
			progress=False,
			particle_count=3,
		)
		result = run_abba4_configuration_comparison(
			self.potential,
			configuration,
			reference,
			config=config,
		)

		self.assertEqual(config.particle_count, 3)
		self.assertEqual(len(result.summaries()), 16)
		for key in ABBA4_CONFIGURATION_KEYS:
			self.assertEqual(len(result.solutions[key]), 3)
			self.assertEqual(result.runtimes[key].shape, (3,))
			for solution in result.solutions[key]:
				source_state = solution.source.initial_state
				assert source_state is not None
				self.assertEqual(
					solution.source.layout.particle_count(source_state),
					1,
				)


if __name__ == "__main__":
	unittest.main()
