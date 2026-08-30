"""Contracts for the shared-time and fully duplicated ABBA2 extensions."""

from __future__ import annotations

import unittest

import numpy as np

from diagnostics import GCGeneralizedEnergyObserver
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA2FullyExtendedImplicit,
	ABBA2Implicit,
	ABBA2SharedTimeExtendedImplicit,
	ABBA_PROJECTION_FORMULATIONS,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods import ABBA2SharedTimeExtendedImplicit as MethodsExport
from simulation.methods.abba.extensions import (
	ABBA2SharedTimeExtendedImplicit as ExtensionsExport,
)


def _problem(*, particle_count: int = 1) -> InitialValueProblem:
	"""Build a deterministic non-autonomous GC problem."""
	potential = Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)
	configuration = GCInitialConfiguration.from_components(
		x=np.linspace(1.0, 1.1, particle_count),
		y=np.linspace(1.2, 1.3, particle_count),
	)
	return InitialValueProblem(
		GuidingCenterDynamics(potential, rho=0.05),
		configuration,
	)


def _request() -> SimulationRequest:
	"""Return a short grid that exposes two accepted steps and three samples."""
	return SimulationRequest.uniform(
		t_span=(0.0, 0.2),
		max_step=0.1,
		sample_count=3,
	)


class ABBA2SharedTimeExtensionTests(unittest.TestCase):
	"""Verify the R6 lift without conflating it with the full R8 method."""

	def test_shared_time_method_is_exported_by_all_public_facades(self) -> None:
		self.assertIs(MethodsExport, ABBA2SharedTimeExtendedImplicit)
		self.assertIs(ExtensionsExport, ABBA2SharedTimeExtendedImplicit)

	def test_physical_map_and_reconstructed_kappa_match_for_both_formulations(
		self,
	) -> None:
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			with self.subTest(formulation=formulation):
				problem = _problem()
				initial_state = np.asarray(problem.initial_state)
				observer = GCGeneralizedEnergyObserver(
					problem.dynamics,
					initial_time=0.0,
					initial_state=initial_state,
				)
				shared = simulate(
					problem,
					ABBA2SharedTimeExtendedImplicit(
						projection_formulation=formulation,
						newton_absolute_tolerance=1e-14,
						newton_relative_tolerance=1e-14,
						step_observer=observer,
					),
					_request(),
				)
				physical = simulate(
					problem,
					ABBA2Implicit(
						projection_formulation=formulation,
						newton_absolute_tolerance=1e-14,
						newton_relative_tolerance=1e-14,
					),
					_request(),
				)

				np.testing.assert_allclose(
					shared.states,
					physical.states,
					rtol=0.0,
					atol=2e-15,
				)
				self.assertEqual(
					shared.diagnostics["projection_formulation"],
					formulation,
				)
				self.assertEqual(
					shared.diagnostics["state_extension"],
					"shared_time_extended",
				)
				self.assertEqual(
					shared.diagnostics["extended_momentum_normalization"],
					"kappa_equals_k_over_2",
				)
				extended_time = np.asarray(shared.diagnostics["extended_time"])
				extended_kappa = np.asarray(shared.diagnostics["extended_kappa"])
				self.assertEqual(extended_time.shape, shared.t.shape)
				self.assertEqual(extended_kappa.shape, shared.t.shape)
				np.testing.assert_allclose(extended_time, shared.t, rtol=0.0, atol=0.0)
				self.assertEqual(float(extended_kappa[0]), 0.0)
				np.testing.assert_allclose(
					extended_kappa,
					[record.kappa for record in observer.records],
					rtol=0.0,
					atol=2e-15,
				)

	def test_shared_time_extension_rejects_multiple_particles(self) -> None:
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			with self.subTest(formulation=formulation):
				with self.assertRaisesRegex(ValueError, "exactly one GC particle"):
					simulate(
						_problem(particle_count=2),
						ABBA2SharedTimeExtendedImplicit(
							projection_formulation=formulation
						),
						_request(),
					)

	def test_fully_duplicated_extension_is_a_distinct_numerical_method(self) -> None:
		problem = _problem()
		shared = simulate(
			problem,
			ABBA2SharedTimeExtendedImplicit(),
			_request(),
		)
		fully_extended = simulate(
			problem,
			ABBA2FullyExtendedImplicit(),
			_request(),
		)

		self.assertEqual(shared.diagnostics["state_extension"], "shared_time_extended")
		self.assertEqual(fully_extended.diagnostics["state_extension"], "fully_extended")
		self.assertIn("extended_kappa", shared.diagnostics)
		self.assertIn("extended_momentum", fully_extended.diagnostics)
		self.assertGreater(
			float(np.max(np.abs(shared.states - fully_extended.states))),
			1e-9,
		)


if __name__ == "__main__":
	unittest.main()
