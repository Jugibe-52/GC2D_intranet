"""Semantic contracts for the canonical ABBA state-extension parameter."""

from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA2Implicit,
	ABBA2Midpoint,
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	ABBA6Implicit,
	ABBA_PROJECTION_FORMULATIONS,
	ABBA_STATE_EXTENSIONS,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)


_ALL_METHODS = (
	ABBA2Midpoint,
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	ABBA6Implicit,
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


def _request(*, duration: float = 0.1) -> SimulationRequest:
	"""Return one accepted step and both endpoint samples."""
	return SimulationRequest.uniform(
		t_span=(0.0, duration),
		max_step=duration,
		sample_count=2,
	)


class ABBAStateExtensionTests(unittest.TestCase):
	"""Verify state-space meaning independently of the configuration smoke test."""

	def test_extension_identifiers_are_stable_and_complete(self) -> None:
		self.assertEqual(
			ABBA_STATE_EXTENSIONS,
			("physical", "shared_time", "fully_extended"),
		)

	def test_shared_and_fully_extended_variants_reject_multiple_particles(
		self,
	) -> None:
		problem = _problem(particle_count=2)
		request = _request(duration=0.02)
		for method_type, extension in product(
			_ALL_METHODS,
			("shared_time", "fully_extended"),
		):
			kwargs: dict[str, object] = {"state_extension": extension}
			if method_type is not ABBA2Midpoint:
				kwargs.update(
					projection_formulation="reduced_multiplier",
					nonlinear_solver="newton",
				)
			with self.subTest(method=method_type.__name__, extension=extension):
				with self.assertRaisesRegex(ValueError, "exactly one GC particle"):
					simulate(problem, method_type(**kwargs), request)

	def test_fully_extended_abba2_is_distinct_from_the_shared_time_lift(self) -> None:
		problem = _problem()
		request = _request()
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			with self.subTest(formulation=formulation):
				shared = simulate(
					problem,
					ABBA2Implicit(
						projection_formulation=formulation,
						state_extension="shared_time",
						newton_absolute_tolerance=1e-14,
						newton_relative_tolerance=1e-14,
					),
					request,
				)
				fully_extended = simulate(
					problem,
					ABBA2Implicit(
						projection_formulation=formulation,
						state_extension="fully_extended",
						newton_absolute_tolerance=1e-14,
						newton_relative_tolerance=1e-14,
					),
					request,
				)

				self.assertEqual(
					shared.diagnostics["state_extension"],
					"shared_time",
				)
				self.assertEqual(
					fully_extended.diagnostics["state_extension"],
					"fully_extended",
				)
				self.assertIn("extended_kappa", shared.diagnostics)
				self.assertIn("extended_momentum", fully_extended.diagnostics)
				self.assertEqual(
					shared.diagnostics["extended_momentum_normalization"],
					"kappa_equals_k_over_2",
				)
				self.assertEqual(
					fully_extended.diagnostics["extended_momentum_normalization"],
					"direct_k",
				)
				self.assertGreater(
					float(
						np.max(
							np.abs(shared.states - fully_extended.states)
						)
					),
					1e-9,
				)

	def test_fully_extended_diagnostics_close_the_generalized_energy_identity(
		self,
	) -> None:
		solution = simulate(
			_problem(),
			ABBA2Implicit(state_extension="fully_extended"),
			_request(duration=0.05),
		)
		hamiltonian = np.asarray(solution.diagnostics["physical_hamiltonian"])
		momentum = np.asarray(solution.diagnostics["extended_momentum"])
		generalized = np.asarray(solution.diagnostics["generalized_energy"])
		error = np.asarray(solution.diagnostics["generalized_energy_error"])
		np.testing.assert_allclose(generalized, hamiltonian + momentum)
		np.testing.assert_allclose(error, generalized - generalized[0])
		np.testing.assert_allclose(
			solution.diagnostics["extended_time"],
			solution.t,
			rtol=0.0,
			atol=5e-15,
		)

	def test_invalid_extension_fails_during_method_configuration(self) -> None:
		for method_type in _ALL_METHODS:
			with self.subTest(method=method_type.__name__):
				with self.assertRaisesRegex(ValueError, "state_extension"):
					method_type(state_extension="unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
	unittest.main()
