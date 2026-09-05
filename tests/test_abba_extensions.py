"""Semantic contracts for the canonical ABBA state-extension parameter."""

from __future__ import annotations

import unittest
from typing import ClassVar

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


class _LinearTimeHamiltonian:
	"""Zero physical field with exactly integrable ``H(t)=slope*t``."""

	state_dimension: ClassVar[int] = 2

	def __init__(self, slope: float) -> None:
		self.slope = float(slope)

	def vector_field(self, _t: float, state: np.ndarray) -> np.ndarray:
		"""Leave every physical particle fixed."""
		return np.zeros_like(state, dtype=float)

	def particle_vector_field_jacobians(
		self,
		_t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return the exact zero planar Jacobian for each particle."""
		return np.zeros((state.size // 2, 2, 2), dtype=float)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate one identical linear-time Hamiltonian per particle."""
		particle_count = state.shape[0] // 2
		times = np.asarray(t, dtype=float)
		return np.broadcast_to(
			self.slope * times,
			(particle_count, *times.shape),
		).copy()

	def extended_momentum_derivative(
		self,
		_t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return the exact conjugate equation ``kappa'=-slope``."""
		return np.full(state.size // 2, -self.slope, dtype=float)


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
			("physical", "fully_extended"),
		)

	def test_tracking_preserves_existing_positional_argument_meanings(self) -> None:
		midpoint = ABBA2Midpoint("physical", True, None)
		self.assertIs(midpoint.progress, True)
		self.assertIs(midpoint.track_energy, False)

		implicit = ABBA2Implicit(
			"reduced_multiplier",
			"physical",
			1e-10,
			1e-9,
			20,
			"broyden",
			True,
			None,
		)
		self.assertEqual(implicit.newton_absolute_tolerance, 1e-10)
		self.assertEqual(implicit.newton_relative_tolerance, 1e-9)
		self.assertEqual(implicit.newton_max_iterations, 20)
		self.assertEqual(implicit.nonlinear_solver, "broyden")
		self.assertIs(implicit.progress, True)
		self.assertIs(implicit.track_energy, False)

	def test_only_fully_extended_rejects_multiple_particles(self) -> None:
		problem = _problem(particle_count=2)
		request = _request(duration=0.02)
		for method_type in _ALL_METHODS:
			kwargs: dict[str, object] = {"state_extension": "fully_extended"}
			if method_type is not ABBA2Midpoint:
				kwargs.update(
					projection_formulation="reduced_multiplier",
					nonlinear_solver="newton",
				)
			with self.subTest(method=method_type.__name__):
				with self.assertRaisesRegex(ValueError, "exactly one GC particle"):
					simulate(problem, method_type(**kwargs), request)

	def test_physical_energy_tracking_supports_multiple_particles(self) -> None:
		problem = _problem(particle_count=3)
		request = _request(duration=0.02)
		for method_type in _ALL_METHODS:
			kwargs: dict[str, object] = {"track_energy": True}
			if method_type is not ABBA2Midpoint:
				kwargs["projection_formulation"] = "reduced_multiplier"
			with self.subTest(method=method_type.__name__):
				tracked = simulate(problem, method_type(**kwargs), request)
				untracked_options = dict(kwargs)
				untracked_options["track_energy"] = False
				untracked = simulate(
					problem,
					method_type(**untracked_options),
					request,
				)
				np.testing.assert_array_equal(tracked.states, untracked.states)
				self.assertEqual(tracked.states.shape, (6, 2))
				self.assertEqual(
					np.asarray(tracked.diagnostics["extended_momentum"]).shape,
					(3, 2),
				)
				self.assertEqual(tracked.diagnostics["state_extension"], "physical")
				self.assertIs(tracked.diagnostics["track_energy"], True)
				self.assertGreaterEqual(tracked.diagnostics["energy_error"], 0.0)

	def test_physical_tracking_has_the_exact_conjugate_sign_and_factor(self) -> None:
		particle_count = 3
		slope = 1.75
		configuration = GCInitialConfiguration.from_components(
			x=np.linspace(0.8, 1.0, particle_count),
			y=np.linspace(1.1, 1.3, particle_count),
		)
		problem = InitialValueProblem(
			_LinearTimeHamiltonian(slope),
			configuration,
		)
		request = SimulationRequest.uniform(
			t_span=(0.2, 0.5),
			max_step=0.1,
			sample_count=4,
		)
		for method_type in _ALL_METHODS:
			with self.subTest(method=method_type.__name__):
				solution = simulate(
					problem,
					method_type(track_energy=True),
					request,
				)
				expected = np.broadcast_to(
					-slope * (solution.t - solution.t[0]),
					(particle_count, solution.t.size),
				)
				np.testing.assert_allclose(
					np.asarray(solution.diagnostics["extended_momentum"]),
					expected,
					rtol=0.0,
					atol=2e-15,
				)
				self.assertLess(
					float(solution.diagnostics["energy_error"]),
					2e-15,
				)

	def test_non_finite_momentum_increment_is_rejected_immediately(self) -> None:
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([0.8]),
			y=np.asarray([1.1]),
		)
		problem = InitialValueProblem(
			_LinearTimeHamiltonian(float(np.finfo(float).max)),
			configuration,
		)
		with self.assertRaisesRegex(ValueError, "momentum increment became non-finite"):
			simulate(
				problem,
				ABBA2Midpoint(track_energy=True),
				_request(duration=1.0),
			)

	def test_fully_extended_abba2_is_distinct_from_tracked_physical(self) -> None:
		problem = _problem()
		request = _request()
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			with self.subTest(formulation=formulation):
				physical = simulate(
					problem,
					ABBA2Implicit(
						projection_formulation=formulation,
						state_extension="physical",
						track_energy=True,
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
					physical.diagnostics["state_extension"],
					"physical",
				)
				self.assertEqual(
					fully_extended.diagnostics["state_extension"],
					"fully_extended",
				)
				self.assertIn("extended_momentum", physical.diagnostics)
				self.assertIn("extended_momentum", fully_extended.diagnostics)
				self.assertEqual(
					physical.diagnostics["extended_momentum_normalization"],
					"kappa_equals_k_over_2",
				)
				self.assertEqual(
					fully_extended.diagnostics["extended_momentum_normalization"],
					"direct_k",
				)
				self.assertGreater(
					float(
						np.max(
							np.abs(physical.states - fully_extended.states)
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
		self.assertEqual(
			solution.diagnostics["energy_error"],
			float(np.max(np.abs(error))),
		)
		np.testing.assert_allclose(
			solution.diagnostics["extended_time"],
			solution.t,
			rtol=0.0,
			atol=5e-15,
		)

	def test_fully_extended_enables_energy_tracking_implicitly(self) -> None:
		for method_type in _ALL_METHODS:
			with self.subTest(method=method_type.__name__):
				method = method_type(
					state_extension="fully_extended",
					track_energy=False,
				)
				self.assertIs(method.track_energy, True)

	def test_disabled_physical_tracking_has_no_momentum_diagnostics(self) -> None:
		solution = simulate(_problem(), ABBA2Implicit(), _request())
		self.assertIs(solution.diagnostics["track_energy"], False)
		self.assertNotIn("extended_momentum", solution.diagnostics)
		self.assertNotIn("energy_error", solution.diagnostics)

	def test_invalid_extension_fails_during_method_configuration(self) -> None:
		for method_type in _ALL_METHODS:
			with self.subTest(method=method_type.__name__):
				with self.assertRaisesRegex(ValueError, "state_extension"):
					method_type(state_extension="unknown")  # type: ignore[arg-type]
				with self.assertRaisesRegex(ValueError, "state_extension"):
					method_type(state_extension="shared_time")  # type: ignore[arg-type]


if __name__ == "__main__":
	unittest.main()
