"""Semantic contracts for the canonical ABBA state-extension parameter."""

from __future__ import annotations

from diagnostics.methods.energy import EnergyObserver
from solvers import BroydenSolver, NewtonSolver
from dataclasses import replace


import unittest
from typing import ClassVar

import numpy as np

from dynamics import GuidingCenterDynamics

from potential import Potential
from simulation import (
	ABBA2Implicit, ABBA2Midpoint, ABBA4Implicit, ABBA_PROJECTION_FORMULATIONS, ABBA_STATE_EXTENSIONS,
	InitialValueProblem, SimulationRequest, simulate,
)


_ALL_METHODS = (
	ABBA2Midpoint,
	ABBA2Implicit,
	ABBA4Implicit,

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
	configuration = GuidingCenterDynamics.coordinates.pack_components(np.linspace(1.0, 1.1, particle_count), np.linspace(1.2, 1.3, particle_count))
	return InitialValueProblem(GuidingCenterDynamics(potential, rho=0.05), configuration, layout=GuidingCenterDynamics.coordinates)


def _request(*, duration: float = 0.1) -> SimulationRequest:
	"""Return one accepted step and both endpoint samples."""
	return SimulationRequest.uniform(
		t_span=(0.0, duration),
		sample_count=2,
	)


class ABBAStateExtensionTests(unittest.TestCase):
	"""Verify state-space meaning independently of the configuration smoke test."""

	def test_extension_identifiers_are_stable_and_complete(self) -> None:
		self.assertEqual(
			ABBA_STATE_EXTENSIONS,
			("physical",),
		)

	def test_solver_and_observers_are_no_longer_method_controls(self) -> None:
		midpoint = ABBA2Midpoint('physical', True)
		self.assertIs(midpoint.progress, True)
		self.assertFalse(hasattr(midpoint, 'track_energy'))
		solver = BroydenSolver(1e-10, 1e-9, 20)
		implicit = ABBA2Implicit('reduced_multiplier', 'physical', True, solver=solver)
		self.assertIs(implicit.solver, solver)
		self.assertIs(implicit.progress, True)
		for controls in ({'track_energy': True}, {'newton_max_iterations': 20}, {'step_observer': None}):
			with self.assertRaises(TypeError): ABBA2Implicit(**controls)

	def test_only_fully_extended_rejects_multiple_particles(self) -> None:
		problem = _problem(particle_count=2)
		request = _request(duration=0.02)
		for method_type in _ALL_METHODS:
			kwargs: dict[str, object] = {"state_extension": "fully_extended"}
			if method_type is not ABBA2Midpoint:
				kwargs.update(
					projection_formulation="reduced_multiplier",
					solver=NewtonSolver(jacobian_method="analytic"),
				)
			with self.subTest(method=method_type.__name__):
				with self.assertRaisesRegex(ValueError, "energy observer"):
					simulate(problem, replace(method_type(**kwargs), step=0.02), request)

	def test_physical_energy_tracking_supports_multiple_particles(self) -> None:
		problem = _problem(particle_count=3)
		request = _request(duration=0.02)
		for method_type in _ALL_METHODS:
			kwargs: dict[str, object] = {}
			if method_type is not ABBA2Midpoint:
				kwargs["projection_formulation"] = "reduced_multiplier"
			with self.subTest(method=method_type.__name__):
				tracked = simulate(problem, replace(method_type(**kwargs), step=0.02), request, observers=(EnergyObserver(),))
				untracked_options = dict(kwargs)
				untracked = simulate(problem, replace(method_type(**untracked_options), step=0.02), request)
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
		configuration = GuidingCenterDynamics.coordinates.pack_components(np.linspace(0.8, 1.0, particle_count), np.linspace(1.1, 1.3, particle_count))
		problem = InitialValueProblem(_LinearTimeHamiltonian(slope), configuration, layout=GuidingCenterDynamics.coordinates)
		request = SimulationRequest.uniform(
			t_span=(0.2, 0.5),
			sample_count=4,
		)
		for method_type in _ALL_METHODS:
			with self.subTest(method=method_type.__name__):
				solution = simulate(problem, replace(method_type(), step=0.1), request, observers=(EnergyObserver(),))
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
		configuration = GuidingCenterDynamics.coordinates.pack_components(np.asarray([0.8]), np.asarray([1.1]))
		problem = InitialValueProblem(_LinearTimeHamiltonian(float(np.finfo(float).max)), configuration, layout=GuidingCenterDynamics.coordinates)
		with self.assertRaisesRegex(ValueError, "momentum increment became non-finite"):
			simulate(problem, ABBA2Midpoint(step=1.0), _request(duration=1.0), observers=(EnergyObserver(),))

	def test_full_projection_is_rejected_instead_of_changing_the_physical_map(self) -> None:
		for formulation in ABBA_PROJECTION_FORMULATIONS:
			with self.assertRaisesRegex(ValueError, "removed"):
				ABBA2Implicit(state_extension="fully_extended", projection_formulation=formulation)

	def test_passive_diagnostics_close_the_generalized_energy_identity(self) -> None:
		solution = simulate(_problem(), ABBA4Implicit(step=0.1), _request(), observers=(EnergyObserver(),))
		d = solution.diagnostics
		np.testing.assert_array_equal(d["generalized_energy"], d["physical_hamiltonian"] + d["extended_momentum"])
		np.testing.assert_array_equal(d["extended_time"], solution.t[None, :])

	def test_old_full_extension_requires_an_explicit_migration(self) -> None:
		for cls in _ALL_METHODS:
			with self.assertRaisesRegex(ValueError, "energy observer"):
				cls(state_extension="fully_extended")

	def test_disabled_physical_tracking_has_no_momentum_diagnostics(self) -> None:
		solution = simulate(_problem(), ABBA2Implicit(step=0.1), _request())
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
