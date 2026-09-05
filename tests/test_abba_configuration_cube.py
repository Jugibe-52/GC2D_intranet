"""Public contracts for the complete canonical ABBA configuration space."""

from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from diagnostics import (
	abba4_implicit_single_projection_step_particle_jacobians,
	abba4_implicit_step_particle_jacobians,
	central_difference_jacobian,
)
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
	InitialValueProblem,
	SimulationRequest,
	simulate,
)


_IMPLICIT_METHODS = (
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	ABBA6Implicit,
)
_NONLINEAR_SOLVERS = ("newton", "broyden")
_ENERGY_STRATEGIES = (
	("physical", False),
	("physical", True),
	("fully_extended", True),
)
_IMPLICIT_CONFIGURATIONS = tuple(
	(method, formulation, solver, extension, track_energy)
	for method, formulation, solver, (
		extension,
		track_energy,
	) in product(
		_IMPLICIT_METHODS,
		ABBA_PROJECTION_FORMULATIONS,
		_NONLINEAR_SOLVERS,
		_ENERGY_STRATEGIES,
	)
)
_MIDPOINT_CONFIGURATIONS = tuple(
	(ABBA2Midpoint, extension, track_energy)
	for extension, track_energy in _ENERGY_STRATEGIES
)
_EXPECTED_DIMENSIONS = {
	"physical": (2, 4),
	"fully_extended": (4, 8),
}
_EXPECTED_NONLINEAR_SOLVES = {
	"ABBA2Implicit": 1,
	"ABBA4Implicit": 3,
	"ABBA4ImplicitSingleProjection": 1,
	"ABBA6Implicit": 7,
}


def _problem(*, particle_count: int = 1) -> InitialValueProblem:
	"""Return one deterministic smooth non-autonomous GC problem."""
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
	"""Use one short accepted step for inexpensive matrix-wide contracts."""
	return SimulationRequest.uniform(
		t_span=(0.0, 0.02),
		max_step=0.02,
		sample_count=2,
	)


def _dense_component_major_jacobian(blocks: np.ndarray) -> np.ndarray:
	"""Expand independent planar particle blocks into the packed layout."""
	values = np.asarray(blocks, dtype=float)
	particle_count = values.shape[0]
	result = np.zeros((2 * particle_count, 2 * particle_count), dtype=float)
	for particle in range(particle_count):
		indices = (particle, particle_count + particle)
		result[np.ix_(indices, indices)] = values[particle]
	return result


class ABBAConfigurationCubeTests(unittest.TestCase):
	"""Exercise three midpoint variants and the 48 implicit combinations once."""

	implicit_solutions: dict[tuple[str, str, str, str, bool], object] = {}
	midpoint_solutions: dict[tuple[str, bool], object] = {}
	configuration_failures: dict[tuple[str, ...], Exception] = {}

	@classmethod
	def setUpClass(cls) -> None:
		"""Cache all 51 one-step runs for smoke and equivalence assertions."""
		problem = _problem()
		request = _request()
		for (
			method_type,
			formulation,
			solver,
			extension,
			track_energy,
		) in _IMPLICIT_CONFIGURATIONS:
			key = (
				method_type.__name__,
				formulation,
				solver,
				extension,
				track_energy,
			)
			try:
				cls.implicit_solutions[key] = simulate(
					problem,
					method_type(
						projection_formulation=formulation,
						nonlinear_solver=solver,
						state_extension=extension,
						track_energy=track_energy,
						newton_absolute_tolerance=1e-12,
						newton_relative_tolerance=1e-12,
						newton_max_iterations=50,
					),
					request,
				)
			except Exception as exc:  # pragma: no cover - reported by the smoke test
				cls.configuration_failures[key] = exc
		for method_type, extension, track_energy in _MIDPOINT_CONFIGURATIONS:
			key = (method_type.__name__, extension, track_energy)
			try:
				cls.midpoint_solutions[(extension, track_energy)] = simulate(
					problem,
					method_type(
						state_extension=extension,
						track_energy=track_energy,
					),
					request,
				)
			except Exception as exc:  # pragma: no cover - reported by the smoke test
				cls.configuration_failures[key] = exc

	def test_configuration_space_contains_exactly_51_variants(self) -> None:
		self.assertEqual(len(_MIDPOINT_CONFIGURATIONS), 3)
		self.assertEqual(len(_IMPLICIT_CONFIGURATIONS), 48)
		self.assertEqual(
			len(_MIDPOINT_CONFIGURATIONS) + len(_IMPLICIT_CONFIGURATIONS),
			51,
		)

	def test_all_51_configurations_run_and_report_canonical_dimensions(self) -> None:
		for (
			method_type,
			formulation,
			solver,
			extension,
			track_energy,
		) in _IMPLICIT_CONFIGURATIONS:
			key = (
				method_type.__name__,
				formulation,
				solver,
				extension,
				track_energy,
			)
			with self.subTest(
				method=method_type.__name__,
				formulation=formulation,
				solver=solver,
				extension=extension,
				track_energy=track_energy,
			):
				if key in self.configuration_failures:
					self.fail(f"{key} failed: {self.configuration_failures[key]}")
				solution = self.implicit_solutions[key]
				self.assertEqual(solution.states.shape, (2, 2))
				self.assertTrue(np.all(np.isfinite(solution.states)))
				diagnostics = solution.diagnostics
				self.assertEqual(diagnostics["state_extension"], extension)
				self.assertIs(diagnostics["track_energy"], track_energy)
				self.assertEqual(
					diagnostics["projection_formulation"],
					formulation,
				)
				self.assertEqual(diagnostics["nonlinear_solver"], solver)
				accepted, base = _EXPECTED_DIMENSIONS[extension]
				self.assertEqual(
					diagnostics["accepted_internal_state_dimension"],
					accepted,
				)
				self.assertEqual(
					diagnostics["base_splitting_state_dimension"],
					base,
				)
				self.assertEqual(
					diagnostics["observer_state_dimension"],
					4 if extension == "fully_extended" else 2,
				)
				self.assertEqual(
					diagnostics["observer_state_kind"],
					"accepted_internal_map"
					if extension == "fully_extended"
					else "physical_map",
				)
				expected_unknown = (
					4
					if extension == "fully_extended"
					and formulation == "reduced_multiplier"
					else 12
					if extension == "fully_extended"
					else 2
					if formulation == "reduced_multiplier"
					else 6
				)
				self.assertEqual(
					diagnostics["nonlinear_unknown_dimension"],
					expected_unknown,
				)
				self.assertEqual(
					diagnostics["nonlinear_solves_per_step"],
					_EXPECTED_NONLINEAR_SOLVES[method_type.__name__],
				)

		for method_type, extension, track_energy in _MIDPOINT_CONFIGURATIONS:
			key = (method_type.__name__, extension, track_energy)
			with self.subTest(
				method=method_type.__name__,
				extension=extension,
				track_energy=track_energy,
			):
				if key in self.configuration_failures:
					self.fail(f"{key} failed: {self.configuration_failures[key]}")
				solution = self.midpoint_solutions[(extension, track_energy)]
				self.assertEqual(solution.states.shape, (2, 2))
				self.assertTrue(np.all(np.isfinite(solution.states)))
				diagnostics = solution.diagnostics
				self.assertEqual(diagnostics["state_extension"], extension)
				self.assertIs(diagnostics["track_energy"], track_energy)
				self.assertEqual(diagnostics["projection_kind"], "arithmetic_mean")
				accepted, base = _EXPECTED_DIMENSIONS[extension]
				self.assertEqual(
					diagnostics["accepted_internal_state_dimension"],
					accepted,
				)
				self.assertEqual(
					diagnostics["base_splitting_state_dimension"],
					base,
				)
				self.assertEqual(
					diagnostics["observer_state_dimension"],
					4 if extension == "fully_extended" else 2,
				)
				self.assertEqual(
					diagnostics["observer_state_kind"],
					"accepted_internal_map"
					if extension == "fully_extended"
					else "physical_map",
				)
				self.assertEqual(
					diagnostics["nonlinear_unknown_dimension"],
					0,
				)
				self.assertNotIn("projection_formulation", diagnostics)
				self.assertNotIn("nonlinear_solver", diagnostics)

	def test_projection_formulations_are_equivalent_for_every_implicit_variant(
		self,
	) -> None:
		for method_type, solver, strategy in product(
			_IMPLICIT_METHODS,
			_NONLINEAR_SOLVERS,
			_ENERGY_STRATEGIES,
		):
			extension, track_energy = strategy
			reduced_key = (
				method_type.__name__,
				"reduced_multiplier",
				solver,
				extension,
				track_energy,
			)
			simultaneous_key = (
				method_type.__name__,
				"simultaneous_state_multiplier",
				solver,
				extension,
				track_energy,
			)
			if (
				reduced_key in self.configuration_failures
				or simultaneous_key in self.configuration_failures
			):
				continue
			with self.subTest(
				method=method_type.__name__,
				solver=solver,
				extension=extension,
				track_energy=track_energy,
			):
				np.testing.assert_allclose(
					self.implicit_solutions[simultaneous_key].states,
					self.implicit_solutions[reduced_key].states,
					rtol=0.0,
					atol=5e-9 if solver == "broyden" else 5e-11,
				)

	def test_newton_and_broyden_converge_to_the_same_implicit_maps(self) -> None:
		for method_type, formulation, strategy in product(
			_IMPLICIT_METHODS,
			ABBA_PROJECTION_FORMULATIONS,
			_ENERGY_STRATEGIES,
		):
			extension, track_energy = strategy
			newton_key = (
				method_type.__name__,
				formulation,
				"newton",
				extension,
				track_energy,
			)
			broyden_key = (
				method_type.__name__,
				formulation,
				"broyden",
				extension,
				track_energy,
			)
			if (
				newton_key in self.configuration_failures
				or broyden_key in self.configuration_failures
			):
				continue
			with self.subTest(
				method=method_type.__name__,
				formulation=formulation,
				extension=extension,
				track_energy=track_energy,
			):
				np.testing.assert_allclose(
					self.implicit_solutions[broyden_key].states,
					self.implicit_solutions[newton_key].states,
					rtol=0.0,
					atol=5e-9,
				)

	def test_energy_tracking_preserves_each_physical_map(self) -> None:
		for method_type, formulation, solver in product(
			_IMPLICIT_METHODS,
			ABBA_PROJECTION_FORMULATIONS,
			_NONLINEAR_SOLVERS,
		):
			untracked_key = (
				method_type.__name__,
				formulation,
				solver,
				"physical",
				False,
			)
			tracked_key = (
				method_type.__name__,
				formulation,
				solver,
				"physical",
				True,
			)
			if (
				untracked_key in self.configuration_failures
				or tracked_key in self.configuration_failures
			):
				continue
			with self.subTest(
				method=method_type.__name__,
				formulation=formulation,
				solver=solver,
			):
				untracked = self.implicit_solutions[untracked_key]
				tracked = self.implicit_solutions[tracked_key]
				np.testing.assert_array_equal(
					tracked.states,
					untracked.states,
				)
				self.assertNotIn(
					"extended_momentum",
					untracked.diagnostics,
				)
				self.assertEqual(
					np.asarray(tracked.diagnostics["extended_momentum"]).shape,
					(1, tracked.t.size),
				)
				self.assertEqual(
					float(tracked.diagnostics["extended_momentum"][0, 0]),
					0.0,
				)
				self.assertGreaterEqual(tracked.diagnostics["energy_error"], 0.0)

		np.testing.assert_array_equal(
			self.midpoint_solutions[("physical", True)].states,
			self.midpoint_solutions[("physical", False)].states,
		)

	def test_physical_dimensions_scale_with_particle_count(self) -> None:
		problem = _problem(particle_count=3)
		for formulation, expected_unknown in (
			("reduced_multiplier", 6),
			("simultaneous_state_multiplier", 18),
		):
			solution = simulate(
				problem,
				ABBA2Implicit(projection_formulation=formulation),
				_request(),
			)
			diagnostics = solution.diagnostics
			self.assertEqual(diagnostics["accepted_internal_state_dimension"], 6)
			self.assertEqual(diagnostics["base_splitting_state_dimension"], 12)
			self.assertEqual(diagnostics["observer_state_dimension"], 6)
			self.assertEqual(
				diagnostics["nonlinear_unknown_dimension"],
				expected_unknown,
			)

		midpoint = simulate(problem, ABBA2Midpoint(), _request())
		self.assertEqual(midpoint.diagnostics["accepted_internal_state_dimension"], 6)
		self.assertEqual(midpoint.diagnostics["base_splitting_state_dimension"], 12)
		self.assertEqual(midpoint.diagnostics["nonlinear_unknown_dimension"], 0)

	def test_composed_residual_ratio_uses_one_consistent_substep(self) -> None:
		for method_type in (ABBA4Implicit, ABBA6Implicit):
			key = (
				method_type.__name__,
				"reduced_multiplier",
				"newton",
				"physical",
				False,
			)
			diagnostics = self.implicit_solutions[key].diagnostics
			substep_residuals = np.asarray(
				diagnostics["substep_nonlinear_residual_norms"]
			)
			substep_tolerances = np.asarray(
				diagnostics["substep_nonlinear_tolerances"]
			)
			worst = np.argmax(substep_residuals / substep_tolerances, axis=1)
			rows = np.arange(substep_residuals.shape[0])
			np.testing.assert_array_equal(
				diagnostics["nonlinear_residual_norms"],
				substep_residuals[rows, worst],
			)
			np.testing.assert_array_equal(
				diagnostics["nonlinear_tolerances"],
				substep_tolerances[rows, worst],
			)

	def test_simultaneous_observers_expose_the_accepted_map_jacobians(self) -> None:
		physical_problem = _problem(particle_count=3)
		for method_type, jacobian_calculator in (
			(ABBA4Implicit, abba4_implicit_step_particle_jacobians),
			(
				ABBA4ImplicitSingleProjection,
				abba4_implicit_single_projection_step_particle_jacobians,
			),
		):
			events = []
			simulate(
				physical_problem,
				method_type(
					projection_formulation="simultaneous_state_multiplier",
					state_extension="physical",
					newton_absolute_tolerance=1e-13,
					newton_relative_tolerance=1e-13,
					step_observer=events.append,
				),
				_request(),
			)
			with self.subTest(method=method_type.__name__, extension="physical"):
				self.assertEqual(len(events), 1)
				event = events[0]
				analytic = _dense_component_major_jacobian(
					jacobian_calculator(event)
				)
				numerical = central_difference_jacobian(
					event.map_state,
					event.state_before,
					relative_step=1e-5,
				)
				relative_error = float(
					np.linalg.norm(analytic - numerical, ord="fro")
					/ np.linalg.norm(numerical, ord="fro")
				)
				self.assertLess(relative_error, 3e-8)

		fully_extended_events = []
		simulate(
			_problem(),
			ABBA4ImplicitSingleProjection(
				projection_formulation="simultaneous_state_multiplier",
				state_extension="fully_extended",
				newton_absolute_tolerance=1e-13,
				newton_relative_tolerance=1e-13,
				step_observer=fully_extended_events.append,
			),
			_request(),
		)
		self.assertEqual(len(fully_extended_events), 1)
		event = fully_extended_events[0]
		self.assertEqual(event.state_before.shape, (4,))
		self.assertEqual(event.jacobian.shape, (4, 4))
		numerical = central_difference_jacobian(
			event.map_state,
			event.state_before,
			relative_step=1e-5,
		)
		relative_error = float(
			np.linalg.norm(event.jacobian - numerical, ord="fro")
			/ np.linalg.norm(numerical, ord="fro")
		)
		self.assertLess(relative_error, 3e-8)


if __name__ == "__main__":
	unittest.main()
