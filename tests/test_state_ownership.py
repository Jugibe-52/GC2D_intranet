"""Physical layout interoperability and immutable problem/dynamics ownership."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pickle
from pathlib import Path
import tempfile
from typing import ClassVar
import unittest

import numpy as np

from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.state_layout import FCStateLayout, GCStateLayout, PackedStateLayout
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from formulations.fc import FCSplitFormulation
from formulations.gc import GCDoubledMaps, GCExtendedFormulation, GCStageProjectedFormulation
from formulations.state import DoubledFormulation, PhysicalFormulation
from initial_conditions.fc import FCInitialConfiguration
from initial_conditions.gc import GCInitialConfiguration
from potential.potential import Potential
from methods.classical.rk4 import RK4
from simulation.runner import simulate


class _ExternalLayout:
	"""Structural component-major implementation without project inheritance."""

	state_dimension: ClassVar[int]

	def __init__(self) -> None:
		self.count_scale = 1
		self.position_shift = 0.0

	def validate_packed_state_layout(self, state: np.ndarray) -> np.ndarray:
		value = np.asarray(state)
		if value.ndim == 0 or not value.shape[0] or value.shape[0] % self.state_dimension:
			raise ValueError("Incomplete physical component blocks.")
		return value

	def split(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
		return tuple(np.split(self.validate_packed_state_layout(state), self.state_dimension))

	def particle_count(self, state: np.ndarray) -> int:
		return self.count_scale * self.validate_packed_state_layout(state).shape[0] // self.state_dimension

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		components = self.split(state)
		return components[0] + self.position_shift, components[1] + self.position_shift


class _ExternalGCLayout(_ExternalLayout):
	state_dimension = 2


class _ExternalFCLayout(_ExternalLayout):
	state_dimension = 4


class _ExternalConfiguration:
	"""Mutable external state provider used to exercise the ownership boundary."""

	def __init__(self, state: np.ndarray, layout: _ExternalLayout) -> None:
		self.state = state.copy()
		self.layout = layout

	@property
	def initial_state(self) -> np.ndarray:
		return self.state.copy()


class StateOwnershipTests(unittest.TestCase):
	"""Changing a source must not silently change a prepared physical problem."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.potential = Potential.random(A=0.04, M=2, nx=8, ny=8)

	def test_public_layout_routes_reexport_one_canonical_implementation(self) -> None:
		from initial_conditions import FCStateLayout as PublicFC, GCStateLayout as PublicGC
		from initial_conditions.base import PackedStateLayout as PublicPacked
		from initial_conditions.fc import FCStateLayout as ModuleFC
		from initial_conditions.gc import GCStateLayout as ModuleGC

		self.assertIs(PublicFC, FCStateLayout)
		self.assertIs(ModuleFC, FCStateLayout)
		self.assertIs(PublicGC, GCStateLayout)
		self.assertIs(ModuleGC, GCStateLayout)
		self.assertIs(PublicPacked, PackedStateLayout)

	def test_structural_gc_sources_support_both_split_formulations(self) -> None:
		state = np.array([1.0, 1.2, 0.7, 0.8])
		dynamics = GuidingCenterDynamics(self.potential, rho=0.1)
		external = InitialValueProblem(dynamics, _ExternalConfiguration(state, _ExternalGCLayout()))
		builtin = InitialValueProblem(dynamics, GCInitialConfiguration(state))
		for formulation in (GCExtendedFormulation(), GCStageProjectedFormulation()):
			for tracking in (False, True):
				with self.subTest(formulation=type(formulation).__name__, tracking=tracking):
					actual = formulation.prepare(external, track_energy=tracking)
					expected = formulation.prepare(builtin, track_energy=tracking)
					for method in ("direct_map", "adjoint_map"):
						np.testing.assert_array_equal(
							getattr(actual, method)(0.02, 0.3, actual.initial_internal_state),
							getattr(expected, method)(0.02, 0.3, expected.initial_internal_state),
						)

	def test_structural_fc_source_supports_exact_split_maps(self) -> None:
		state = np.array([1.0, 1.2, 0.7, 0.8, 0.1, 0.2, -0.2, 0.3])
		dynamics = FullCyclotronDynamics(self.potential, rho=0.2, eta=-0.1)
		external = InitialValueProblem(dynamics, _ExternalConfiguration(state, _ExternalFCLayout()))
		builtin = InitialValueProblem(dynamics, FCInitialConfiguration(state))
		for tracking in (False, True):
			actual = FCSplitFormulation().prepare(external, track_energy=tracking)
			expected = FCSplitFormulation().prepare(builtin, track_energy=tracking)
			for method in ("direct_map", "adjoint_map"):
				with self.subTest(tracking=tracking, method=method):
					np.testing.assert_array_equal(
						getattr(actual, method)(0.02, 0.3, actual.initial_internal_state),
						getattr(expected, method)(0.02, 0.3, expected.initial_internal_state),
					)

	def test_problem_and_prepared_formulations_own_state_and_layout_snapshots(self) -> None:
		state = np.array([1.0, 0.7])
		source = _ExternalConfiguration(state, _ExternalGCLayout())
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential), source)
		physical = PhysicalFormulation(problem, 0.3)
		doubled = DoubledFormulation(problem, 0.3, True)
		maps = GCDoubledMaps(problem)
		initial_maps = maps.initial_internal_state.copy()
		source.state = np.array([1.0, 1.1, 0.7, 0.8])
		source.layout.count_scale = 7
		returned = problem.initial_state
		returned[:] = -100.0
		exposed_layout = problem.layout
		assert isinstance(exposed_layout, _ExternalLayout)
		exposed_layout.count_scale = 9

		self.assertIs(problem.initial_configuration, source)
		np.testing.assert_array_equal(problem.initial_state, state)
		self.assertEqual(problem.particle_count, 1)
		self.assertEqual(problem.layout.particle_count(state), 1)
		np.testing.assert_array_equal(physical.initial_state, state)
		self.assertEqual(doubled.initial_state.shape, (6,))
		np.testing.assert_array_equal(maps.initial_internal_state, initial_maps)
		np.testing.assert_array_equal(GCDoubledMaps(problem).initial_internal_state, initial_maps)

	def test_problem_snapshot_survives_pickle_after_source_changes(self) -> None:
		state = np.array([1.0, 0.7])
		source = GCInitialConfiguration(state)
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential, rho=0.1), source)
		expected = problem.dynamics.vector_field(0.3, state)
		source.set_initial_state(np.array([1.0, 1.1, 0.7, 0.8]))
		restored = pickle.loads(pickle.dumps(problem))
		np.testing.assert_array_equal(restored.initial_state, state)
		self.assertEqual(restored.particle_count, 1)
		np.testing.assert_array_equal(restored.dynamics.vector_field(0.3, state), expected)
		self.assertEqual(GCDoubledMaps(restored).initial_internal_state.shape, (4,))

	def test_solution_uses_problem_snapshot_and_owns_its_layout(self) -> None:
		state = np.array([1.0, 0.7])
		source = _ExternalConfiguration(state, _ExternalGCLayout())
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential), source)
		source.state = np.array([1.0, 1.1, 0.7, 0.8])
		source.layout.position_shift = 100.0
		solution = simulate(problem, RK4(), SimulationRequest.uniform(t_span=(0.0, 0.01), sample_count=2))
		self.assertIs(solution.source, source)
		np.testing.assert_array_equal(solution.initial_state, state)
		np.testing.assert_array_equal(solution.positions()[0], solution.states[:1])
		returned_layout = solution.layout
		assert isinstance(returned_layout, _ExternalLayout)
		returned_layout.position_shift = 200.0
		np.testing.assert_array_equal(solution.positions()[0], solution.states[:1])

	def test_saved_solution_uses_initial_snapshot_after_source_changes(self) -> None:
		from diagnostics.persistence import load_solution, save_solution

		state = np.array([1.0, 0.7])
		source = GCInitialConfiguration(state)
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential), source)
		solution = simulate(problem, RK4(), SimulationRequest.uniform(t_span=(0.0, 0.01), sample_count=2))
		source.set_initial_state(np.array([1.0, 1.1, 0.7, 0.8]))
		with tempfile.TemporaryDirectory() as directory:
			location = Path(directory) / "trajectory"
			save_solution(solution, location, metadata={})
			restored = load_solution(location).solution
		np.testing.assert_array_equal(restored.initial_state, state)
		np.testing.assert_array_equal(restored.states, solution.states)

	def test_physical_parameters_are_frozen_before_and_after_pickle(self) -> None:
		cases = (
			(GuidingCenterDynamics(self.potential, rho=0.1), {"rho": 0.2, "potential": self.potential,
				"effective_potential": self.potential}, np.array([1.0, 0.7])),
			(FullCyclotronDynamics(self.potential, rho=0.2, eta=-0.1),
				{"rho": 0.3, "eta": 0.2, "potential": self.potential}, np.array([1.0, 0.7, 0.1, -0.2])),
		)
		for original, changes, state in cases:
			expected = original.vector_field(0.3, state)
			for dynamics in (original, pickle.loads(pickle.dumps(original))):
				for name, value in changes.items():
					with self.subTest(dynamics=type(dynamics).__name__, parameter=name):
						with self.assertRaises(FrozenInstanceError):
							setattr(dynamics, name, value)
				np.testing.assert_array_equal(dynamics.vector_field(0.3, state), expected)


if __name__ == "__main__":
	unittest.main()
