"""Physical layout interoperability and immutable problem/dynamics ownership."""

from __future__ import annotations


from dataclasses import FrozenInstanceError
import pickle
from pathlib import Path
import tempfile
from typing import ClassVar
import unittest

import numpy as np

from problem import InitialValueProblem
from request import SimulationRequest
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from dynamics.coordinates import CoordinateValues, PhysicalCoordinates
from methods.splitting import FCSplitMaps
from methods.extended.maps import GCDoubledMaps
from formulations.state import DoubleFormulation, SingleFormulation

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
	component_names = ('x', 'y')
	position_names = ('x', 'y')
	velocity_names = ()


class _ExternalFCLayout(_ExternalLayout):
	state_dimension = 4
	component_names = ('x', 'y', 'vx', 'vy')
	position_names = ('x', 'y')
	velocity_names = ('vx', 'vy')


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

	def test_structural_sources_support_method_owned_maps(self):
		for dynamics, layout, maps_type, copies in (
			(GuidingCenterDynamics(self.potential), _ExternalGCLayout(), GCDoubledMaps, 2),
			(FullCyclotronDynamics(self.potential, rho=.2, eta=-.1), _ExternalFCLayout(), FCSplitMaps, 1)):
			state = np.arange(layout.state_dimension * 2, dtype=float) / 10
			external = InitialValueProblem(dynamics, state, layout=layout)
			for tracking in (False, True):
				actual = maps_type(dynamics, state.size, coordinates=external.layout, track_energy=tracking)
				expected = maps_type(dynamics, state.size, track_energy=tracking)
				work = np.tile(state, copies)
				if tracking: work = np.r_[work, [.1, -.2]]
				for operation in ('direct_map', 'adjoint_map'):
					np.testing.assert_array_equal(getattr(actual, operation)(.02, .3, work), getattr(expected, operation)(.02, .3, work))
				self.assertFalse(hasattr(actual, 'initial_internal_state'))
				self.assertFalse(hasattr(actual, 'configuration'))

	def test_problem_and_prepared_formulations_own_state_and_layout_snapshots(self) -> None:
		state = np.array([1.0, 0.7])
		source = _ExternalConfiguration(state, _ExternalGCLayout())
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential), source.initial_state, layout=source.layout)
		physical = SingleFormulation(state.size)
		doubled = DoubleFormulation(state.size)
		maps = GCDoubledMaps(problem.dynamics, state.size)
		initial_maps = doubled.lift(state)
		source.state = np.array([1.0, 1.1, 0.7, 0.8])
		source.layout.count_scale = 7
		returned = problem.initial_state
		returned[:] = -100.0
		exposed_layout = problem.layout
		with self.assertRaises(FrozenInstanceError):
			exposed_layout.component_names = ('y', 'x')

		self.assertFalse(np.shares_memory(problem.initial_state, source.state))
		np.testing.assert_array_equal(problem.initial_state, state)
		self.assertEqual(problem.particle_count, 1)
		self.assertEqual(problem.layout.particle_count(state), 1)
		np.testing.assert_array_equal(physical.lift(problem.initial_state), state)
		self.assertEqual(doubled.lift(problem.initial_state).shape, (4,))
		np.testing.assert_array_equal(maps.state_formulation.lift(problem.initial_state), initial_maps)
		self.assertFalse(hasattr(maps, 'initial_internal_state'))

	def test_problem_snapshot_survives_pickle_after_source_changes(self) -> None:
		state = np.array([1.0, 0.7])
		source = _ExternalConfiguration(state, _ExternalGCLayout())
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential, rho=0.1), source.initial_state, layout=source.layout)
		expected = problem.dynamics.vector_field(0.3, state)
		source.state = np.array([1.0, 1.1, 0.7, 0.8])
		restored = pickle.loads(pickle.dumps(problem))
		np.testing.assert_array_equal(restored.initial_state, state)
		self.assertEqual(restored.particle_count, 1)
		np.testing.assert_array_equal(restored.dynamics.vector_field(0.3, state), expected)
		self.assertEqual(GCDoubledMaps(restored.dynamics, state.size).state_formulation.lift(restored.initial_state).shape, (4,))

	def test_solution_uses_problem_snapshot_and_owns_its_layout(self) -> None:
		state = np.array([1.0, 0.7])
		source = _ExternalConfiguration(state, _ExternalGCLayout())
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential), source.initial_state, layout=source.layout)
		source.state = np.array([1.0, 1.1, 0.7, 0.8])
		source.layout.position_shift = 100.0
		solution = simulate(problem, RK4(step=0.1), SimulationRequest.uniform(t_span=(0.0, 0.01), sample_count=2))
		self.assertFalse(np.shares_memory(solution.initial_state, source.state))
		np.testing.assert_array_equal(solution.initial_state, state)
		np.testing.assert_array_equal(solution.positions()[0], solution.states[:1])
		returned_layout = solution.layout
		with self.assertRaises(FrozenInstanceError):
			returned_layout.position_names = ('y', 'x')
		np.testing.assert_array_equal(solution.positions()[0], solution.states[:1])

	def test_saved_solution_uses_initial_snapshot_after_source_changes(self) -> None:
		from diagnostics.io.persistence import load_solution, save_solution

		state = np.array([1.0, 0.7])
		source = _ExternalConfiguration(state, GuidingCenterDynamics.coordinates)
		problem = InitialValueProblem(GuidingCenterDynamics(self.potential), source.initial_state, layout=source.layout)
		solution = simulate(problem, RK4(step=0.1), SimulationRequest.uniform(t_span=(0.0, 0.01), sample_count=2))
		source.state = np.array([1.0, 1.1, 0.7, 0.8])
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
