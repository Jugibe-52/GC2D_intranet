"""Fast contract tests for the notebook-facing simulation core."""

from __future__ import annotations

import unittest

import numpy as np

from dynamics import (
	FullCyclotronDynamics,
	GuidingCenterDynamics,
)
from initial_conditions import (
	Area,
	FCInitialConfiguration,
	GCInitialConfiguration,
	TrajectoryFC,
	TrajectoryGC,
)
from potential import Potential
from simulation import (
	BM4Composition,
	FCSplitFormulation,
	GCExtendedFormulation,
	GCStageProjectedFormulation,
	InitialValueProblem,
	IntegrationStage,
	ProjectedBM4Composition,
	SimulationRequest,
	simulate,
)
from visualization import animate_gc_area_comparison, animate_gc_area_solution


def random_potential(*, interpolation_order: int = 3) -> Potential:
	"""Return the small deterministic field shared by the fast tests.

	``A`` fixes the spectral amplitude, ``M`` the radial wave-number cutoff and
	``nx``/``ny`` the number of samples along each periodic spatial axis.
	"""
	return Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=interpolation_order,
	)


def gc_problem(
	trajectory: TrajectoryGC,
	potential: Potential | None = None,
) -> InitialValueProblem:
	"""Build a guiding-centre problem for fast simulation tests."""
	field = random_potential() if potential is None else potential
	return InitialValueProblem(
		GuidingCenterDynamics(field, rho=trajectory.rho),
		trajectory,
	)


def fc_problem(
	trajectory: TrajectoryFC,
	potential: Potential | None = None,
) -> InitialValueProblem:
	"""Build a full-cyclotron problem for fast simulation tests."""
	field = random_potential() if potential is None else potential
	return InitialValueProblem(
		FullCyclotronDynamics(field, rho=trajectory.rho, eta=trajectory.eta),
		trajectory,
	)


def uniform_request(
	*,
	step: float,
	t_span: tuple[float, float],
	sample_count: int,
) -> SimulationRequest:
	"""Build the common uniform temporal request used by contract tests."""
	return SimulationRequest.uniform(
		t_span=t_span,
		max_step=step,
		sample_count=sample_count,
	)


class PotentialTests(unittest.TestCase):
	"""Contracts for field construction, evaluation and interpolation."""

	def test_random_is_deterministic_and_derivatives_are_finite(self) -> None:
		first = random_potential()
		second = random_potential()

		np.testing.assert_allclose(first.evaluate(0.3), second.evaluate(0.3))
		times = np.asarray([0.0, 0.2, 0.5])
		fields = first.evaluate(times)
		self.assertEqual(fields.shape, first.grid.shape + (times.size,))
		for index, time in enumerate(times):
			np.testing.assert_allclose(fields[..., index], first.evaluate(time))

		x = np.asarray([0.7, 1.4, 2.1])
		y = np.asarray([0.9, 1.7, 2.5])
		for derivative in (
			first.evaluate(0.3, x, y, dx=1),
			first.evaluate(0.3, x, y, dy=1),
			first.evaluate(0.3, x, y, dt=1),
		):
			self.assertTrue(np.all(np.isfinite(derivative)))

		ex, ey = first.electric_field(0.3, x, y)
		np.testing.assert_allclose(ex, -first.evaluate(0.3, x, y, dx=1))
		np.testing.assert_allclose(ey, -first.evaluate(0.3, x, y, dy=1))

	def test_gyroaverage_preserves_the_original_potential(self) -> None:
		potential = random_potential(interpolation_order=5)
		original = potential.evaluate(0.2).copy()

		averaged = potential.gyroaverage(0.1)

		self.assertIsInstance(averaged, Potential)
		self.assertTrue(np.all(np.isfinite(averaged.evaluate(0.2))))
		np.testing.assert_allclose(potential.evaluate(0.2), original)
		self.assertTrue(callable(potential.plot))
		self.assertTrue(callable(potential.animate))

		with self.assertRaises(ValueError):
			Potential.random(A=0.1, M=2, nx=0, ny=8, interpolation_order=3)
		with self.assertRaises(ValueError):
			Potential.random(A=0.1, M=2, nx=8, ny=8, interpolation_order=1)


class TrajectoryTests(unittest.TestCase):
	"""Contracts for physical-state layouts and finite-area boundaries."""

	def test_canonical_configurations_contain_no_physical_parameters(self) -> None:
		"""Keep state layout independent from the selected dynamical model."""
		gc = GCInitialConfiguration.from_components(
			x=np.asarray([1.0]),
			y=np.asarray([2.0]),
		)
		fc = FCInitialConfiguration.from_components(
			x=np.asarray([1.0]),
			y=np.asarray([2.0]),
			vx=np.asarray([0.5]),
			vy=np.asarray([-0.5]),
		)

		self.assertFalse(hasattr(gc, "rho"))
		self.assertFalse(hasattr(gc, "eta"))
		self.assertFalse(hasattr(fc, "rho"))
		self.assertFalse(hasattr(fc, "eta"))

	def test_gc_state_layout_and_copy(self) -> None:
		# A GC state is component-major: all particle x values precede all y values.
		state = np.asarray([1.0, 2.0, 3.0, 4.0])
		trajectory = TrajectoryGC(state, rho=0.2)
		state[0] = -10.0

		stored = trajectory.state
		self.assertIsNotNone(stored)
		assert stored is not None
		np.testing.assert_allclose(stored, [1.0, 2.0, 3.0, 4.0])
		x, y = trajectory.positions(stored)
		np.testing.assert_allclose(x, [1.0, 2.0])
		np.testing.assert_allclose(y, [3.0, 4.0])
		components = trajectory.split(stored)
		np.testing.assert_allclose(components.x, x)
		np.testing.assert_allclose(components.y, y)
		np.testing.assert_allclose(trajectory.pack_components(*components), stored)
		self.assertEqual(trajectory.particle_count(stored), 2)
		self.assertIs(trajectory.validate_packed_state(stored), stored)
		with self.assertRaises(ValueError):
			trajectory.validate_packed_state(np.asarray([1.0, 2.0, 3.0]))

		stored[0] = -20.0
		np.testing.assert_allclose(trajectory.state, [1.0, 2.0, 3.0, 4.0])

	def test_component_constructors_and_explicit_block_views(self) -> None:
		"""Named constructors hide packing while block transforms remain views."""
		x = np.asarray([1.0, 2.0])
		y = np.asarray([3.0, 4.0])
		np.testing.assert_allclose(
			TrajectoryGC.pack_components(x, y),
			[1.0, 2.0, 3.0, 4.0],
		)
		gc = TrajectoryGC.from_components(x=x, y=y, rho=0.2)
		gc_state = gc.state
		assert gc_state is not None
		np.testing.assert_allclose(gc_state, [1.0, 2.0, 3.0, 4.0])

		blocks = gc.as_blocks(gc_state)
		self.assertEqual(blocks.shape, (2, 2))
		self.assertTrue(np.shares_memory(blocks, gc_state))
		flat = gc.from_blocks(blocks)
		self.assertTrue(np.shares_memory(flat, blocks))
		np.testing.assert_allclose(flat, gc_state)

		fc = TrajectoryFC.from_components(
			x=x,
			y=y,
			vx=np.asarray([0.5, 0.6]),
			vy=np.asarray([-0.5, -0.6]),
			rho=0.4,
			eta=-0.2,
		)
		fc_state = fc.state
		assert fc_state is not None
		np.testing.assert_allclose(
			fc_state,
			[1.0, 2.0, 3.0, 4.0, 0.5, 0.6, -0.5, -0.6],
		)
		self.assertEqual(fc.as_blocks(fc_state).shape, (4, 2))

		with self.assertRaises(ValueError):
			TrajectoryGC.from_components(x=x, y=y[:-1], rho=0.2)

	def test_fc_state_layout_and_scales(self) -> None:
		# FC appends velocity blocks to the GC position blocks: [x, y, vx, vy].
		state = np.asarray([1.0, 2.0, 3.0, 4.0, 0.5, 0.6, -0.5, -0.6])
		trajectory = TrajectoryFC(state, rho=0.4, eta=-0.2)

		x, y = trajectory.positions(state)
		vx, vy = trajectory.velocities(state)
		np.testing.assert_allclose(x, [1.0, 2.0])
		np.testing.assert_allclose(y, [3.0, 4.0])
		np.testing.assert_allclose(vx, [0.5, 0.6])
		np.testing.assert_allclose(vy, [-0.5, -0.6])
		self.assertAlmostEqual(trajectory.velocity_scale, 1.0)
		self.assertAlmostEqual(trajectory.electric_scale, -2.5)
		self.assertAlmostEqual(trajectory.larmor_frequency, -2.5)
		components = trajectory.split(state)
		np.testing.assert_allclose(components.x, x)
		np.testing.assert_allclose(components.y, y)
		np.testing.assert_allclose(components.vx, vx)
		np.testing.assert_allclose(components.vy, vy)
		np.testing.assert_allclose(trajectory.pack_components(*components), state)
		self.assertEqual(trajectory.particle_count(state), 2)

		with self.assertRaises(ValueError):
			trajectory.pack_components(x, y, vx)
		with self.assertRaises(ValueError):
			trajectory.pack_components(x, y[:-1], vx, vy)

	def test_area_constructors_and_area_calculation(self) -> None:
		# This center makes the unit square cross both periodic cell boundaries.
		square = Area.square(
			center=(2 * np.pi - 0.25, 2 * np.pi - 0.25),
			side=1.0,
			points_per_side=4,
			rho=0.2,
		)
		self.assertIsInstance(square, GCInitialConfiguration)
		self.assertEqual(square.shape, "square")
		self.assertAlmostEqual(float(square.calculate_area()), 1.0)

		square_state = square.state
		assert square_state is not None
		x, y = square.positions(square_state)
		# Force every vertex back into the base cell to exercise periodic unwrapping.
		wrapped = np.concatenate((x % (2 * np.pi), y % (2 * np.pi)))
		self.assertAlmostEqual(
			float(square.calculate_area(wrapped, period=2 * np.pi)),
			1.0,
		)

		circle = Area.circle(center=(1.0, 2.0), radius=0.5, points=128)
		self.assertEqual(circle.shape, "circle")
		# The sampled circle is a regular polygon, so its exact discrete area is used.
		expected_polygon_area = 128 * 0.5**2 * np.sin(2 * np.pi / 128) / 2
		self.assertAlmostEqual(
			float(circle.calculate_area()),
			expected_polygon_area,
		)

		circle_state = circle.state
		assert circle_state is not None
		# Columns are successive saved times; translation must not change the area.
		time_series = np.column_stack((circle_state, circle_state + 0.1))
		areas = circle.calculate_area(time_series)
		self.assertEqual(areas.shape, (2,))
		np.testing.assert_allclose(areas, expected_polygon_area)

		solution = simulate(
			gc_problem(square),
			BM4Composition(GCExtendedFormulation()),
			uniform_request(
				step=0.01,
				t_span=(0.0, 0.02),
				sample_count=3,
			),
		)
		transported_area = square.calculate_area(solution.y, period=2 * np.pi)
		self.assertEqual(transported_area.shape, (3,))
		self.assertTrue(np.all(np.isfinite(transported_area)))


class SimulationTests(unittest.TestCase):
	"""Contracts for composition, BM4 integration and study visualizations."""

	def test_gc_coupling_frequency_is_owned_by_the_formulation(self) -> None:
		default = GCExtendedFormulation()
		configured = GCExtendedFormulation(coupling_frequency=2.5)

		self.assertAlmostEqual(default.coupling_frequency, np.pi / 8)
		self.assertEqual(configured.coupling_frequency, 2.5)
		self.assertEqual(
			GCExtendedFormulation(coupling_frequency=0.0).coupling_frequency,
			0.0,
		)
		with self.assertRaises(ValueError):
			GCExtendedFormulation(coupling_frequency=-0.1)
		with self.assertRaises(ValueError):
			GCExtendedFormulation(coupling_frequency=np.inf)

	def test_composition_stage_observer_receives_fixed_gc_maps(self) -> None:
		"""Observers see twelve immutable stage snapshots per complete BM4 step."""
		trajectory = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		problem = gc_problem(trajectory)
		request = uniform_request(
			step=0.01,
			t_span=(0.0, 0.01),
			sample_count=2,
		)
		events: list[IntegrationStage] = []
		solution = simulate(
			problem,
			BM4Composition(
				GCExtendedFormulation(),
				stage_observer=events.append,
			),
			request,
		)
		reference = simulate(
			problem,
			BM4Composition(GCExtendedFormulation()),
			request,
		)

		self.assertEqual(solution.n_steps, 1)
		np.testing.assert_array_equal(solution.y, reference.y)
		self.assertEqual(len(events), 12)
		self.assertEqual([event.stage_index for event in events], list(range(12)))
		self.assertEqual(events[0].flow_name, "adjoint_flow")
		self.assertEqual(events[1].flow_name, "flow")
		for event in events:
			self.assertEqual(event.dynamics_name, "GuidingCenterDynamics")
			self.assertEqual(event.formulation_name, "GCExtendedFormulation")
			self.assertEqual(event.method_name, "BM4Composition")
			self.assertEqual(event.step_index, 0)
			np.testing.assert_allclose(
				event.map_state(event.state_before),
				event.state_after,
			)

	def test_projected_bm4_reembeds_gc_copies_after_every_stage(self) -> None:
		"""Every projected direct or adjoint stage ends on the GC diagonal."""
		trajectory = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		problem = gc_problem(trajectory)
		events: list[IntegrationStage] = []
		solution = simulate(
			problem,
			ProjectedBM4Composition(
				GCStageProjectedFormulation(),
				track_energy=True,
				stage_observer=events.append,
			),
			uniform_request(
				step=0.01,
				t_span=(0.0, 0.02),
				sample_count=3,
			),
		)

		self.assertEqual(solution.n_steps, 2)
		self.assertEqual(len(events), 24)
		initial_state = trajectory.initial_state
		assert initial_state is not None
		physical_size = initial_state.size
		for event in events:
			state = event.state_after
			np.testing.assert_allclose(
				state[:physical_size],
				state[physical_size : 2 * physical_size],
			)
			np.testing.assert_allclose(
				event.map_state(event.state_before),
				event.state_after,
			)
		self.assertEqual(events[-1].method_name, "ProjectedBM4Composition")
		self.assertEqual(
			events[-1].formulation_name,
			"GCStageProjectedFormulation",
		)
		self.assertTrue(np.all(np.isfinite(solution.states)))
		self.assertTrue(np.isfinite(float(solution.err)))

		with self.assertRaises(TypeError):
			simulate(
				problem,
				ProjectedBM4Composition(GCExtendedFormulation()),
				uniform_request(
					step=0.01,
					t_span=(0.0, 0.01),
					sample_count=2,
				),
			)

	def test_output_grid_uses_shadow_steps_without_changing_bm4_path(self) -> None:
		"""Shared output times and the BM4 path are output-grid independent."""
		gc_trajectory = TrajectoryGC(
			np.asarray([1.0, 1.2]),
			rho=0.05,
		)
		fc_trajectory = TrajectoryFC(
			np.asarray([1.0, 1.2, 0.4, -0.3]),
			rho=0.2,
			eta=0.1,
		)
		cases = (
			(
				gc_problem(gc_trajectory),
				BM4Composition(GCExtendedFormulation()),
			),
			(
				fc_problem(fc_trajectory),
				BM4Composition(FCSplitFormulation()),
			),
		)

		for problem, method in cases:
			with self.subTest(dynamics=type(problem.dynamics).__name__):
				sparse = simulate(
					problem,
					method,
					uniform_request(
						step=0.02,
						t_span=(0.0, 0.05),
						sample_count=3,
					),
				)
				dense = simulate(
					problem,
					method,
					uniform_request(
						step=0.02,
						t_span=(0.0, 0.05),
						sample_count=7,
					),
				)

				# The common midpoint is a shadow sample rather than a BM4 node.
				self.assertEqual(sparse.n_steps, 3)
				self.assertEqual(dense.n_steps, 3)
				self.assertEqual(sparse.t[1], dense.t[3])
				np.testing.assert_array_equal(sparse.y[:, 1], dense.y[:, 3])
				np.testing.assert_array_equal(sparse.y[:, -1], dense.y[:, -1])

		events: list[IntegrationStage] = []
		observed = simulate(
			cases[0][0],
			BM4Composition(
				GCExtendedFormulation(),
				stage_observer=events.append,
			),
			uniform_request(
				step=0.02,
				t_span=(0.0, 0.05),
				sample_count=11,
			),
		)
		# Shadow maps do not emit diagnostic stages.
		self.assertEqual(observed.n_steps, 3)
		self.assertEqual(len(events), 12 * observed.n_steps)
		self.assertEqual(sorted({event.step_index for event in events}), [0, 1, 2])

	def test_gc_area_animation_tracks_relative_error(self) -> None:
		area = Area.square(
			center=(np.pi, np.pi),
			side=1.0,
			points_per_side=4,
			rho=0.05,
		)
		dynamics = GuidingCenterDynamics(random_potential(), rho=area.rho)
		solution = simulate(
			InitialValueProblem(dynamics, area),
			BM4Composition(GCExtendedFormulation()),
			uniform_request(
				step=0.01,
				t_span=(0.0, 0.02),
				sample_count=3,
			),
		)

		animation = animate_gc_area_solution(
			dynamics.effective_potential,
			area,
			solution,
			frames=3,
			interval=20,
		)
		# Calling the frame callback checks synchronized artists without rendering HTML.
		artists = animation._func(2)
		self.assertEqual(len(artists), 6)
		self.assertEqual(artists[1].__class__.__name__, "Quiver")
		expected_area = area.calculate_area(solution.y, period=2 * np.pi)
		expected_error = (expected_area - expected_area[0]) / abs(expected_area[0])
		np.testing.assert_allclose(artists[3].get_xdata(), solution.t)
		np.testing.assert_allclose(artists[3].get_ydata(), expected_error)
		self.assertIn("varepsilon_A", artists[5].get_text())
		self.assertIn("max", animation._fig.axes[1].get_title())
		# Mark the test animation as consumed so Matplotlib does not emit a warning.
		animation._draw_was_started = True

		diagnostic_times = np.asarray(solution.t)
		relative_symplecticity_errors = np.asarray([0.0, 1e-12, 2e-12])
		relative_copy_separations = np.asarray([0.0, 3e-13, 5e-13])
		diagnostic_animation = animate_gc_area_solution(
			dynamics.effective_potential,
			area,
			solution,
			frames=3,
			interval=20,
			diagnostic_times=diagnostic_times,
			relative_symplecticity_errors=relative_symplecticity_errors,
			relative_copy_separations=relative_copy_separations,
		)
		diagnostic_artists = diagnostic_animation._func(2)
		self.assertEqual(len(diagnostic_artists), 12)
		np.testing.assert_allclose(diagnostic_artists[6].get_xdata(), solution.t)
		self.assertAlmostEqual(diagnostic_artists[6].get_ydata()[-1], 2e-12)
		self.assertAlmostEqual(diagnostic_artists[8].get_ydata()[-1], 5e-13)
		self.assertIn("max", diagnostic_artists[10].get_text())
		self.assertIn("max", diagnostic_artists[11].get_text())
		diagnostic_animation._draw_was_started = True

		comparison_solutions = {
			"dt=0.02": solution,
			"dt=0.01": solution,
			"dt=0.005": solution,
		}
		comparison_times = {
			label: diagnostic_times for label in comparison_solutions
		}
		comparison_defects = {
			label: relative_symplecticity_errors * (index + 1)
			for index, label in enumerate(comparison_solutions)
		}
		comparison_separations = {
			label: relative_copy_separations * (index + 1)
			for index, label in enumerate(comparison_solutions)
		}
		comparison_animation = animate_gc_area_comparison(
			dynamics.effective_potential,
			area,
			comparison_solutions,
			diagnostic_times=comparison_times,
			relative_symplecticity_errors=comparison_defects,
			relative_copy_separations=comparison_separations,
			frames=3,
			interval=20,
		)
		comparison_artists = comparison_animation._func(2)
		self.assertEqual(len(comparison_artists), 26)
		# One color consistently identifies a step in every animated panel.
		self.assertEqual(
			comparison_artists[2].get_color(),
			comparison_artists[5].get_color(),
		)
		self.assertEqual(
			comparison_artists[2].get_color(),
			comparison_artists[12].get_color(),
		)
		self.assertEqual(
			comparison_artists[2].get_color(),
			comparison_artists[18].get_color(),
		)
		np.testing.assert_allclose(comparison_artists[5].get_xdata(), solution.t)
		comparison_animation._draw_was_started = True

		plain_trajectory = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		with self.assertRaises(TypeError):
			animate_gc_area_solution(
				dynamics.effective_potential,
				plain_trajectory,  # type: ignore[arg-type]
				solution,
			)
		with self.assertRaises(ValueError):
			animate_gc_area_solution(
				dynamics.effective_potential,
				area,
				solution,
				frames=1,
			)
		with self.assertRaises(ValueError):
			animate_gc_area_solution(
				dynamics.effective_potential,
				area,
				solution,
				diagnostic_times=diagnostic_times,
			)
		with self.assertRaises(ValueError):
			animate_gc_area_comparison(
				dynamics.effective_potential,
				area,
				{"only": solution},
			)

	def test_problem_and_request_validate_initial_state_and_time_span(self) -> None:
		with self.assertRaises(ValueError):
			gc_problem(TrajectoryGC(rho=0.05))

		trajectory = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		with self.assertRaises(ValueError):
			uniform_request(
				step=0.01,
				t_span=(1.0, 1.0),
				sample_count=3,
			)

	def test_gc_bm4_simulation_tracks_generalized_energy(self) -> None:
		trajectory = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		problem = gc_problem(trajectory)
		solution = simulate(
			problem,
			BM4Composition(
				GCExtendedFormulation(),
				track_energy=True,
			),
			uniform_request(
				step=0.01,
				t_span=(0.0, 0.04),
				sample_count=5,
			),
		)

		np.testing.assert_allclose(solution.t, np.linspace(0.0, 0.04, 5))
		self.assertEqual(solution.y.shape, (2, 5))
		# Extended momentum has one row per simulated particle.
		self.assertEqual(np.asarray(solution.k).shape, (1, 5))
		self.assertGreater(solution.n_steps, 0)
		self.assertTrue(np.all(np.isfinite(np.asarray(solution.err))))
		self.assertTrue(
			np.all(
				np.isfinite(
					problem.dynamics.hamiltonian(solution.t, solution.y)
				)
			)
		)
		self.assertIs(solution.trajectory, trajectory)
		components = solution.components()
		self.assertEqual(components.x.shape, (1, 5))
		self.assertEqual(components.y.shape, (1, 5))
		with self.assertRaises(TypeError):
			solution.components(TrajectoryFC(rho=0.2, eta=0.1))

	def test_fc_bm4_simulation_tracks_generalized_energy(self) -> None:
		trajectory = TrajectoryFC(
			np.asarray([1.0, 1.2, 0.4, -0.3]),
			rho=0.2,
			eta=0.1,
		)
		problem = fc_problem(trajectory)
		solution = simulate(
			problem,
			BM4Composition(
				FCSplitFormulation(),
				track_energy=True,
			),
			uniform_request(
				step=0.01,
				t_span=(0.0, 0.04),
				sample_count=5,
			),
		)

		np.testing.assert_allclose(solution.t, np.linspace(0.0, 0.04, 5))
		self.assertEqual(solution.y.shape, (4, 5))
		# Extended momentum is stripped from y and exposed separately as k.
		self.assertEqual(np.asarray(solution.k).shape, (1, 5))
		self.assertGreater(solution.n_steps, 0)
		self.assertTrue(np.all(np.isfinite(np.asarray(solution.err))))
		self.assertTrue(
			np.all(
				np.isfinite(
					problem.dynamics.hamiltonian(solution.t, solution.y)
				)
			)
		)
		self.assertIs(solution.trajectory, trajectory)
		components = solution.components()
		self.assertEqual(components.x.shape, (1, 5))
		self.assertEqual(components.vx.shape, (1, 5))


if __name__ == "__main__":
	unittest.main()
