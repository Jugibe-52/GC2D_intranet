"""Contracts for full ``(z,t,k)`` duplication and projection methods."""

from __future__ import annotations

from itertools import product
import unittest

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import central_difference_jacobian
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA4ImplicitSingleProjection,
	BM4Implicit2,
	BM4_implicit2,
	FullyExtendedImplicitIntegrationStep,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from studies import (
	FULLY_EXTENDED_IMPLICIT_METHODS,
	FullyExtendedImplicitConfig,
	run_fully_extended_implicit_study,
)
from visualization import (
	plot_fully_extended_symplecticity,
	plot_generalized_energy_components,
	plot_generalized_energy_convergence,
	plot_generalized_energy_errors,
)


def _problem() -> tuple[Potential, GCInitialConfiguration, InitialValueProblem]:
	"""Return one inexpensive smooth non-autonomous one-particle problem."""
	potential = Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)
	configuration = GCInitialConfiguration.from_components(
		x=np.asarray([1.0]),
		y=np.asarray([1.2]),
	)
	dynamics = GuidingCenterDynamics(potential, rho=0.05)
	return potential, configuration, InitialValueProblem(dynamics, configuration)


class FullyExtendedImplicitMethodTests(unittest.TestCase):
	"""Verify the full state, diagonal projection, and public identifiers."""

	def test_canonical_abba_names_do_not_replace_historical_bm4_formulation(self) -> None:
		self.assertEqual(ABBA2Implicit.__name__, "ABBA2Implicit")
		self.assertEqual(ABBA4Implicit.__name__, "ABBA4Implicit")
		self.assertEqual(BM4_implicit2.__name__, "BM4_implicit2")
		self.assertIsNot(BM4_implicit2, BM4Implicit2)

	def test_one_step_projects_all_four_extended_coordinates(self) -> None:
		_, _, problem = _problem()
		records: list[FullyExtendedImplicitIntegrationStep] = []
		solution = simulate(
			problem,
			ABBA2Implicit(
				state_extension="fully_extended",
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
				step_observer=records.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.1),
				max_step=0.1,
				sample_count=2,
			),
		)
		self.assertEqual(len(records), 1)
		record = records[0]
		self.assertEqual(record.state_before.shape, (4,))
		self.assertEqual(record.state_after.shape, (4,))
		self.assertEqual(record.multiplier.shape, (4,))
		self.assertEqual(record.base_maps[0].state_before.shape, (8,))
		self.assertAlmostEqual(record.state_after[2], 0.1)
		np.testing.assert_allclose(record.state_after[:2], solution.states[:, -1])
		corrected_first = record.base_maps[0].state_after[:4] + record.multiplier
		corrected_second = record.base_maps[0].state_after[4:] - record.multiplier
		np.testing.assert_allclose(
			corrected_first,
			corrected_second,
			rtol=0.0,
			atol=2e-14,
		)
		self.assertAlmostEqual(
			float(solution.diagnostics["extended_time"][-1]),
			0.1,
		)
		self.assertIs(solution.diagnostics["track_energy"], True)
		self.assertEqual(
			solution.diagnostics["extended_momentum_normalization"],
			"direct_k",
		)
		generalized_error = np.asarray(
			solution.diagnostics["generalized_energy_error"]
		)
		self.assertEqual(
			solution.diagnostics["energy_error"],
			float(np.max(np.abs(generalized_error))),
		)
		self.assertEqual(
			solution.diagnostics["projection_jacobian"],
			"analytic_stage_product",
		)

		base_map = record.base_maps[0]
		analytic_dpsi = base_map.jacobian_state(base_map.state_before)
		numerical_dpsi = central_difference_jacobian(
			base_map.map_state,
			base_map.state_before,
		)
		self.assertLess(
			float(
				np.linalg.norm(analytic_dpsi - numerical_dpsi, ord="fro")
				/ np.linalg.norm(analytic_dpsi, ord="fro")
			),
			1e-8,
		)

		physical_start = 0.5 * (
			base_map.state_before[:4] + base_map.state_before[4:]
		)

		def residual_map(multiplier: np.ndarray) -> np.ndarray:
			mapped = base_map.map_state(
				np.concatenate(
					(physical_start + multiplier, physical_start - multiplier)
				)
			)
			return np.asarray(mapped[:4] - mapped[4:] + 2.0 * multiplier)

		numerical_dr = central_difference_jacobian(
			residual_map,
			base_map.projection_multiplier,
		)
		self.assertLess(
			float(
				np.linalg.norm(
					base_map.residual_jacobian - numerical_dr,
					ord="fro",
				)
				/ np.linalg.norm(base_map.residual_jacobian, ord="fro")
			),
			1e-8,
		)
		numerical_projected = central_difference_jacobian(
			record.map_state,
			record.state_before,
		)
		self.assertLess(
			float(
				np.linalg.norm(record.jacobian - numerical_projected, ord="fro")
				/ np.linalg.norm(record.jacobian, ord="fro")
			),
			1e-8,
		)

	def test_long_abba4_runs_pin_extended_time_to_the_output_grid(self) -> None:
		"""Do not accumulate roundoff in the exactly solvable time coordinate."""
		_, _, problem = _problem()
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.84),
			max_step=0.0025,
			sample_count=85,
		)
		for method_type, formulation, nonlinear_solver in product(
			(ABBA4Implicit, ABBA4ImplicitSingleProjection),
			("reduced_multiplier", "simultaneous_state_multiplier"),
			("newton", "broyden"),
		):
			with self.subTest(
				method=method_type.__name__,
				formulation=formulation,
				nonlinear_solver=nonlinear_solver,
			):
				solution = simulate(
					problem,
					method_type(
						state_extension="fully_extended",
						projection_formulation=formulation,
						nonlinear_solver=nonlinear_solver,
						newton_absolute_tolerance=1e-14,
						newton_relative_tolerance=1e-13,
						newton_max_iterations=40,
					),
					request,
				)
				np.testing.assert_array_equal(
					solution.diagnostics["extended_time"],
					request.output_times,
				)
				self.assertIs(solution.diagnostics["track_energy"], True)


class FullyExtendedImplicitStudyTests(unittest.TestCase):
	"""Verify energy orders, both symplecticity spaces, and plots."""

	def test_three_methods_return_expected_short_refinements(self) -> None:
		potential, configuration, _ = _problem()
		config = FullyExtendedImplicitConfig(
			steps=(0.1, 0.05),
			t_span=(0.0, 0.2),
			output_sample_count=3,
			rho=0.05,
		)
		for method in FULLY_EXTENDED_IMPLICIT_METHODS:
			with self.subTest(method=method):
				result = run_fully_extended_implicit_study(
					potential,
					configuration,
					method=method,
					config=config,
				)
				self.assertEqual(len(result.runs), 2)
				order = result.convergence_orders()[0].maximum_error_order
				minimum_order = (
					1.8 if method == "abba2_fully_extended_implicit" else 3.8
				)
				self.assertGreater(order, minimum_order)
				for run, step_count in zip(result.runs, (2, 4), strict=True):
					self.assertEqual(run.solution.n_steps, step_count)
					self.assertEqual(len(run.energy_records), step_count + 1)
					self.assertEqual(len(run.symplecticity_records), step_count)
					np.testing.assert_allclose(
						run.generalized_energy,
						run.hamiltonian + run.k,
					)
					self.assertLess(float(np.max(run.r8_relative_defects)), 5e-13)
					self.assertLess(float(np.max(run.r4_relative_defects)), 5e-13)
					self.assertLess(
						float(np.max(run.dpsi_jacobian_audit_errors)),
						1e-8,
					)
					self.assertLess(
						float(np.max(run.dr_jacobian_audit_errors)),
						1e-8,
					)
					self.assertLess(
						float(np.max(run.r4_jacobian_audit_errors)),
						1e-8,
					)

				component_figure, _ = plot_generalized_energy_components(
					result.runs[-1],
					method_name=result.method_name,
					momentum_symbol="k",
				)
				error_figure, _ = plot_generalized_energy_errors(
					result.runs,
					method_name=result.method_name,
				)
				convergence_figure, _ = plot_generalized_energy_convergence(
					result.summaries(),
					method_name=result.method_name,
					expected_order=(
						2.0 if method == "abba2_fully_extended_implicit" else 4.0
					),
				)
				symplecticity_figure, _ = plot_fully_extended_symplecticity(
					result.runs,
					method_name=result.method_name,
				)
				for figure in (
					component_figure,
					error_figure,
					convergence_figure,
					symplecticity_figure,
				):
					plt.close(figure)


if __name__ == "__main__":
	unittest.main()
