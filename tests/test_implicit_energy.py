"""Generalized-energy reconstruction contracts for projected implicit GC methods."""

from __future__ import annotations

from dataclasses import fields, replace

import unittest

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import (
	GCGeneralizedEnergyObserver,
	gc_reduced_time_extended_symplectic_form,
	gc_time_extended_symplectic_form,
)
from dynamics import GuidingCenterDynamics

from potential import Potential
from diagnostics.methods.energy import EnergyObserver
from contracts.methods.observation import IntegrationStep
from solvers.nonlinear import NewtonSolver
from simulation import (
	BM4Implicit,
	ImplicitBM4IntegrationStep,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from studies import (
	IMPLICIT_ENERGY_METHODS,
	ImplicitGeneralizedEnergyConfig,
	run_implicit_generalized_energy_study,
)
from visualization import (
	plot_generalized_energy_components,
	plot_generalized_energy_convergence,
	plot_generalized_energy_errors,
	plot_time_extended_symplecticity,
	plot_reduced_time_extended_symplecticity,
)


def _problem() -> tuple[Potential, np.ndarray, InitialValueProblem]:
	"""Return one inexpensive time-dependent GC problem with exact Hessians."""
	potential = Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=5,
	)
	configuration = GuidingCenterDynamics.coordinates.pack_components(np.asarray([1.0]), np.asarray([1.2]))
	dynamics = GuidingCenterDynamics(potential, rho=0.05)
	return potential, configuration, InitialValueProblem(dynamics, configuration, layout=GuidingCenterDynamics.coordinates)


class GeneralizedEnergyObserverTests(unittest.TestCase):
	"""Audit the stage reconstruction against energy-augmented BM4 maps."""

	def test_bm4_kappa_matches_independent_signed_shear_reconstruction(self) -> None:
		_, configuration, problem = _problem()
		events: list[ImplicitBM4IntegrationStep] = []
		solution = simulate(
			problem.with_request(SimulationRequest.uniform(
				t_span=(0.0, 0.1),
				sample_count=2,
			)),
			BM4Implicit(step=0.1,
				solver=NewtonSolver(absolute_tolerance=1e-14,
					relative_tolerance=1e-14, jacobian_method="analytic"),
			),
			observers=(EnergyObserver(), events.append),
		)
		self.assertEqual(len(events), 1)
		event = events[0]
		observer = GCGeneralizedEnergyObserver(
			problem.dynamics,
			initial_time=0.0,
			initial_state=problem.initial_state,
		)
		with self.assertRaisesRegex(ValueError, "twelve stages"):
			observer(replace(event, base_stages=event.base_stages[:-1]))
		plain = IntegrationStep(**{item.name: getattr(event, item.name) for item in fields(IntegrationStep)})
		with self.assertRaisesRegex(TypeError, "supports ABBA2Implicit"):
			observer(plain)
		self.assertEqual(len(observer.records), 1)
		observer(event)

		# Independently reconstruct the two signed shears in each accepted map.
		kappa = 0.
		self.assertEqual(event.coupling_frequency, 0.)
		for stage in event.base_stages:
			first, second = np.split(stage.state_before, 2)
			duration, time = stage.duration, stage.time
			if stage.flow_name == "flow":
				second_after = second + duration * problem.dynamics.vector_field(time, first)
				first_after = first + duration * problem.dynamics.vector_field(time, second_after)
				sources = (first, second_after)
			else:
				first_after = first + duration * problem.dynamics.vector_field(time, second)
				second_after = second + duration * problem.dynamics.vector_field(time, first_after)
				sources = (second, first_after)
			kappa += duration * sum(float(problem.dynamics.extended_momentum_derivative(
				time, state)[0]) for state in sources) / 2
			np.testing.assert_allclose(np.concatenate((first_after, second_after)),
				stage.state_after, rtol=2e-14, atol=2e-14)
		self.assertAlmostEqual(observer.records[-1].kappa, kappa)
		self.assertAlmostEqual(float(solution.diagnostics["extended_momentum"][0, -1]), kappa)



class ImplicitGeneralizedEnergyStudyTests(unittest.TestCase):
	"""Verify all three method studies, summaries, orders, and plots."""

	def test_time_extended_form_is_non_degenerate_and_skew(self) -> None:
		for form, shape in (
			(gc_time_extended_symplectic_form(), (6, 6)),
			(gc_reduced_time_extended_symplectic_form(), (4, 4)),
		):
			self.assertEqual(form.shape, shape)
			np.testing.assert_array_equal(form.T, -form)
			self.assertNotEqual(np.linalg.det(form), 0.0)

	def test_three_implicit_methods_return_complete_energy_histories(self) -> None:
		potential, configuration, _ = _problem()
		config = ImplicitGeneralizedEnergyConfig(
			steps=(0.1, 0.05),
			t_span=(0.0, 0.2),
			output_sample_count=3,
			rho=0.05,
			newton_absolute_tolerance=1e-14,
			newton_relative_tolerance=1e-14,
		)
		for method in IMPLICIT_ENERGY_METHODS:
			with self.subTest(method=method):
				result = run_implicit_generalized_energy_study(
					potential,
					configuration,
					method=method,
					config=config,
				)
				self.assertEqual(result.steps, config.steps)
				self.assertEqual(len(result.summaries()), 2)
				self.assertEqual(len(result.convergence_orders()), 1)
				for run, step_count in zip(result.runs, (2, 4), strict=True):
					self.assertEqual(run.solution.n_steps, step_count)
					self.assertEqual(len(run.records), step_count + 1)
					self.assertEqual(
						len(run.extended_symplecticity_records),
						step_count,
					)
					self.assertEqual(
						len(run.reduced_extended_symplecticity_records),
						step_count,
					)
					np.testing.assert_allclose(
						run.generalized_energy,
						run.hamiltonian + run.kappa,
					)
					self.assertEqual(run.kappa[0], 0.0)
					self.assertEqual(run.relative_errors[0], 0.0)
					self.assertTrue(np.all(np.isfinite(run.relative_errors)))
					self.assertTrue(
						np.all(np.isfinite(run.extended_relative_defects))
					)
					self.assertLess(
						float(np.max(run.extended_relative_defects)),
						1e-8,
					)
					self.assertLess(
						float(np.max(run.extended_determinant_errors)),
						1e-8,
					)
					self.assertTrue(
						np.all(
							np.isfinite(run.reduced_extended_relative_defects)
						)
					)
					self.assertLess(
						float(np.max(run.reduced_extended_determinant_errors)),
						1e-8,
					)
					expected_map_count = 3 if method == "abba4_implicit" else 1
					self.assertTrue(
						all(
							record.base_map_count == expected_map_count
							for record in run.extended_symplecticity_records
						)
					)

				component_figure, _ = plot_generalized_energy_components(
					result.runs[-1],
					method_name=result.method_name,
				)
				error_figure, _ = plot_generalized_energy_errors(
					result.runs,
					method_name=result.method_name,
				)
				convergence_figure, _ = plot_generalized_energy_convergence(
					result.summaries(),
					method_name=result.method_name,
					expected_order=(
						2.0
						if method == "abba2_implicit_reduced_multiplier"
						else 4.0
					),
				)
				extended_figure, _ = plot_time_extended_symplecticity(
					result.runs,
					method_name=result.method_name,
				)
				reduced_extended_figure, _ = (
					plot_reduced_time_extended_symplecticity(
						result.runs,
						method_name=result.method_name,
					)
				)
				plt.close(component_figure)
				plt.close(error_figure)
				plt.close(convergence_figure)
				plt.close(extended_figure)
				plt.close(reduced_extended_figure)


if __name__ == "__main__":
	unittest.main()
