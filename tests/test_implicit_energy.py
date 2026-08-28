"""Generalized-energy reconstruction contracts for projected implicit GC methods."""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import (
	GCGeneralizedEnergyObserver,
	gc_reduced_time_extended_symplectic_form,
	gc_time_extended_symplectic_form,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	BM4Implicit1,
	GCExtendedFormulation,
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


def _problem() -> tuple[Potential, GCInitialConfiguration, InitialValueProblem]:
	"""Return one inexpensive time-dependent GC problem with exact Hessians."""
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


class GeneralizedEnergyObserverTests(unittest.TestCase):
	"""Audit the stage reconstruction against energy-augmented BM4 maps."""

	def test_bm4_kappa_matches_replayed_energy_augmented_stages(self) -> None:
		_, configuration, problem = _problem()
		events: list[ImplicitBM4IntegrationStep] = []
		simulate(
			problem,
			BM4Implicit1(
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
				step_observer=events.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.1),
				max_step=0.1,
				sample_count=2,
			),
		)
		self.assertEqual(len(events), 1)
		event = events[0]
		observer = GCGeneralizedEnergyObserver(
			problem.dynamics,
			initial_time=0.0,
			initial_state=problem.initial_state,
		)
		observer(event)

		prepared = GCExtendedFormulation(
			coupling_frequency=event.coupling_frequency
		).prepare(problem, track_energy=True)
		augmented = np.concatenate((event.base_stages[0].state_before, [0.0]))
		for stage in event.base_stages:
			mapper = (
				prepared.direct_map
				if stage.flow_name == "flow"
				else prepared.adjoint_map
			)
			augmented = mapper(stage.duration, stage.time, augmented)
			np.testing.assert_allclose(
				augmented[:4],
				stage.state_after,
				rtol=2e-14,
				atol=2e-14,
			)
		self.assertEqual(configuration.layout.particle_count(problem.initial_state), 1)
		self.assertAlmostEqual(observer.records[-1].kappa, augmented[-1] / 2.0)


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
					expected_map_count = 3 if method == "abba4_implicit_1" else 1
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
					expected_order=2.0 if method == "implicit_abba_1" else 4.0,
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
