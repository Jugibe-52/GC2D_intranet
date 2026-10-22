"""Observable constructor policies retained when input preparation is extracted."""

from dataclasses import replace
import unittest

import numpy as np

from contracts.studies.comparison import EnergyAccuracySeries, TrajectoryAccuracySeries
from dynamics.fc import FullCyclotronDynamics
from dynamics.gc import GuidingCenterDynamics
from formulations.layout import GCStateLayout
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.sdirk import SDIRK4
from potential.potential import Potential
from solvers.nonlinear import BroydenSolver, NewtonSolver
from states.history import States
from studies.poincare_automation import AutomationConfig
from studies.three_method_newton_comparison import ThreeMethodNewtonComparisonConfig


class PreparedConstructorContractTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.potential = Potential.random(A=.03, M=2, nx=8, ny=8, seed=3)

	def test_physical_parameters_keep_signed_values_and_conversion_order(self) -> None:
		gc = GuidingCenterDynamics(self.potential, rho="0", B="-2")
		fc = FullCyclotronDynamics(self.potential, rho=".5", eta="-1")
		self.assertEqual((gc.rho, gc.B, fc.rho, fc.eta), (0., -2., .5, -1.))
		np.testing.assert_array_equal(gc.effective_potential.mean, self.potential.mean / -2.)
		with self.assertRaisesRegex(TypeError, "Potential instance"):
			GuidingCenterDynamics(None, rho=-1., B=0.)
		with self.assertRaisesRegex(ValueError, "`B`"):
			GuidingCenterDynamics(self.potential, rho=-1., B=0.)
		with self.assertRaisesRegex(ValueError, "finite, non-zero parameters"):
			FullCyclotronDynamics(self.potential, rho=-1., eta=0.)
		with self.assertRaisesRegex(ValueError, "`rho` must be positive"):
			FullCyclotronDynamics(self.potential, rho=-1., eta=1.)

	def test_supplied_solver_overrides_legacy_controls_before_validation(self) -> None:
		solver = NewtonSolver(absolute_tolerance=1e-9, relative_tolerance=1e-8, max_iterations=7)
		for constructor in (GaussLegendre4, SDIRK4):
			with self.subTest(method=constructor.__name__):
				method = constructor(solver=solver, newton_absolute_tolerance=-1.,
					newton_relative_tolerance=np.nan, newton_max_iterations=0)
				self.assertEqual((method.newton_absolute_tolerance, method.newton_relative_tolerance,
					method.newton_max_iterations), (1e-9, 1e-8, 7))
				with self.assertRaisesRegex(ValueError, "NewtonSolver only"):
					constructor(solver=BroydenSolver(), newton_absolute_tolerance=-1.)
				with self.assertRaisesRegex(ValueError, "newton_jacobian_relative_step"):
					constructor(solver=solver, newton_jacobian_relative_step=0., newton_jacobian_method="invalid")

	def test_accuracy_records_own_arrays_and_preserve_signed_energy(self) -> None:
		distances = np.ones((2, 3))
		reduction = np.ones(3)
		trajectory = TrajectoryAccuracySeries("sample", distances, reduction, reduction, reduction)
		errors = np.array([[-1., 0., 2.]])
		energy = EnergyAccuracySeries("sample", errors)
		distances[:] = 9.
		reduction[:] = 9.
		errors[:] = 9.
		for values in (trajectory.distances, trajectory.rms_distance, trajectory.mean_distance,
			trajectory.maximum_distance):
			np.testing.assert_array_equal(values, np.ones(values.shape))
			self.assertFalse(values.flags.writeable)
		np.testing.assert_array_equal(energy.errors, [[-1., 0., 2.]])
		self.assertFalse(energy.errors.flags.writeable)
		with self.assertRaisesRegex(ValueError, "finite and non-negative"):
			replace(trajectory, rms_distance=np.array([-1., 0.]))
		with self.assertRaisesRegex(ValueError, "method_name"):
			EnergyAccuracySeries("", np.array([np.nan]))

	def test_failed_diagnostic_preparation_does_not_seal_the_recorder(self) -> None:
		result = States.prepare(np.array([0., 1.]), np.zeros(2), layout=GCStateLayout())
		result.record_history(np.ones((2, 2)))
		metric = np.array([3., 4.])
		with self.assertRaisesRegex(TypeError, "scalar values or NumPy arrays"):
			result.finalize({"metric": metric, "invalid": []})
		self.assertFalse(result.complete)
		self.assertEqual(dict(result.diagnostics), {})
		self.assertTrue(metric.flags.writeable)
		self.assertIs(result.finalize({"metric": metric}), result)
		metric[:] = -1.
		np.testing.assert_array_equal(result.diagnostics["metric"], [3., 4.])

	def test_poincare_sequences_are_owned_but_numeric_policies_remain_strict(self) -> None:
		rhos, fields, methods = [0., .3], [-1.5, 1.5], ["RK4"]
		config = AutomationConfig(rhos=rhos, magnetic_fields=fields, methods=methods, sigma=0.)
		rhos[0], fields[0], methods[0] = 4., 0., "invalid"
		self.assertEqual(config.rhos, (0., .3))
		self.assertEqual(config.magnetic_fields, (-1.5, 1.5))
		self.assertEqual(config.methods, ("RK4",))
		self.assertEqual(len(config.cases()), 4)
		for field, value, message in (
			("rhos", (".3",), "finite number"), ("rhos", (True,), "finite number"),
			("rhos", (.3, .3), "unique values"), ("magnetic_fields", (0.,), "nonzero"),
			("cycles", np.int64(2), "positive integer"), ("cycles", True, "positive integer"),
			("reference_length", "1", "finite number"), ("sigma", -1., "nonnegative"),
		):
			with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
				replace(config, **{field: value})
		with self.assertRaisesRegex(ValueError, "unique values"):
			replace(config, rhos=(), methods=("invalid",), source="")
		with self.assertRaisesRegex(ValueError, "source must name"):
			replace(config, source="", backend="invalid")

	def test_comparison_audit_errors_precede_sampling_alignment(self) -> None:
		with self.assertRaisesRegex(ValueError, "audit tolerance cannot be looser"):
			ThreeMethodNewtonComparisonConfig(t_span=(0., 1.), integration_step=.3,
				audit_relative_tolerance=1e-3)
		with self.assertRaisesRegex(ValueError, "duration / integration_step"):
			ThreeMethodNewtonComparisonConfig(t_span=(0., 1.), integration_step=.3, save_interval=.4)
		config = ThreeMethodNewtonComparisonConfig(t_span=(0., 1.), integration_step=.1)
		self.assertEqual((config.save_interval, config.step_count, config.output_sample_count), (.1, 10, 11))


if __name__ == "__main__":
	unittest.main()
