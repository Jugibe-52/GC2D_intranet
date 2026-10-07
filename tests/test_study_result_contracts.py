"""Persisted study records must remain readable without campaign orchestration."""

from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from contracts.comparison import (
	AdaptiveReference, ComparisonReadView, EnergyAccuracySeries,
	ExecutionLogEntry, FiveMethodComparisonSummary, TrajectoryAccuracySeries,
)
from contracts.study_results import (
	GCEnergyBoundResult, ParallelBM4RecurrenceConfig, ParallelBM4RecurrenceResult,
)
from diagnostics.five_method_comparison_csv import (
	load_five_method_comparison_csv, write_five_method_comparison_csv,
)
from diagnostics.gc_energy_bound import save_energy_bound_result
from diagnostics.parallel_bm4_recurrence_npz import write_parallel_bm4_recurrence_npz
from initial_conditions.gc import GCInitialConfiguration
from potential import Grid, Potential
from studies.five_method_sdirk_rk4_comparison import FiveMethodComparisonConfig, run_five_method_comparison
from studies.four_method_sdirk_comparison import FourMethodSDIRKComparisonConfig, run_four_method_sdirk_comparison
from studies.three_method_newton_comparison import ThreeMethodNewtonComparisonConfig, run_three_method_newton_comparison
from visualization.notebooks import records_table_html


class StudyResultContractTests(unittest.TestCase):
	"""Exercise the common live/archive contract and loader dependency boundary."""

	def test_comparison_families_preserve_shared_trajectories_and_metrics(self) -> None:
		grid = Grid.periodic(8, 8)
		potential = Potential(grid, mean=0.1 * np.cos(grid.x)[:, None] * np.ones(grid.shape))
		initial = GCInitialConfiguration.from_components(x=np.array([0.2]), y=np.array([0.3]))
		config = ThreeMethodNewtonComparisonConfig(t_span=(0.0, 0.02), integration_step=0.01,
			timing_warmups=0, timing_repeats=1)
		three = run_three_method_newton_comparison(potential, initial, config=config)
		four = run_four_method_sdirk_comparison(potential, initial,
			config=FourMethodSDIRKComparisonConfig(**asdict(config)))
		five = run_five_method_comparison(potential, initial,
			config=FiveMethodComparisonConfig(**asdict(config)))
		for name in three.solutions:
			for other in (four, five):
				np.testing.assert_array_equal(three.solutions[name].states, other.solutions[name].states)
				np.testing.assert_array_equal(three.accuracy[name].distances, other.accuracy[name].distances)
				np.testing.assert_array_equal(three.energy_accuracy[name].errors, other.energy_accuracy[name].errors)
		for left, middle, right in zip(three.summaries(), four.summaries(), five.summaries()):
			self.assertEqual(left.time_integrated_rms_energy_error, middle.time_integrated_rms_energy_error)
			self.assertEqual(left.time_integrated_rms_energy_error, right.time_integrated_rms_energy_error)
			self.assertEqual(left.total_residual_evaluations, middle.total_residual_evaluations)

	def test_named_table_rows_do_not_need_attribute_wrappers(self) -> None:
		html = records_table_html([{"method": "RK4 <saved>", "error": 1e-8}],
			columns=(("method", "Method", None), ("error", "Error", ".2e")))
		self.assertIn("RK4 &lt;saved&gt;", html)
		self.assertIn("1.00e-08", html)

	def test_typed_csv_and_independent_energy_and_recurrence_loaders(self) -> None:
		grid = Grid.periodic(8, 8)
		potential = Potential(grid, mean=0.1 * np.cos(grid.x)[:, None] * np.ones(grid.shape))
		initial = GCInitialConfiguration.from_components(x=np.array([0.2]), y=np.array([0.3]))
		config = FiveMethodComparisonConfig(t_span=(0.0, 0.02), integration_step=0.01,
			timing_warmups=0, timing_repeats=1)
		live = run_five_method_comparison(potential, initial, config=config, method_names=("RK4",))
		# Both concrete owners satisfy this structural read API without attribute bags.
		live_view: ComparisonReadView = live
		with TemporaryDirectory() as directory:
			root = Path(directory)
			csv_path = write_five_method_comparison_csv(live, root / "comparison.csv")
			stored = load_five_method_comparison_csv(csv_path)
			stored_view: ComparisonReadView = stored
			self.assertIsInstance(stored.reference, AdaptiveReference)
			self.assertIsInstance(stored.accuracy["RK4"], TrajectoryAccuracySeries)
			self.assertIsInstance(stored.energy_accuracy["RK4"], EnergyAccuracySeries)
			self.assertIsInstance(stored.execution_log[0], ExecutionLogEntry)
			self.assertIsInstance(stored.summaries()[0], FiveMethodComparisonSummary)
			self.assertEqual(asdict(stored_view.summaries()[0]), asdict(live_view.summaries()[0]))
			for values in (stored.reference.states, stored.accuracy["RK4"].distances,
				stored.energy_accuracy["RK4"].errors):
				self.assertFalse(values.flags.writeable)
			with self.assertRaises(ValueError):
				replace(stored.energy_accuracy["RK4"], errors=np.array([np.nan]))

			energy = GCEnergyBoundResult({"reference/times": np.array([0.0, 1.0])},
				{"field": "synthetic"}, [{"method": "RK4"}], [], [], [])
			save_energy_bound_result(root / "energy.h5", energy)
			recurrence_config = ParallelBM4RecurrenceConfig(particle_count=1, t_span=(0.0, 1.0),
				steps_per_cycle=1, saved_samples_per_cycle=1, worker_count=1)
			recurrence = ParallelBM4RecurrenceResult(recurrence_config, np.array([0.0, 1.0]),
				np.array([[0.2, 0.3]]), np.array([[[0.2, 0.4], [0.3, 0.5]]]),
				np.array([0.1]), np.array([2]), np.array([2.0]), np.array([2]), np.array([0.01]), 0.2)
			write_parallel_bm4_recurrence_npz(recurrence, root / "recurrence.npz")
			script = """
import builtins
from pathlib import Path
import sys
original = builtins.__import__
def without_orchestration(name, *args, **kwargs):
    if name.split('.')[0] in {'studies', 'simulation'}:
        raise ImportError('Study orchestration must not be needed to read results: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = without_orchestration
from diagnostics.five_method_comparison_csv import load_five_method_comparison_csv
from diagnostics.gc_energy_bound import load_energy_bound_result
from diagnostics.parallel_bm4_recurrence_npz import load_parallel_bm4_recurrence_npz
from contracts.comparison import AdaptiveReference
from contracts.study_results import GCEnergyBoundResult, ParallelBM4RecurrenceResult
root = Path(sys.argv[1])
comparison = load_five_method_comparison_csv(root / 'comparison.csv')
assert isinstance(comparison.reference, AdaptiveReference)
assert isinstance(load_energy_bound_result(root / 'energy.h5'), GCEnergyBoundResult)
assert isinstance(load_parallel_bm4_recurrence_npz(root / 'recurrence.npz').result, ParallelBM4RecurrenceResult)
assert not any(name == 'studies' or name.startswith('studies.') for name in sys.modules)
"""
			completed = subprocess.run([sys.executable, "-c", script, str(root)],
				text=True, capture_output=True, check=False)
			self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
	unittest.main()
