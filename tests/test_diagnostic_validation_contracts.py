"""Regression contracts for diagnostic admission and immutable stored histories."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from contracts.observation import IntegrationStep
from diagnostics.abba_reversibility import ImplicitABBAReversibilityObserver
from diagnostics.energy.observer import GCGeneralizedEnergyObserver
from diagnostics.five_method_comparison_csv import load_five_method_comparison_csv
from diagnostics.reference_trajectory import ReferenceTrajectoryPaths, StoredReferenceTrajectory
from diagnostics.symplecticity.observer import SymplecticityObserver, gc_physical_symplectic_form
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import ABBA2Implicit, InitialValueProblem, SimulationRequest, simulate


class DiagnosticValidationContracts(unittest.TestCase):
	"""Keep ordering, sampling and ownership stable across helper extraction."""

	def test_reference_history_owns_copies_and_retains_error_precedence(self) -> None:
		path = Path("reference")
		paths = ReferenceTrajectoryPaths(path, path / "trajectory.npz", path / "metadata.json", path / "README.md")
		times = np.array([0.0, 1.0])
		states = np.array([[0.2, 0.3], [0.4, 0.5]])
		initial = states[:, 0].copy()
		audit = states.copy()
		distances = np.zeros((1, 2))
		metadata = {"schema_version": 2}
		reference = StoredReferenceTrajectory(times, states, initial, audit, distances, metadata, paths)
		for source, stored in zip(
			(times, states, initial, audit, distances),
			(reference.times, reference.states, reference.initial_state, reference.audit_states, reference.audit_distances),
			strict=True,
		):
			self.assertFalse(np.shares_memory(source, stored))
			self.assertFalse(stored.flags.writeable)
		metadata["schema_version"] = 99
		self.assertEqual(reference.metadata["schema_version"], 2)
		with self.assertRaisesRegex(ValueError, "first reference sample"):
			StoredReferenceTrajectory(times, states, initial + 1, np.zeros((0,)), distances, {}, paths)

	def test_csv_header_checks_precede_payload_and_history_reconstruction(self) -> None:
		with TemporaryDirectory() as temporary:
			path = Path(temporary) / "bad.csv"
			path.write_text("schema_version,schema_version\n1,not-json\n", encoding="utf-8")
			with self.assertRaisesRegex(ValueError, "malformed or duplicate columns"):
				load_five_method_comparison_csv(path)
			path.write_text('schema_version,metadata_json,sample_index,time\n99,"{}",0,0\n', encoding="utf-8")
			with self.assertRaisesRegex(ValueError, "Unsupported comparison CSV schema version"):
				load_five_method_comparison_csv(path)

	def test_observer_controls_reject_booleans_before_preparing_output(self) -> None:
		for value in (True, np.bool_(True)):
			with self.subTest(value=value):
				with self.assertRaisesRegex(ValueError, "particle_count"):
					gc_physical_symplectic_form(value)
		with TemporaryDirectory() as temporary:
			root = Path(temporary)
			with self.assertRaisesRegex(ValueError, "relative_step"):
				SymplecticityObserver(
					notebook_path=root / "notebooks" / "invalid.ipynb",
					project_root=root, particle_count=np.int64(1), relative_step=float("nan"),
				)
			self.assertEqual(tuple(root.iterdir()), ())

	def test_sampling_skips_snapshot_checks_and_energy_rejection_is_atomic(self) -> None:
		potential = Potential.random(A=0.03, M=2, nx=8, ny=8, seed=7, interpolation_order=5)
		dynamics = GuidingCenterDynamics(potential, rho=0.02)
		configuration = GCInitialConfiguration.from_components(x=np.array([1.0]), y=np.array([1.2]))
		events: list[IntegrationStep] = []
		simulate(
			InitialValueProblem(dynamics, configuration),
			ABBA2Implicit(step_observer=events.append),
			SimulationRequest.uniform(t_span=(0.0, 0.02), max_step=0.01, sample_count=3),
		)
		reversibility = ImplicitABBAReversibilityObserver(sample_every=2)
		reversibility(events[0])
		# Unsampled snapshots still advance sequence tracking, but their numerical
		# payload is intentionally not admitted to the expensive diagnostic path.
		reversibility(replace(events[1], dynamics=None, state_before=np.array([np.nan])))
		self.assertEqual(len(reversibility.samples), 1)
		with self.assertRaisesRegex(ValueError, "consecutively"):
			reversibility(events[1])

		energy = GCGeneralizedEnergyObserver(dynamics, initial_time=0.0, initial_state=events[0].state_before)
		with self.assertRaisesRegex(ValueError, "continuous time grid"):
			energy(replace(events[0], start_time=1.0))
		self.assertEqual(len(energy.records), 1)
		energy(events[0])
		self.assertEqual(len(energy.records), 2)


if __name__ == "__main__":
	unittest.main()
