"""Round-trip, integrity and transport failure contracts for saved simulations."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics.persistence import load_solution, save_solution
from diagnostics.paths import solution_destination
from diagnostics.storage import StorageError
from initial_conditions.fc import FCInitialConfiguration
from initial_conditions.gc import GCInitialConfiguration
from potential.load import GC2DH5Metadata
from potential.potential import Potential
from solution import Solution
from studies.persistence_demo import PersistenceDemoConfig, run_persistence_demo


class SolutionPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.field = Potential.random(A=0.1, M=2, nx=12, ny=12, seed=17)

    def solution(self, full: bool = False) -> Solution:
        configuration = FCInitialConfiguration if full else GCInitialConfiguration
        state = np.arange(8 if full else 4, dtype=float) / 10
        return Solution(t=np.array([0.0, 0.1, 0.2]),
                        states=np.column_stack([state, state + 0.01, state + 0.02]),
                        source=configuration(state),
                        diagnostics={"step_count": 2, "label": "example", "enabled": True,
                                     "error": 1e-5, "work": np.array([1, 2], dtype=np.int64),
                                     "complex": np.array([1 + 2j]), "missing": np.array([np.nan])})

    def test_destination_defaults_to_bucket_and_local_is_explicit(self) -> None:
        self.assertEqual(solution_destination("developements/demo", "run_01"),
                         "gc2d_data:gc2d-notebooks-data/developements/demo/run_01")
        self.assertEqual(solution_destination("developements/demo", "run_01", storage="local",
                                              project_root=self.directory),
                         str(self.directory / "outputs/developements/demo/run_01"))
        self.assertEqual(solution_destination("developements/demo", "run_01", bucket_root="test:bucket"),
                         "test:bucket/developements/demo/run_01")
        for path in ("../demo", "/developements/demo", "notebooks/developements/demo", "demo"):
            with self.assertRaises(ValueError):
                solution_destination(path, "run_01")
        with self.assertRaises(ValueError):
            solution_destination("developements/demo", "../outside")

    def test_default_save_requires_unambiguous_experiment_and_run(self) -> None:
        for arguments in ({}, {"experiment_path": "developements/demo"},
                          {"destination": self.directory, "run_id": "run_01"}):
            with self.assertRaises(ValueError):
                save_solution(self.solution(), metadata={}, **arguments)

    def test_gc_and_fc_roundtrip_preserves_layout_diagnostics_and_field(self) -> None:
        for full in (False, True):
            with self.subTest(full=full):
                solution = self.solution(full)
                location = self.directory / str(full)
                save_solution(solution, location, metadata={"rho": 0.3}, potential=self.field)
                stored = load_solution(location)
                np.testing.assert_array_equal(stored.solution.states, solution.states)
                np.testing.assert_array_equal(stored.solution.t, solution.t)
                self.assertIs(type(stored.solution.source.layout), type(solution.source.layout))
                self.assertFalse(stored.solution.states.flags.writeable)
                for name, value in solution.diagnostics.items():
                    np.testing.assert_array_equal(stored.solution.diagnostics[name], value)
                for original, restored in zip(solution.positions(), stored.solution.positions()):
                    np.testing.assert_array_equal(original, restored)
                self.assertEqual(stored.metadata, {"rho": 0.3})
                assert stored.potential is not None
                for dx, dy in ((0, 0), (1, 0), (0, 1), (2, 0)):
                    np.testing.assert_array_equal(
                        stored.potential.evaluate(0.37, np.array([0.13, 2.27]), np.array([0.49, 0.49]), dx=dx, dy=dy),
                        self.field.evaluate(0.37, np.array([0.13, 2.27]), np.array([0.49, 0.49]), dx=dx, dy=dy))

    def test_hdf5_provenance_is_preserved_without_source_file(self) -> None:
        provenance = GC2DH5Metadata(
            source_field_indices=np.array([0, 1]), source_x=np.arange(12, dtype=float),
            source_y=np.arange(12, dtype=float), source_frequencies=np.array([1.0, 2.0]),
            characteristic_length=0.06, characteristic_period=0.5,
            normalization_factor=2.0, attributes={"B": 1.5, "label": np.bytes_("measured"),
                                                "phase": np.array([1 + 2j])},
            source_path=Path("missing.h5"))
        field = Potential(self.field.grid, self.field.mean, self.field.modes,
                          self.field.frequencies, metadata=provenance)
        location = self.directory / "hdf5"
        save_solution(self.solution(), location, metadata={}, potential=field)
        loaded = load_solution(location).potential
        assert loaded is not None
        self.assertIsInstance(loaded.metadata, GC2DH5Metadata)
        self.assertEqual(loaded.metadata.source_path, Path("missing.h5"))
        for name in provenance.attributes:
            np.testing.assert_array_equal(loaded.metadata.attributes[name], provenance.attributes[name])
        np.testing.assert_array_equal(loaded.modes, field.modes)

    def test_no_potential_and_no_initial_configuration_state(self) -> None:
        original = self.solution()
        solution = Solution(t=original.t, states=original.states, source=GCInitialConfiguration())
        location = self.directory / "plain"
        save_solution(solution, location, metadata={})
        restored = load_solution(location)
        self.assertIsNone(restored.potential)
        self.assertIsNone(restored.solution.source.initial_state)

    def test_collision_corruption_and_incomplete_run_are_rejected(self) -> None:
        location = self.directory / "run"
        save_solution(self.solution(), location, metadata={})
        with self.assertRaises(FileExistsError):
            save_solution(self.solution(), location, metadata={})
        path = location / "solution.npz"
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_solution(location)
        (location / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_solution(location)

    def test_manifest_cannot_select_arbitrary_paths_or_unknown_versions(self) -> None:
        location = self.directory / "run"
        location.mkdir()
        for manifest in (
            {"schema_version": 2, "artifact_kind": "gc2d_solution", "files": {}},
            {"schema_version": 1, "artifact_kind": "gc2d_solution",
             "files": {"../secret": {}, "solution.npz": {}, "metadata.json": {}}},
        ):
            (location / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_solution(location)

    def test_actual_demo_energy_matches_reconstructed_effective_potential(self) -> None:
        config = PersistenceDemoConfig(0.1, 2, 12, 12, 17, 3, 0.3,
                                       (0.1, 0.2, 0.3), 0.0, (0.0, 0.1), 0.01, 11)
        solution, location = run_persistence_demo(config, self.directory / "demo")
        stored = load_solution(location)
        np.testing.assert_array_equal(stored.solution.states, solution.states)
        assert stored.potential is not None
        field = stored.potential.gyroaverage(stored.metadata["dynamics"]["rho"])
        x, y = stored.solution.positions()
        np.testing.assert_allclose(field.evaluate(solution.t, x, y),
                                   solution.diagnostics["physical_hamiltonian"], atol=1e-15)
        with self.assertRaises(ValueError):
            run_persistence_demo(replace(config, radial_fractions=(-0.1,)), self.directory / "bad")

    def test_remote_roundtrip_and_manifest_last(self) -> None:
        objects: dict[str, bytes] = {}
        uploaded: list[str] = []

        def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            operation, source = args[1:3]
            if operation == "lsjson":
                listing = [{"Name": name.rsplit("/", 1)[-1]} for name in objects
                           if name.startswith(source + "/")]
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(listing))
            destination = args[3]
            if source.startswith(("test:", "gc2d_data:")):
                Path(destination).write_bytes(objects[source])
            else:
                self.assertIn("--immutable", args)
                objects[destination] = Path(source).read_bytes()
                uploaded.append(destination.rsplit("/", 1)[-1])
            return subprocess.CompletedProcess(args, 0)

        with patch("diagnostics.storage.shutil.which", return_value="rclone"), \
             patch("diagnostics.storage.subprocess.run", side_effect=command):
            location = "test:bucket/experiment/run"
            save_solution(self.solution(), location, metadata={}, potential=self.field)
            self.assertEqual(uploaded[-1], "manifest.json")
            restored = load_solution(location)
            np.testing.assert_array_equal(restored.solution.states, self.solution().states)
            with self.assertRaises(FileExistsError):
                save_solution(self.solution(), location, metadata={})
            default_location = save_solution(self.solution(), metadata={},
                                             experiment_path="developements/demo", run_id="run_01")
            self.assertEqual(default_location,
                             "gc2d_data:gc2d-notebooks-data/developements/demo/run_01")
            np.testing.assert_array_equal(load_solution(default_location).solution.states,
                                           self.solution().states)

    def test_authentication_failure_keeps_local_archive_and_does_not_publish(self) -> None:
        with patch("diagnostics.storage.shutil.which", return_value="rclone"), \
             patch("diagnostics.storage.subprocess.run", return_value=subprocess.CompletedProcess([], 7)) as run:
            with self.assertRaises(StorageError) as caught:
                save_solution(self.solution(), metadata={},
                              experiment_path="developements/demo", run_id="run_01")
            self.assertEqual(run.call_count, 1)
            staging = Path(str(caught.exception).split("retained at ")[1].rstrip("."))
            self.addCleanup(shutil.rmtree, staging)
            self.assertTrue((staging / "manifest.json").is_file())
            np.testing.assert_array_equal(load_solution(staging).solution.states, self.solution().states)

    def test_interrupted_upload_leaves_no_remote_manifest(self) -> None:
        responses = [subprocess.CompletedProcess([], code) for code in (4, 0, 7)]
        with patch("diagnostics.storage.shutil.which", return_value="rclone"), \
             patch("diagnostics.storage.subprocess.run", side_effect=responses) as run:
            with self.assertRaises(StorageError) as caught:
                save_solution(self.solution(), "test:bucket/run", metadata={})
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list[1:]:
                self.assertFalse(call.args[0][3].endswith("/manifest.json"))
            staging = Path(str(caught.exception).split("retained at ")[1].rstrip("."))
            self.addCleanup(shutil.rmtree, staging)
            np.testing.assert_array_equal(load_solution(staging).solution.states, self.solution().states)


if __name__ == "__main__":
    unittest.main()
