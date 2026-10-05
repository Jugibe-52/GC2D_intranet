"""Offline checks for parallel submissions, failure isolation and batch recovery."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from threading import Barrier
from time import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from contracts.result import IntegrationData
from execution._modal_worker import execute_cycle_payload
from studies.poincare_rho_batch import RhoBatchJob, run_modal_rho_batch, _update_summary
from studies.poincare_rho_sweep import RhoStarConfig
from studies.poincare_gap_probes import GapProbeSeeds


class RhoBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "field.h5"
        self.source.write_bytes(b"test physical source")

    def jobs(self, count=33):
        return [RhoBatchJob(f"job_{i}", "developements/batch", f"rho_{i}",
                           RhoStarConfig(rho_hat=i / 100, method="RK4"))
                for i in range(count)]

    def run_batch(self, jobs, **kwargs):
        return run_modal_rho_batch(self.source, jobs,
                                   record_directory=self.root / "receipts", **kwargs)

    def test_33_calls_overlap_with_unique_receipts_and_one_failure_isolated(self):
        """A barrier requires every call to enter before any can complete."""
        barrier = Barrier(33, timeout=20)
        executor_paths = []

        def make_executor(directory, **kwargs):
            executor_paths.append(directory)
            return SimpleNamespace(last_call_id=f"fc-{directory.name}",
                                   last_record_path=directory / "receipt.json")

        def execute(prepared, destination, *, executor, options, save_folded_returns):
            self.assertEqual((options.backend, options.device), ("jax", "cpu"))
            self.assertTrue(save_folded_returns)
            self.assertTrue(destination.startswith("gc2d_data:gc2d-notebooks-data/developements/batch/"))
            start = time()
            barrier.wait()
            if prepared.config.rho_hat == 0:
                raise RuntimeError("secret-test-token must never appear in progress")
            return SimpleNamespace(
                solution=SimpleNamespace(diagnostics={
                    "execution_backend": "jax", "step_count": 250000,
                    "worker_started_unix_seconds": start,
                    "worker_finished_unix_seconds": time(),
                }), metadata={"modal_call_id": executor.last_call_id,
                              "modal_receipt": str(executor.last_record_path)},
            )

        with patch("studies.poincare_rho_batch.load_dimensional_h5_field") as load, \
             patch("studies.poincare_rho_batch.build_rho_star",
                   side_effect=lambda field, config, **kw: SimpleNamespace(config=config)), \
             patch("studies.poincare_rho_batch.modal_rho_executor", side_effect=make_executor), \
             patch("studies.poincare_rho_batch.run_and_save_rho_star", side_effect=execute):
            report = self.run_batch(self.jobs())
        load.assert_called_once()
        self.assertEqual(len(set(executor_paths)), 33)
        self.assertEqual((report["status"], report["completed_count"], report["failed_count"]),
                         ("failed", 32, 1))
        # The failed call has no returned worker interval, so the measured lower
        # bound uses all 32 successful workers, rather than claiming 33.
        self.assertEqual(report["measured_peak_worker_overlap"], 32)
        saved = (self.root / "receipts" / "progress.json").read_text()
        self.assertNotIn("secret-test-token", saved)
        self.assertEqual(json.loads(saved), report)
        self.assertEqual(report["jobs"][0]["modal_call_id"], "fc-job_0")
        self.assertFalse(list((self.root / "receipts").glob("*.tmp")))

    def test_invalid_batch_or_fingerprint_never_submits(self):
        jobs = self.jobs(2)
        cases = (([jobs[0], jobs[0]], {}),
                 ([jobs[0], replace(jobs[1], run_id=jobs[0].run_id)], {}),
                 (jobs, {"expected_source_sha256": "wrong"}),
                 (jobs, {"max_workers": 45}))
        with patch("studies.poincare_rho_batch.modal_rho_executor") as executor:
            for inputs, kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.run_batch(inputs, **kwargs)
        executor.assert_not_called()

    def test_44_probe_jobs_use_explicit_geometry_concurrently(self):
        """All four methods and eleven rho runs may enter their workers together."""
        seeds = GapProbeSeeds(((.43, .58), (.44, .57)), (41, 42), ("NW", "NW"))
        jobs = [replace(job, config=replace(job.config, particles=2, arms=2), probe_seeds=seeds)
                for job in self.jobs(44)]
        barrier = Barrier(44, timeout=20)

        def execute(prepared, destination, **kwargs):
            barrier.wait()
            return SimpleNamespace(solution=SimpleNamespace(diagnostics={"execution_backend": "jax"}),
                                   metadata={})

        with patch("studies.poincare_rho_batch.load_dimensional_h5_field"), \
             patch("studies.poincare_rho_batch.build_rho_star") as star, \
             patch("studies.poincare_rho_batch.build_gap_probes") as probes, \
             patch("studies.poincare_rho_batch.modal_rho_executor"), \
             patch("studies.poincare_rho_batch.run_and_save_rho_star", side_effect=execute):
            report = self.run_batch(jobs, max_workers=44)
        star.assert_not_called()
        self.assertEqual(probes.call_count, 44)
        self.assertTrue(all(call.kwargs["seeds"] == seeds for call in probes.call_args_list))
        self.assertEqual((report["completed_count"], report["failed_count"]), (44, 0))

    def test_confirmed_receipt_is_resumed_and_scipy_result_is_rejected(self):
        job = self.jobs(1)[0]
        directory = self.root / "receipts" / job.job_id
        directory.mkdir(parents=True)
        (directory / "receipt.json").write_text(json.dumps({
            "version": 1, "app_name": "gc2d-poincare-methods", "function_name": "integrate",
            "environment_name": None, "input_sha256": "digest", "call_id": "fc-original",
            "submission": "confirmed",
        }))

        def retrieve(prepared, destination, *, executor, **kwargs):
            self.assertEqual(executor.resume_call_id, "fc-original")
            return SimpleNamespace(solution=SimpleNamespace(diagnostics={"execution_backend": "scipy"}),
                                   metadata={})

        with patch("studies.poincare_rho_batch.load_dimensional_h5_field"), \
             patch("studies.poincare_rho_batch.build_rho_star"), \
             patch("studies.poincare_rho_batch.run_and_save_rho_star", side_effect=retrieve):
            report = self.run_batch([job], expected_source_sha256=sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["jobs"][0]["error_type"], "ValueError")

    def test_overlap_uses_this_batch_and_does_not_count_touching_intervals(self):
        now = time()
        report = {"started_unix_seconds": now - 5, "jobs": [
            dict(status="completed", worker_started_unix_seconds=now - 20,
                 worker_finished_unix_seconds=now - 10),
            dict(status="completed", worker_started_unix_seconds=now - 6,
                 worker_finished_unix_seconds=now - 3),
            dict(status="completed", worker_started_unix_seconds=now - 3,
                 worker_finished_unix_seconds=now - 1),
        ]}
        _update_summary(report)
        self.assertEqual(report["worker_intervals_available"], 3)
        self.assertEqual(report["measured_peak_worker_overlap"], 1)

    def test_worker_keeps_observations_and_times_but_removes_per_step_arrays(self):
        data = IntegrationData(np.array([0., 1.]), np.zeros((2, 2)),
                               {"execution_backend": "jax", "corrections": np.ones(50)})
        with patch("execution._modal_worker.execute_payload", return_value=(1, "digest", data)):
            version, digest, reduced = execute_cycle_payload(b"opaque")
        self.assertEqual((version, digest), (1, "digest"))
        np.testing.assert_array_equal(reduced.states, data.states)
        self.assertNotIn("corrections", reduced.diagnostics)
        self.assertEqual(reduced.diagnostics["execution_backend"], "jax")
        self.assertGreaterEqual(reduced.diagnostics["worker_finished_unix_seconds"],
                                reduced.diagnostics["worker_started_unix_seconds"])


if __name__ == "__main__":
    unittest.main()
