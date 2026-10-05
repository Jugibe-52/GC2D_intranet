"""Concurrent, recoverable Modal execution of independently saved rho studies."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import file_digest
import json
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import time
from typing import Any, Literal

from contracts.execution_options import ExecutionOptions
from diagnostics.paths import DEFAULT_RESULTS_BUCKET, solution_destination
from diagnostics.storage import StorageError
from studies.dimensional_h5_midpoint import load_dimensional_h5_field, resolve_h5_source
from studies.poincare_gap_probes import GapProbeSeeds, build_gap_probes
from studies.poincare_rho_sweep import (
    RhoStarConfig, build_rho_star, modal_rho_executor, run_and_save_rho_star,
)


@dataclass(frozen=True)
class RhoBatchJob:
    """One method/rho calculation with an independent archive and receipt path."""

    job_id: str
    experiment_path: str
    run_id: str
    config: RhoStarConfig
    probe_seeds: GapProbeSeeds | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.job_id):
            raise ValueError("job_id must contain only letters, numbers, '_' and '-'.")
        if not isinstance(self.config, RhoStarConfig):
            raise TypeError("config must be a RhoStarConfig.")
        if self.probe_seeds is not None:
            if not isinstance(self.probe_seeds, GapProbeSeeds):
                raise TypeError("probe_seeds must be GapProbeSeeds or None.")
            if self.config.particles != len(self.probe_seeds.fractions):
                raise ValueError("config.particles must equal the explicit probe count.")
        solution_destination(self.experiment_path, self.run_id)


def _field_key(config: RhoStarConfig) -> tuple:
    """Group jobs that use identical dimensional HDF5 preprocessing."""
    return (config.magnetic_field, config.characteristic_length,
            tuple(config.source_selection), config.interpolation_order)


def _write_progress(path: Path, report: dict[str, Any]) -> None:
    """Replace complete metadata snapshots atomically; never persist credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _update_summary(report: dict[str, Any]) -> None:
    """Measure actual worker overlap, excluding historical completed archives.

    Worker intervals use Unix seconds. Intervals are clipped to this invocation;
    a resumed integration already running at launch still contributes. Finishes
    precede starts at the same instant, so touching intervals do not overlap.
    """
    now = time()
    events = []
    interval_count = 0
    for row in report["jobs"]:
        start = row.get("worker_started_unix_seconds")
        end = row.get("worker_finished_unix_seconds")
        if start is None or end is None:
            continue
        if not (math.isfinite(start) and math.isfinite(end) and end >= start):
            continue
        interval_count += 1
        left, right = max(start, report["started_unix_seconds"]), min(end, now)
        if right > left:
            events.extend(((left, 1), (right, -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    report.update(
        updated_unix_seconds=now,
        completed_count=sum(row["status"] == "completed" for row in report["jobs"]),
        failed_count=sum(row["status"] == "failed" for row in report["jobs"]),
        worker_intervals_available=interval_count,
        measured_peak_worker_overlap=peak,
    )


def run_modal_rho_batch(
    source: str | Path, jobs: Sequence[RhoBatchJob], *,
    record_directory: str | Path, progress_path: str | Path | None = None,
    max_workers: int = 33, app_name: str = "gc2d-poincare-methods",
    storage: Literal["bucket", "local"] = "bucket",
    bucket_root: str = DEFAULT_RESULTS_BUCKET, project_root: str | Path | None = None,
    save_folded_returns: bool = True, expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Submit independent JAX CPU calls concurrently and save every valid result.

    Each job has its own executor and receipt directory. Confirmed receipts are
    recovered without resubmission; completed matching archives are reused.
    A failure leaves other jobs running and appears in the returned report.
    Upload failures remain failures, with any retained recovery archive recorded.
    The caller must inspect ``failed_count`` before treating a batch as complete.
    No deployment, retries, local numerical fallback or bucket secrets are used.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= 44:
        raise ValueError("max_workers must be an integer between 1 and 44.")
    jobs = tuple(jobs)
    if not jobs or any(not isinstance(job, RhoBatchJob) for job in jobs):
        raise ValueError("Supply at least one RhoBatchJob.")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Every batch job must have a unique job_id.")
    destinations = [solution_destination(
        job.experiment_path, job.run_id, storage=storage,
        bucket_root=bucket_root, project_root=project_root,
    ) for job in jobs]
    if len(set(destinations)) != len(jobs):
        raise ValueError("Every batch job must have a distinct saved destination.")
    directory = Path(record_directory).expanduser().resolve()
    progress = directory / "progress.json" if progress_path is None else Path(progress_path)
    # HDF5 access and fingerprinting precede submissions. Thread workers share
    # immutable source arrays, but build independent dynamics and execution state.
    resolved = resolve_h5_source(source)
    with resolved.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if expected_source_sha256 is not None and digest != expected_source_sha256:
        raise ValueError("The HDF5 source does not match expected_source_sha256.")
    fields = {}
    for job in jobs:
        key = _field_key(job.config)
        if key not in fields:
            fields[key] = load_dimensional_h5_field(
                resolved, magnetic_field=job.config.magnetic_field,
                characteristic_length=job.config.characteristic_length,
                selectors=job.config.source_selection,
                interpolation_order=job.config.interpolation_order,
            )
    report: dict[str, Any] = {
        "schema_version": 1, "app_name": app_name,
        "requested_max_workers": max_workers, "source_sha256": digest,
        "execution_backend": "jax", "execution_device": "cpu", "storage": storage,
        "started_unix_seconds": time(), "status": "running",
        "overlap_scope": "Returned worker intervals intersected with this batch invocation; excludes provisioning and transfers.",
        "jobs": [dict(job_id=job.job_id, method=job.config.method,
                      rho_hat=float(job.config.rho_hat), destination=destination,
                      receipt_directory=str(directory / job.job_id), status="pending")
                 for job, destination in zip(jobs, destinations)],
    }
    _update_summary(report)
    _write_progress(progress, report)

    def run_one(job: RhoBatchJob, destination: str) -> dict[str, Any]:
        """Keep credentials and exception messages outside the progress archive."""
        executor = None
        try:
            if job.probe_seeds is None:
                prepared = build_rho_star(fields[_field_key(job.config)], job.config,
                                          source_sha256=digest)
            else:
                prepared = build_gap_probes(fields[_field_key(job.config)], job.config,
                                            source_sha256=digest, seeds=job.probe_seeds)
            executor = modal_rho_executor(directory / job.job_id, app_name=app_name)
            stored = run_and_save_rho_star(
                prepared, destination, executor=executor,
                options=ExecutionOptions(backend="jax", device="cpu"),
                save_folded_returns=save_folded_returns,
            )
            diagnostics = stored.solution.diagnostics
            if diagnostics.get("execution_backend") != "jax":
                raise ValueError("The saved result does not report JAX execution.")
            result = dict(status="completed", execution_backend="jax",
                          modal_call_id=stored.metadata.get("modal_call_id"),
                          modal_receipt=stored.metadata.get("modal_receipt"))
            for name in ("step_count", "worker_wall_seconds", "worker_started_unix_seconds",
                         "worker_finished_unix_seconds"):
                if name in diagnostics:
                    result[name] = int(diagnostics[name]) if name == "step_count" else float(diagnostics[name])
            return result
        except Exception as exc:
            result = dict(status="failed", error_type=type(exc).__name__,
                          recovery_hint="Inspect the confirmed Modal receipt before retrying; failures are never resubmitted automatically.")
            if isinstance(exc, StorageError):
                # Only this fixed persistence message is copied; remote error
                # text may contain account details and is deliberately omitted.
                match = re.fullmatch(r"Publication failed\. Complete local archive retained at (.+)\.", str(exc))
                if match and Path(match.group(1)).is_dir():
                    result["recovery_archive"] = match.group(1)
            if executor is not None:
                result["modal_call_id"] = executor.last_call_id
                result["modal_receipt"] = str(executor.last_record_path) if executor.last_record_path else None
            return result

    # Future failures do not cancel siblings. Main-thread-only snapshots avoid
    # races while independent SDK calls wait in separate client threads.
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rho-modal") as pool:
        futures = {pool.submit(run_one, job, destination): index
                   for index, (job, destination) in enumerate(zip(jobs, destinations))}
        for row in report["jobs"]:
            row["status"] = "running"
        _write_progress(progress, report)
        for future in as_completed(futures):
            report["jobs"][futures[future]].update(future.result())
            _update_summary(report)
            _write_progress(progress, report)
    report["status"] = "failed" if report["failed_count"] else "completed"
    report["finished_unix_seconds"] = time()
    _update_summary(report)
    _write_progress(progress, report)
    return report


__all__ = ["RhoBatchJob", "run_modal_rho_batch"]
