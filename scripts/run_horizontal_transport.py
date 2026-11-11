"""Repeat the archived experiment with full-step horizontal quadrature on Modal.

Preparation downloads metadata only. Paid work requires --mode benchmark or run.
Completed archives and confirmed Modal receipts are reused, never resubmitted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
from time import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/developements/poincare_section/four_methods_star_40_rho_sweep_folded_radial_potential"
DEFAULT_OUTPUT = NOTEBOOK / "horizontal_transport"
SOURCE = ROOT / "data/potential/V1/PHI_2.h5"
APP = "gc2d-horizontal-transport"
SOURCE_HASH = "4bbd873cd05992e0a615d66ebbf4d96b38c8b3c36be2ca62932e66af7e6d4ced"
HTML_HASH = "7f04b36ccf18b684cfdc7eb16bd31a46ea730abfa566671e0bd33fda2cbc1705"
METHODS = {
    "BM4Implicit": "bm4_implicit", "GaussLegendre4": "gauss_legendre",
    "BM4Midpoint": "bm4_midpoint", "RK4": "rk4",
}
REMOTE = "gc2d_data:gc2d-notebooks-data/developements/poincare_section"


def write_json(path: Path, data: Any) -> None:
    """Atomically record progress without allowing non-JSON numeric values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False,
                                   default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + "\n")
    temporary.replace(path)


def jobs() -> list[dict[str, Any]]:
    """Retain separate forty-star and eight-probe implicit solve groups."""
    result = []
    for method, slug in METHODS.items():
        for k in range(11):
            rho = k / 20
            for group in ("star", "probes"):
                digits = 2 if method == "BM4Midpoint" and group == "star" else 6
                suffix = f"rho_{rho:.{digits}f}_v1".replace(".", "_")
                folder = f"{slug}_star_40_rho_0_30_folded_radial_potential"
                remote = f"{REMOTE}/{folder}/" + ("gap_probes/" if group == "probes" else "") + suffix
                result.append(dict(id=f"{slug}_{k:02d}_{group}", method=method, rho=rho,
                                   group=group, remote=remote))
    return result


def fetch_metadata(output: Path) -> None:
    """Fetch and authenticate original experiment metadata with its manifest."""
    directory = output / "original_metadata"
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(job: dict[str, Any]) -> None:
        path = directory / f"{job['id']}.json"
        manifest_path = directory / f"{job['id']}.manifest.json"
        for name, target in (("manifest.json", manifest_path), ("metadata.json", path)):
            if not target.exists():
                data = subprocess.run(["rclone", "cat", f"{job['remote']}/{name}", "--retries", "2"],
                                      capture_output=True, check=True, timeout=120).stdout
                target.write_bytes(data)
        expected = json.loads(manifest_path.read_text())["files"]["metadata.json"]["sha256"]
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original metadata fingerprint mismatch: {job['id']}")
        experiment = json.loads(path.read_text())["experiment"]
        if experiment["source_sha256"] != SOURCE_HASH:
            raise ValueError("Unexpected original field fingerprint.")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, jobs()))
    write_json(output / "original_jobs.json", jobs())
    print("Validated original metadata: 88 cases.", flush=True)


def billing() -> dict[str, Any]:
    """Read current API rates and workspace charges without changing settings."""
    import modal
    workspace = modal.Workspace.from_context()
    summary = workspace.billing.summary()
    rates = workspace.billing.rates()
    hourly = 2 * float(rates["cpu_hour_cost"]) + 4 * float(rates["mem_gib_hour_cost"])
    return dict(workspace=workspace.name, metered_usd=float(summary.metered_cost),
                billed_usd=float(summary.billed_cost), max_worker_hour_usd=hourly,
                checked_utc=datetime.now(timezone.utc).isoformat())


def run_one(job: dict[str, Any], output: Path, *, cycles: int = 5000,
            steps: int = 50, benchmark: bool = False, stop: Event | None = None) -> dict[str, Any]:
    """Recover a confirmed invocation or create one fresh bounded cloud job."""
    from diagnostics.execution.horizontal_transport import HorizontalTransportObserver
    from diagnostics.io.persistence import load_solution, save_solution
    from execution.options import ExecutionOptions
    from execution.execution_modal import Execution_Modal
    from studies.horizontal_transport_reproduction import prepare_horizontal_reproduction
    from studies.poincare_rho_sweep import _rho_star_method, validate_rho_solution, folded_rho_positions

    name = job["id"] + (f"_{cycles}c_{steps}s" if benchmark else "")
    destination = output / ("benchmarks" if benchmark else "runs") / name
    original = json.loads((output / "original_metadata" / f"{job['id']}.json").read_text())["experiment"]
    prepared = prepare_horizontal_reproduction(SOURCE, original, cycles=cycles, steps_per_cycle=steps)
    options = ExecutionOptions(backend="jax", device="cpu")
    if (destination / "manifest.json").exists():
        stored = load_solution(destination)
        result = stored.solution
        if stored.metadata.get("original_archive") != job["remote"]:
            raise ValueError("Completed archive has different provenance.")
    else:
        if stop is not None and stop.is_set():
            raise RuntimeError("Batch budget guard stopped further submissions.")
        records = output / "receipts" / name
        confirmed = []
        for path in sorted(records.glob("*.json")):
            record = json.loads(path.read_text())
            if not record.get("call_id"):
                raise RuntimeError(f"Inspect unconfirmed submission before continuing: {path}")
            confirmed.append((record["call_id"], path))
        if len({row[0] for row in confirmed}) > 1:
            raise RuntimeError("Multiple call IDs for one job; refusing to resubmit.")
        executor = (Execution_Modal.from_record(confirmed[-1][1]) if confirmed else
                    Execution_Modal(app_name=APP, function_name="benchmark" if benchmark else "integrate",
                                    record_directory=records))
        result = executor.run(prepared.problem.with_request(prepared.request), _rho_star_method(prepared.config),
                              options=options, observers=(HorizontalTransportObserver(),))
        validate_rho_solution(result, prepared, options=options)
        wrapped, fractions = folded_rho_positions(result, prepared.metadata)
        result = result.with_diagnostics({"cycle_positions_wrapped": wrapped,
                                          "cycle_positions_cell_fraction": fractions})
        save_solution(result, destination, metadata={**prepared.metadata,
            "original_archive": job["remote"], "execution_options": asdict(options),
            "execution_executor": "modal", "modal_call_id": executor.last_call_id,
            "horizontal_metric": "integral_abs_vx/(cycles*(max_x-min_x))",
            "saved_folded_returns": True})
        stored = load_solution(destination)
    expected_config = json.loads(json.dumps(asdict(prepared.config)))
    if stored.metadata.get("config") != expected_config or stored.metadata.get("source_sha256") != SOURCE_HASH:
        raise ValueError("Saved scientific configuration or source fingerprint differs from the requested case.")
    validate_rho_solution(result, prepared, options=options)
    d = result.diagnostics
    if d.get("horizontal_observed_steps") != cycles * steps:
        raise ValueError("Observer must cover every complete accepted step.")
    if not np.all(np.isfinite(d["horizontal_travel"])):
        raise ValueError("Nonfinite horizontal integral.")
    if d.get("horizontal_quadrature") != "trapezoid_accepted_endpoints" or not np.isclose(d["horizontal_cycles"], cycles):
        raise ValueError("Unexpected quadrature or cycle normalization.")
    return dict(id=name, status="completed", method=job["method"], rho=job["rho"], group=job["group"],
                destination=str(destination), cycles=cycles, steps_per_cycle=steps,
                worker_seconds=float(d["worker_wall_seconds"]), call_id=d.get("execution_call_id"),
                worker_start=float(d["worker_started_unix_seconds"]),
                worker_end=float(d["worker_finished_unix_seconds"]))


def run_batch(output: Path, *, benchmark: bool) -> None:
    """Use bounded independent jobs, never automatic resubmission of failures."""
    before = billing()
    # Even all timeout ceilings cost less than USD 9 at current workspace rates;
    # reserve USD 2 for startup/build/egress. This is well below the EUR 15 budget.
    worst_compute = before["max_worker_hour_usd"] * (88 * 2400 + 16 * 600) / 3600
    if worst_compute + 2 > 11:
        raise RuntimeError("Resource rates exceed the conservative USD 11 budget envelope.")
    baseline = output / "billing_baseline.json"
    if not baseline.exists():
        write_json(baseline, before)
    if before["metered_usd"] - json.loads(baseline.read_text())["metered_usd"] > 8:
        raise RuntimeError("Reached the conservative usage stop threshold.")
    selected = [j for j in jobs() if j["rho"] == .3 and j["group"] == "star"] if benchmark else jobs()
    invocations = [(j, 100, 50) for j in selected] if benchmark else [(j, 5000, 50) for j in selected]
    if benchmark:
        # Short matched trajectories distinguish quadrature refinement from chaotic drift.
        invocations += [(j, 1, n) for j in selected for n in (50, 100, 200)]
    report = dict(status="running", started=time(), benchmark=benchmark, before=before,
                  timeout_compute_ceiling_usd=worst_compute, jobs=[])
    progress = output / ("benchmark_progress.json" if benchmark else "progress.json")
    write_json(progress, report)
    stop = Event()
    finished = Event()

    def watch_budget() -> None:
        """Stop this task's confirmed calls if actual spend or elapsed time grows."""
        import modal
        original_charge = json.loads(baseline.read_text())["metered_usd"]
        deadline_seconds = 1200 if benchmark else 3600
        while not finished.wait(30):
            reason = ""
            if time() - report["started"] > deadline_seconds:
                reason = "Batch wall time reached its conservative cost limit."
            try:
                current = billing()
                write_json(output / "billing_latest.json", current)
                if current["metered_usd"] - original_charge >= 8:
                    reason = "Workspace metered usage reached the USD 8 stop threshold."
            except Exception:
                # The independent wall-clock ceiling still applies during API delays.
                pass
            if reason:
                stop.set()
                write_json(output / "budget_stop.json", dict(reason=reason, at=time()))
                for path in (output / "receipts").glob("*/*.json"):
                    record = json.loads(path.read_text())
                    if record.get("call_id"):
                        try:
                            modal.FunctionCall.from_id(record["call_id"]).cancel(terminate_containers=True)
                        except Exception:
                            pass
                return

    watcher = Thread(target=watch_budget, daemon=True)
    watcher.start()
    with ThreadPoolExecutor(max_workers=4 if benchmark else 44) as pool:
        futures = {pool.submit(run_one, j, output, cycles=c, steps=s, benchmark=benchmark, stop=stop): (j,c,s)
                   for j,c,s in invocations}
        for future in as_completed(futures):
            job, cycles, steps = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(id=job["id"], cycles=cycles, steps_per_cycle=steps,
                           status="failed", error_type=type(exc).__name__, error=str(exc))
            report["jobs"].append(row)
            report["updated"] = time()
            write_json(progress, report)
            print(json.dumps(row), flush=True)
    finished.set()
    watcher.join(timeout=5)
    report.update(status="failed" if any(x["status"] != "completed" for x in report["jobs"]) else "completed",
                  finished=time(), after=billing())
    write_json(progress, report)
    if report["status"] != "completed":
        raise RuntimeError("Some jobs failed. Inspect receipts and errors before any retry.")


def publish_data(output: Path) -> None:
    """Join original groups by particle ID and export matching fresh trajectories."""
    from diagnostics.io.persistence import load_solution
    from visualization.poincare_transport import export_horizontal_transport_viewer
    cases = []
    trajectories = {}
    for method in METHODS:
        for k in range(11):
            selected = [j for j in jobs() if j["method"] == method and j["rho"] == k/20]
            for job in selected:
                if not (output / "runs" / job["id"] / "manifest.json").exists():
                    raise ValueError("Publication requires every completed archive; it never launches jobs.")
                run_one(job, output)
            records = [load_solution(output / "runs" / j["id"]) for j in selected]
            ids = np.concatenate([r.metadata["particle_id"] for r in records])
            if not np.array_equal(ids, np.arange(1,49)):
                raise ValueError("Expected original star IDs 1-40 and probe IDs 41-48.")
            case = dict(rho=k/20, method=method, particle_ids=ids.tolist(), cycles=5000,
                        steps_per_cycle=50, source_sha256=SOURCE_HASH)
            for public, diagnostic in (("q_x", "horizontal_transport_rate"), ("travel_x", "horizontal_travel"),
                                       ("width_x", "horizontal_width"), ("x_min", "horizontal_min"),
                                       ("x_max", "horizontal_max")):
                values = np.concatenate([r.solution.diagnostics[diagnostic] for r in records])
                case[public] = [float(v) if np.isfinite(v) else None for v in values]
            cases.append(case)
            trajectories[(k/20,method)] = np.concatenate(
                [r.solution.diagnostics["cycle_positions_cell_fraction"] for r in records], axis=1)
    summary = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                   metric=dict(coordinate="physical_unwrapped_x", integral="accepted_endpoint_trapezoid_abs_vx",
                               source_html_sha256=HTML_HASH, source_sha256=SOURCE_HASH,
                               units="m", normalization="cycles*(max_x-min_x)", cycles=5000), cases=cases)
    write_json(output.parent / "horizontal_transport_results.json", summary)
    html = output.parent / "four_methods_poincare.html"
    original = output / "original_four_methods_poincare.html"
    if not original.exists():
        if sha256(html.read_bytes()).hexdigest() != HTML_HASH:
            raise ValueError("The original viewer changed; verify before publication.")
        original.write_bytes(html.read_bytes())
    export_horizontal_transport_viewer(original, summary, html, cycle_positions=trajectories)
    print(f"Exported {len(cases)*48} values with matching new cycle histories.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "benchmark", "run", "publish-data"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        fetch_metadata(args.output)
    elif args.mode == "publish-data":
        publish_data(args.output)
    else:
        run_batch(args.output, benchmark=args.mode == "benchmark")


if __name__ == "__main__":
    main()
