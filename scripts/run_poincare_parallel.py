"""Prepare, run and audit the authorized 36-neighbour Modal JAX study."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Event, Thread
from time import time

import numpy as np

from diagnostics.execution.horizontal_transport import HorizontalTransportObserver
from diagnostics.io.persistence import load_solution, save_solution
from execution.execution_modal import Execution_Modal
from execution.options import ExecutionOptions
from studies.poincare_parallel import METHOD_SLUGS, build_seed_manifest, prepare_parallel
from studies.poincare_rho_sweep import _rho_star_method, folded_rho_positions, validate_rho_solution
from run_horizontal_transport import billing, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/developements/poincare_section/four_methods_star_40_rho_sweep_folded_radial_potential"
OUTPUT = ROOT / "notebooks/developements/poincare_section/parallel_poincare_families"
SOURCE = ROOT / "data/potential/V1/PHI_2.h5"
APP = "gc2d-parallel-poincare"


def prepare() -> dict:
    """Freeze seed geometry and the exact published inputs before paid work."""
    receipt = json.loads((BASE / "cloudflare_pages/deployment.json").read_text())
    for filename, key in (("horizontal_transport_results.json", "results_sha256"),
                          ("four_methods_poincare.html", "source_html_sha256")):
        if sha256((BASE/filename).read_bytes()).hexdigest() != receipt[key]:
            raise ValueError(f"Published baseline changed: {filename}")
    seeds = build_seed_manifest(BASE / "horizontal_transport/runs")
    path = OUTPUT / "seeds.json"
    if path.exists() and json.loads(path.read_text()) != seeds:
        raise ValueError("Existing seed manifest differs; do not silently change initial conditions.")
    write_json(path, seeds)
    write_json(OUTPUT/"baseline.json", dict(source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),
        original_html_sha256=receipt["source_html_sha256"], original_results_sha256=receipt["results_sha256"],
        seeds_sha256=sha256(path.read_bytes()).hexdigest(), original_ids=list(range(1,49)),
        new_ids=list(range(49,85)), rho_values=[i/20 for i in range(11)]))
    return seeds


def run_one(method: str, rho_index: int, *, cycles: int = 5000, steps: int = 50,
            benchmark: bool = False, stop: Event | None = None, recovery: bool = False,
            retry_timeout: bool = False) -> dict:
    """Submit once or recover a confirmed call, then validate its complete result."""
    name = f"{METHOD_SLUGS[method]}_{rho_index:02d}" + (f"_{cycles}c_{steps}s" if benchmark else "")
    destination = OUTPUT / ("benchmarks" if benchmark else "runs") / (name+"_damped" if benchmark and recovery else name)
    if (destination/"metadata.json").exists():
        recovery = bool(json.loads((destination/"metadata.json").read_text())["experiment"].get("solver_recovery"))
    if recovery and method != "BM4Implicit":
        raise ValueError("Damped Newton recovery is limited to failed BM4Implicit cases.")
    seeds = json.loads((OUTPUT/"seeds.json").read_text())
    original = json.loads((BASE/f"horizontal_transport/runs/{METHOD_SLUGS[method]}_{rho_index:02d}_star/metadata.json").read_text())
    prepared = prepare_parallel(SOURCE, original, seeds, cycles=cycles, steps_per_cycle=steps)
    if recovery:
        config=replace(prepared.config,newton_max_iterations=80)
        prepared=replace(prepared,config=config,metadata={**prepared.metadata,"config":asdict(config),
            "solver_recovery":{"reason":"Confirmed undamped Newton nonconvergence", "backtracking":True,
                "max_iterations":80,"unchanged":"seeds, field, fixed step, duration and tolerances"}})
    numerical_method=_rho_star_method(prepared.config)
    if recovery:numerical_method=replace(numerical_method,solver=replace(numerical_method.solver,backtracking=True))
    options = ExecutionOptions(backend="jax", device="cpu")
    if (destination/"manifest.json").exists():
        stored = load_solution(destination)
        if stored.metadata.get("seed_manifest") != seeds or stored.metadata.get("config") != json.loads(json.dumps(asdict(prepared.config))):
            raise ValueError("Saved seeds or scientific configuration differ.")
        result = stored.solution
    else:
        if stop is not None and stop.is_set():
            raise RuntimeError("The study budget guard stopped further submissions.")
        suffix = "_damped" if recovery else "_retry_timeout" if retry_timeout else ""
        records = OUTPUT/"receipts"/(name+suffix)
        confirmed = []
        for path in sorted(records.glob("*.json")):
            record = json.loads(path.read_text())
            if not record.get("call_id"):
                raise RuntimeError(f"Inspect unconfirmed Modal submission before retrying: {path}")
            confirmed.append((record["call_id"],path))
        if len({c[0] for c in confirmed}) > 1:
            raise RuntimeError("Multiple Modal calls exist for this job.")
        executor = (Execution_Modal.from_record(confirmed[-1][1]) if confirmed else
                    Execution_Modal(app_name=APP, function_name="benchmark" if benchmark else "integrate",
                                    record_directory=records))
        result = executor.run(prepared.problem.with_request(prepared.request),
            numerical_method, options=options, observers=(HorizontalTransportObserver(),))
        validate_rho_solution(result, prepared, options=options)
        wrapped, fractions = folded_rho_positions(result, prepared.metadata)
        result = result.with_diagnostics({"cycle_positions_wrapped":wrapped,"cycle_positions_cell_fraction":fractions})
        save_solution(result, destination, metadata={**prepared.metadata,
            "execution_options":asdict(options),"execution_executor":"modal","modal_call_id":executor.last_call_id,
            "execution_retry": {"reason":"Confirmed FunctionTimeoutError", "attempt":2} if retry_timeout else None,
            "horizontal_metric":"integral_abs_vx/(cycles*(max_x-min_x))","saved_folded_returns":True})
        stored = load_solution(destination)
    validate_rho_solution(result, prepared, options=options)
    d = result.diagnostics
    if d["horizontal_observed_steps"] != cycles*steps or d["horizontal_cycles"] != cycles:
        raise ValueError("Horizontal observer did not cover the full integration.")
    widths = d["horizontal_width"]
    q = d["horizontal_transport_rate"]
    if not np.isfinite(q).all() or np.any(widths<=0):
        raise ValueError("Unexpected undefined transport metric.")
    np.testing.assert_allclose(q, d["horizontal_travel"]/(cycles*widths),rtol=1e-14,atol=0)
    if stored.metadata["source_sha256"] != sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError("Saved field fingerprint differs.")
    return dict(id=name, method=method,rho=rho_index/20,status="completed",cycles=cycles,
        steps_per_cycle=steps,worker_seconds=float(d["worker_wall_seconds"]),
        call_id=d.get("execution_call_id"),path=str(destination),particles=36,solver_recovery=recovery,
        execution_retry=stored.metadata.get("execution_retry"),
        grouping_recovery=stored.metadata.get("grouping_recovery"),
        individual_call_ids=stored.metadata.get("modal_call_ids"))


def run_batch(benchmark: bool) -> None:
    """Run bounded independent cases and retain every result and receipt."""
    prepare()
    before = billing()
    baseline = OUTPUT/"billing_before.json"
    if not baseline.exists(): write_json(baseline,before)
    limit_usd = 6.0
    invocations = ([(m,6,c,s) for m in METHOD_SLUGS for c,s in [(1,50),(1,100),(20,50)]]
                   if benchmark else [(m,k,5000,50) for m in METHOD_SLUGS for k in range(11)])
    ceiling = before["max_worker_hour_usd"] * len(invocations) * (600 if benchmark else 2400)/3600
    if ceiling > limit_usd: raise ValueError(f"Worst-case worker ceiling {ceiling:.2f} USD exceeds {limit_usd} USD.")
    report = dict(status="running",started=time(),before=before,worker_cost_ceiling_usd=ceiling,
                  stop_threshold_usd=limit_usd,jobs=[])
    report_path = OUTPUT/("benchmark_progress.json" if benchmark else "progress.json")
    write_json(report_path,report)
    finished,stop = Event(),Event()

    def guard() -> None:
        import modal
        original = json.loads(baseline.read_text())["metered_usd"]
        while not finished.wait(30):
            reason = ""
            if time()-report["started"] > (1800 if benchmark else 3000): reason="Wall-time ceiling reached."
            try:
                current=billing();write_json(OUTPUT/"billing_latest.json",current)
                if current["metered_usd"]-original >= limit_usd:reason="Study cost ceiling reached."
            except Exception:
                pass
            if reason:
                stop.set();write_json(OUTPUT/"budget_stop.json",dict(reason=reason,at=time()))
                for path in (OUTPUT/"receipts").glob("*/*.json"):
                    record=json.loads(path.read_text())
                    if record.get("call_id"):
                        try:modal.FunctionCall.from_id(record["call_id"]).cancel(terminate_containers=True)
                        except Exception:pass
                return

    watcher=Thread(target=guard,daemon=True);watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=4 if benchmark else 44) as pool:
            futures={pool.submit(run_one,m,k,cycles=c,steps=s,benchmark=benchmark,stop=stop):(m,k,c,s)
                     for m,k,c,s in invocations}
            for future in as_completed(futures):
                m,k,c,s=futures[future]
                try:row=future.result()
                except Exception as exc:row=dict(method=m,rho=k/20,cycles=c,steps=s,status="failed",error=str(exc),error_type=type(exc).__name__)
                report["jobs"].append(row);report["updated"]=time();write_json(report_path,report)
                print(json.dumps(row),flush=True)
    finally:
        finished.set();watcher.join(timeout=5)
    report.update(status="failed" if any(j["status"]!="completed" for j in report["jobs"]) else "completed",
                  finished=time(),after=billing())
    write_json(report_path,report)
    if report["status"]!="completed":raise RuntimeError("Inspect failed cases before any new submission.")


def audit() -> None:
    """Validate every complete archive without submitting missing jobs."""
    for m in METHOD_SLUGS:
        for k in range(11):
            if not (OUTPUT/f"runs/{METHOD_SLUGS[m]}_{k:02d}/manifest.json").exists():
                raise ValueError("Audit requires all 44 completed archives.")
    rows=[run_one(m,k) for m in METHOD_SLUGS for k in range(11)]
    write_json(OUTPUT/"audit.json",dict(status="passed",jobs=rows,trajectories=1584,
        total_worker_seconds=sum(r["worker_seconds"] for r in rows)))
    print("Audited 44 jobs and 1584 trajectories.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("prepare","benchmark","run","audit","recover","retry-timeout"),required=True)
    parser.add_argument("--rho-index",type=int)
    parser.add_argument("--method",choices=tuple(METHOD_SLUGS))
    args=parser.parse_args()
    if args.mode=="prepare":print(json.dumps(dict(particles=len(prepare()["particles"]),output=str(OUTPUT))))
    elif args.mode=="audit":audit()
    elif args.mode=="retry-timeout":
        if args.rho_index not in range(11) or args.method is None:
            raise ValueError("A timeout retry requires a method and rho index from 0 through 10.")
        progress=json.loads((OUTPUT/"progress.json").read_text())
        failures=[j for j in progress["jobs"] if j["status"]=="failed" and j["method"]==args.method
                  and j["rho"]==args.rho_index/20 and j.get("error_type")=="FunctionTimeoutError"]
        if len(failures)!=1:raise ValueError("Retry requires one confirmed remote execution timeout.")
        current=billing();baseline=json.loads((OUTPUT/"billing_before.json").read_text())
        if current["metered_usd"]-baseline["metered_usd"]+current["max_worker_hour_usd"]*2400/3600>=6:
            raise ValueError("Insufficient remaining study budget for a bounded retry.")
        row=run_one(args.method,args.rho_index,retry_timeout=True)
        write_json(OUTPUT/f"retry_{METHOD_SLUGS[args.method]}_{args.rho_index:02d}.json",
                   dict(original_failure=failures[0],result=row))
        print(json.dumps(row),flush=True)
    elif args.mode=="recover":
        if args.rho_index not in range(11):raise ValueError("Recovery requires a rho index from 0 through 10.")
        progress=json.loads((OUTPUT/"progress.json").read_text())
        failures=[j for j in progress["jobs"] if j["status"]=="failed" and j["method"]=="BM4Implicit"
                  and j["rho"]==args.rho_index/20 and "did not converge" in j.get("error","")]
        if len(failures)!=1:raise ValueError("Recovery requires one confirmed nonlinear failure.")
        current=billing();baseline=json.loads((OUTPUT/"billing_before.json").read_text())
        if current["metered_usd"]-baseline["metered_usd"]+current["max_worker_hour_usd"]*2400/3600>=6:
            raise ValueError("Insufficient remaining study budget for a bounded recovery.")
        row=run_one("BM4Implicit",args.rho_index,recovery=True)
        write_json(OUTPUT/f"recovery_{args.rho_index:02d}.json",dict(original_failure=failures[0],result=row))
        print(json.dumps(row),flush=True)
    else:run_batch(args.mode=="benchmark")
