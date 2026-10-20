+++
schema_version = 2
kind = "change-spec"
revision = 1
status = "verified"
approval = "2026-10-20: User reviewed the architecture and integration-flow proposal in conversation and explicitly requested SDD preparation and execution of this scope."
id = "2026-10-20-integration-state-results"
sources = ["src/methods/base.py", "src/execution/_host.py", "src/execution/state.py", "src/states/history.py", "src/formulations/layout.py", "src/request.py", "src/contracts/problem.py"]
documents = ["specs/current/states.md", "specs/current/results.md", "specs/current/simulation.md"]
open_questions = []
requirements = ["SIM-001", "SIM-002", "SIM-003", "SIM-004", "SOL-001", "SOL-002", "SOL-003", "STA-001", "STA-002", "STA-003"]
proposed_requirements = ["STA-003"]

[[criteria]]
id = "AC-001"
description = "Separate accepted internal state/time from saved physical samples; fixed shadow maps and adaptive dense output preserve numerical trajectories and diagnostics."
requirements = ["SIM-004", "STA-003"]

[[criteria]]
id = "AC-002"
description = "States owns recording, complete immutable results, layouts, provenance snapshots and distinct main-step/sample diagnostics; simulate returns that same result."
requirements = ["SOL-001", "SOL-002", "SOL-003", "STA-001", "SIM-003"]

[[criteria]]
id = "AC-003"
description = "Raw physical initial arrays and layouts initialize problems independently of request time; generators preserve exact coordinates."
requirements = ["SIM-001", "STA-002"]

[[criteria]]
id = "AC-004"
description = "Request is a standalone module; canonical layouts live in formulations.layout without importing numerical methods."
requirements = ["SIM-002", "STA-001"]

[[criteria]]
id = "AC-005"
description = "Remove State, Solution and IntegrationCollector; migrate consumers, archives, CPU/JAX boundaries, docs and thesis while preserving scientific behavior."
requirements = ["STA-001", "STA-002", "STA-003"]
+++
# Integration states and results: specification

## Problem

The current snapshot, collector and completed-result containers obscure the difference between the state that advances and the samples saved. Layout ownership and temporal configuration are distributed across old public routes.

## Scope

Implement the architecture reviewed in conversation: IntegrationState(time, values), States recording and final results, raw initial arrays, request module and reusable layouts in formulations. Migrate local supported consumers, saved archive readers, CPU/JAX execution, model documentation and the software chapter of the TFM. No numerical equations, units, step coefficients, solver tolerances, archive schemas, remote submissions or synchronization changes.

## Proposed behavior

Request owns t0, tf, maximum step and saved times; zero remains a default, not an imposed origin. Problems own validated initial values and layouts. Canonical GC/FC layouts can be selected from the dynamics dimension when omitted; explicit structural layouts remain supported. Generators return packed NumPy arrays, never a clock or a solver instance.

IntegrationState owns a finite one-dimensional internal vector and its finite time. Fixed control retains the interval-start snapshot and advances only accepted principal states. Independent shadow maps start at that same snapshot, endpoints are reused, and adaptive control continues using dense output. No saved sample aliases live work storage. JAX keeps equivalent functional device arrays and constructs the same completed States on return.

States records physical arrays of shape (d*N,T), times (T,), and optional per-sample auxiliary arrays. GC order is x then y; FC appends vx then vy, in existing units. Recording checks order, shapes, finiteness and completeness. Only recorded columns are readable; finalization prevents further writes and freezes diagnostics. Main-step statistics retain their own time axis. Preserve existing read-only t/states result access alongside clearer times/values names; remove retired class/module routes rather than aliasing them. Source identity is replaced by an independent initial-array snapshot. A loaded archive may explicitly have no initial snapshot. Layouts remain reusable NumPy/JAX operations; formulations are shared by multiple methods.

Eliminate Solution, IntegrationCollector and the initial State wrapper. Methods and execution return States; the runner validates the returned result against request and initial values without constructing another container. Distinguish internal samples and physical/auxiliary extraction; States has no numerical map or formulation-specific logic.


Reference note (2026-10-20): source references now resolve to methods.base and execution after the package merger. Historical design and results in this record retain their original execution context.
