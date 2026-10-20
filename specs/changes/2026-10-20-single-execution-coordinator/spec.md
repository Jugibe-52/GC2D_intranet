+++
schema_version = 2
kind = "change-spec"
id = "2026-10-20-single-execution-coordinator"
revision = 1
status = "verified"
approval = "2026-10-20: User explicitly requested SDD preparation and implementation of the reviewed consolidation: one execution coordinator, executor-owned hybrid adaptation, and preserved standalone JaxPotential."
sources = ["src/execution/execution.py", "src/execution/_host.py", "src/execution/_jax.py", "src/execution/_jax_fixed.py", "src/methods/_jax_dispatch.py", "src/states/history.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md", "docs/dynamics/jax-potential-evaluation.md"]
requirements = ["SIM-003", "SIM-004", "STA-001", "STA-003"]
proposed_requirements = []
open_questions = []

[[criteria]]
id = "AC-001"
description = "One coordinator owns fresh-run validation, status transitions, result allocation and finalization for host and JAX execution."
requirements = ["SIM-003", "SIM-004"]

[[criteria]]
id = "AC-002"
description = "JAX preparation and the SciPy/JAX adapter belong to execution; physics has no dependency on execution. Standalone JaxPotential and shared spline data remain supported."
requirements = ["SIM-003"]

[[criteria]]
id = "AC-003"
description = "Fixed and adaptive algorithms, physical histories, sample independence and every baseline diagnostic remain exactly unchanged."
requirements = ["STA-003", "SIM-004"]

[[criteria]]
id = "AC-004"
description = "Bulk physical recording preserves ownership, shape, finiteness and completion checks without per-sample Python work after JAX."
requirements = ["STA-001", "STA-003"]

[[criteria]]
id = "AC-005"
description = "Custom providers/controllers, lifecycle failures, optional imports and mocked remote round trips remain valid."
requirements = ["SIM-003", "SIM-004"]

[[criteria]]
id = "AC-006"
description = "Supported imports, current contracts, model documentation, packaging and Spanish reading copies agree with the verified implementation."
requirements = ["SIM-003", "STA-003"]
+++

# Single execution coordinator: specification

## Problem

Execution.run delegates complete jobs to two internal coordinators that independently manage status and finalized States. JAX preparation is split into two modules, and dynamics contains an execution adapter that imports executor utilities. Compatibility checks also repeat the fixed-method support list.

## Scope

Consolidate complete-job ownership in Execution, merge JAX preparation, move the hybrid adapter out of dynamics, simplify numerical dispatch and retire redundant private modules and complete-integration functions. Preserve public method names, NumericalMethod capabilities, controllers, simulate/Execution signatures and JaxPotential. No change to equations, algorithms, tolerances, global JAX settings, archives, remote deployment, notebooks outputs, TFM content or Git publication.

## Proposed behavior

Execution prepares a fresh run, owns its lifecycle on success and failure, allocates one physical recorder and finalizes the result once. Host and compiled loops perform advancement, sampling and numerical export only. SciPy retains adaptive acceptance, rejected trials and dense output; JAX retains device-resident loops. A complete compiled history can be recorded in bulk before the same finalization boundary.

The executor selects simulation devices and prepares built-in GC/FC dynamics using JaxPotential without refitting splines or mutating physical inputs. JaxPotential remains independently usable and owns evaluation kernels and concrete coefficient caches. Local evaluator guards remain meaningful outside simulation; centralization does not remove numerical JAX implementations.

GC/FC physical histories retain shape (d*N,T), component-major coordinates and existing physical units. Main accepted trajectories must be independent of off-grid shadow samples. Compare all 23 deterministic CPU/JAX baseline cases exactly, including every diagnostic. Existing accuracy and convergence tolerances remain unchanged.
