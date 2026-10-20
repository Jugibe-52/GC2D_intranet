+++
schema_version = 2
kind = "change-spec"
id = "2026-10-20-unified-execution"
revision = 1
status = "verified"
approval = "2026-10-20: User accepted the discussed merger of integration into execution, with the numerical base in methods and internal CPU/JAX loops, and explicitly invoked SDD to prepare and execute it."
sources = ["src/execution/execution.py", "src/execution/_host.py", "src/execution/state.py", "src/methods/base.py", "src/methods/adaptive/scipy.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md"]
requirements = ["SIM-003", "SIM-004", "STA-003"]
proposed_requirements = []
open_questions = []

[[criteria]]
id = "AC-001"
description = "Execution contains preparation, host/device time loops, sampling and current snapshots; the separate integration package is removed."
requirements = ["SIM-003"]

[[criteria]]
id = "AC-002"
description = "The numerical base belongs to methods; supported callers and extension imports use canonical paths without compatibility aliases."
requirements = ["SIM-003"]

[[criteria]]
id = "AC-003"
description = "Physical samples, accepted steps, energy and diagnostics remain exactly equal to the deterministic baseline; independent sampling and lifecycle are preserved."
requirements = ["SIM-004", "STA-003"]

[[criteria]]
id = "AC-004"
description = "CPU imports remain independent of optional JAX/Modal; local and remote job/result contracts, serialized results and isolated imports remain valid."
requirements = ["SIM-003", "SIM-004"]

[[criteria]]
id = "AC-005"
description = "Packaging, current contracts, affected documentation and Spanish copies reflect the unified structure; required verification succeeds."
requirements = ["SIM-003", "STA-003"]
+++

# Unified execution: specification

## Problem

The executor owns a complete simulation but still imports its time loops and controllers from a separate integration package. That package also owns the numerical base, mixing method preparation with execution. The extra boundary adds navigation and coupling without an independent runtime responsibility.

## Scope

Merge integration into execution and place the numerical base in methods. Migrate supported imports, package metadata, current contracts and affected documentation. Keep numerical formulas, state formulations, tolerances, output schedules, observers, archive schemas and public simulate/executor signatures unchanged. No remote deployment, TFM edits, commits or further synchronization are included.

## Proposed behavior

Execution.run remains the complete-job entry point. Execution internally owns preparation, host and compiled JAX loops, fixed scheduling, sampling, progress and result collection. Methods own their reusable configuration, fresh runs, numerical advancement and solver-specific adaptive behavior. Modal remains a complete-job transport adapter. IntegrationMethod keeps its class name in methods.base. Controllers and IntegrationState keep their behavior with canonical execution imports. The old integration package is retired without aliases.

GC/FC physical arrays remain (d*N,T), with component order x,y and x,y,vx,vy. No units, Hamiltonian normalization, nonlinear or adaptive tolerances change. Independent off-grid shadow samples must not alter main accepted states; adaptive dense output and JAX device loops retain their arithmetic. Every saved time, physical value and diagnostic in the 23-case deterministic baseline must match exactly. Existing numerical tests retain their tolerances.
