+++
schema_version = 2
kind = "change-spec"
revision = 1
status = "verified"
approval = "2026-10-20: User accepted the executor-owned integration design discussed in this task and explicitly invoked SDD to prepare and execute it."
id = "2026-10-20-executor-owned-integration"
sources = ["src/execution/execution.py", "src/integration/core.py", "src/methods/base.py", "src/methods/adaptive/scipy.py", "src/methods/_jax_dispatch.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md", "docs/simulation/jax-execution.md"]
requirements = ["SIM-003", "SIM-004", "STA-003"]
proposed_requirements = []
open_questions = []

[[criteria]]
id = "AC-001"
description = "Execution owns complete integration preparation and driver selection; methods expose preparation and numerical operations without integrate/_integrate_jax; CompiledFixedMethod is removed."
requirements = ["SIM-003"]

[[criteria]]
id = "AC-002"
description = "Fixed/adaptive GC/FC histories, energy diagnostics, accepted steps and independent shadow sampling retain their numerical behavior on SciPy and available JAX devices."
requirements = ["SIM-003", "STA-003"]

[[criteria]]
id = "AC-003"
description = "Fresh runs, failure propagation, optional JAX imports, callback constraints, unsupported types and device validation remain explicit."
requirements = ["SIM-003"]

[[criteria]]
id = "AC-004"
description = "simulate and Modal retain their job/result signatures and validation; structural method providers use the new preparation contract."
requirements = ["SIM-003", "SIM-004"]

[[criteria]]
id = "AC-005"
description = "Supported callers, current contracts, model documentation and Spanish reading copies reflect the implemented lifecycle; repository checks pass."
requirements = ["SIM-003", "SIM-004", "STA-003"]
+++

# Executor-owned integration: specification

## Problem

Execution.run forwards the complete job to method.integrate. IntegrationMethod then owns preparation and backend dispatch, while CompiledFixedMethod and adaptive methods arrange JAX execution. This distributes execution responsibilities through the numerical inheritance hierarchy.

## Scope

Implement the reviewed ownership change across local and remote execution, numerical method contracts, supported callers, tests and architecture documentation. Remove CompiledFixedMethod; retain IntegrationMethod, controllers, numerical methods, state storage and executor classes. No numerical formulas, tolerances, unit conventions, sampling policies, archive schemas, remote deployment or TFM edits are included.

## Proposed behavior

Execution prepares a fresh method run, validates the requested execution capability and environment, and invokes the appropriate shared integration driver. Methods retain initialization, numerical advancement, observations and controller capabilities. They no longer launch complete integrations or resolve devices. NumericalMethod describes preparation instead of integrate. The supported complete-job APIs remain simulate(problem, method, request, execution=..., options=...) and Execution.run(problem, method, request, options=...). Direct method.integrate consumers migrate to simulate; no compatibility wrapper is retained for the retired route.

The default path remains local NumPy/SciPy without importing JAX. Fixed JAX paths retain device-resident loops; DOP853/Radau retain SciPy acceptance and dense output with prepared JAX field evaluations. Device resolution and execution metadata belong to execution. Numerical adapters retain algorithm-specific kernel construction and explicit rejection of unsupported overrides. Executors select generic execution capabilities rather than branching on BM4 or RK4 identities.

Saved physical arrays remain (d*N,T), with GC components x,y and FC x,y,vx,vy, unchanged normalization and strictly identical requested times. Configurations remain reusable; per-run resources are independent. Fixed samples inside a step remain independent shortened maps from its left endpoint. Nonlinear tolerances and convergence criteria are unchanged. Compare deterministic before/after results on identical inputs; bitwise equality is expected where the arithmetic is unchanged. Cross-backend tests retain their existing scientific tolerances.
