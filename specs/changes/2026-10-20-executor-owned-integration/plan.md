+++
schema_version = 2
kind = "change-plan"
revision = 1
spec_revision = 1
id = "2026-10-20-executor-owned-integration"
sources = ["src/execution/execution.py", "src/integration/core.py", "src/methods/base.py", "src/methods/adaptive/scipy.py", "src/methods/_jax_dispatch.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md", "docs/simulation/jax-execution.md"]
open_questions = []
+++

# Executor-owned integration: implementation plan

## Design

Keep shared drivers in integration. Simplify IntegrationMethod to preparation, advancement, observations and controller access. Delete its integrate and _integrate_jax methods and remove CompiledFixedMethod; concrete fixed methods inherit IntegrationMethod directly. Preserve new_run's isolated dataclass construction and immutable snapshots. Replace run execution options with an optional evaluated-dynamics input so adaptive solvers can receive an already prepared field implementation.

NumericalMethod remains a structural protocol with new_run and a declarative JAX execution capability (unsupported, device-resident or hybrid SciPy control). Execution.run validates options, prepares the runtime and fresh run, invokes integrate_method or integrate_fixed, and records execution metadata. Optional JAX support is loaded lazily through an execution helper. Device preparation and host field binding receive an already resolved device. Method family adapters receive prepared dynamics and a device; they only bind algorithm controls and validate supported method types. The executor uses the capability contract without concrete numerical-method branches.

Retain StepController, FixedStepController and ScipyAdaptiveController. Retain IntegrationState, States and CompiledStep. Modal dispatches the same serialized complete job and uses Execution.run in its worker. Preparation and integration failures must leave consumed runs finished; returned data remain immutable and validated by simulate.

## Compatibility

Keep simulate, executor signatures, numerical constructor names/options, controller behavior, saved archives and global JAX configuration policy. Direct method.integrate and new_run(execution=...) are retired and callers migrate. External providers implement new_run and the declared JAX capability; prepared runs use IntegrationMethod's numerical interface. Preserve supported custom CPU dynamics and custom controllers. Unsupported compiled subclasses continue to fail explicitly. Existing prepared-kernel APIs are internal and may accept the resolved environment instead of ExecutionOptions.

Update affected source callers, tests, examples and any directly affected notebook cells without changing outputs or running long notebooks. Reconcile all affected model execution sections and current specifications. Preserve historical SDD outcomes; adjust references only if a referenced selector is retired. Update relevant project memory after verification, with its Spanish copy. No changes to the memory or SDD skill itself are authorized.

## Verification plan

Capture deterministic baseline histories and diagnostics before source edits for all built-in CPU methods, representative FC methods and BM4/RK4/adaptive JAX runs, including nonzero initial time, irregular output and energy tracking. Require exact before/after arrays and unchanged diagnostic values, allowing only explicitly documented metadata additions. Existing tests establish convergence, projection residuals, adaptive solver agreement, cross-backend accuracy and immutable snapshots.

Exercise structural method providers, executor ownership, callback rejection, unavailable devices, unsupported subclasses, repeat-run isolation, consumed failed runs, optional imports and mocked Modal round trips. Audit direct consumers, notebook syntax and shipped imports. Run mypy, the full headless test suite (including available JAX), supported examples, package build and SDD checker. Report unavailable GPU validation and do not claim a real Modal deployment test.
