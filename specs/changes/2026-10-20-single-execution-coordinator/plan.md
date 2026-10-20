+++
schema_version = 2
kind = "change-plan"
id = "2026-10-20-single-execution-coordinator"
revision = 1
spec_revision = 1
sources = ["src/execution/execution.py", "src/execution/_host.py", "src/execution/_jax.py", "src/execution/_jax_fixed.py", "src/methods/_jax_dispatch.py", "src/states/history.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md", "docs/dynamics/jax-potential-evaluation.md"]
open_questions = []
+++

# Single execution coordinator: implementation plan

## Design

Execution.run resolves optional JAX preparation before creating a fresh numerical run, validates that run once, then owns running/finished transitions and one States.prepare/finalize pair. Internal host and compiled workers fill the supplied recorder and return numerical diagnostics. Introduce States.record_history for validated bulk loading into an empty recorder; preserve the existing streaming record API and avoid a second complete result container.

Merge execution/_jax_run.py and the hybrid adapter in dynamics/_jax_host.py into execution/_jax.py, retaining lazy JAX imports and bounded caches. Preparation supplies physical equations and the selected device; fixed numerical adapters remain in methods. Keep jax_mode as the extension contract, but remove the duplicate fixed-method whitelist and select exact built-in types in the adapter branches. Preserve explicit rejection of unsupported overrides and callback modes.

Move progress and shared step-count helpers from execution/_fixed.py into execution/_host.py. Retain FixedStepController and StepController and preserve their extension behavior. Remove integrate_method and integrate_fixed as independent complete coordinators; migrate tests to the complete-job boundary or the worker contract. Keep algorithms, sampling priority, adaptive solver lifetime and compiled arithmetic unchanged.

Preserve JaxPotential's independent class-selected evaluation, precision guards, tracing-safe caches and serialization. Execution continues automatic preparation with shared immutable samples/splines. Modal transports the same complete job and does not gain a separate numerical implementation. Update relevant current and model documentation plus Spanish copies. Historical records retain their outcomes while checked source paths are migrated.

## Compatibility

Public entry points, configured method constructors, NumericalMethod.jax_mode/new_run, controller extensions, saved schemas and optional dependencies remain stable. Retired private paths have no aliases. Bulk recording adds one operation to the existing recorder, with no new result class. Caller-owned observers and JAX precision configuration retain their ownership.

## Verification plan

Capture the 23-case deterministic baseline from clean commit 1a21217 before runtime edits, then compare exact times, states and all diagnostics. Cover fixed/adaptive CPU and JAX, GC/FC, energy and irregular samples. Test lifecycle failure and successful single finalization, custom controllers/providers, bulk ownership/rejection, unsupported overrides and standalone potential behavior. Run required mypy, complete unittest suite, examples and isolated package build; inspect a clean wheel and retired imports. Audit notebook code without executing long studies, validate SDD references and bilingual documents. GPU and live Modal are optional environmental limits and must be reported accurately.
