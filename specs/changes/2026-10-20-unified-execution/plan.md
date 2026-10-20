+++
schema_version = 2
kind = "change-plan"
id = "2026-10-20-unified-execution"
revision = 1
spec_revision = 1
sources = ["src/execution/execution.py", "src/execution/_host.py", "src/execution/state.py", "src/methods/base.py", "src/methods/adaptive/scipy.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md"]
open_questions = []
+++

# Unified execution: implementation plan

## Design

Move IntegrationMethod and its immutable-preparation helpers into methods/base.py beside NumericalMethod. Keep the controller protocol and fixed controller with the host loop in execution/_host.py. Move current snapshots to execution/state.py, progress/count helpers to execution/_fixed.py and the compiled loop to execution/_jax_fixed.py. The executor calls these internal functions directly. Retain method-specific ScipyAdaptiveController beside the live solver adapter. Use type-only references and local imports at preparation/controller boundaries to avoid package-initialization cycles; do not add registries, compatibility modules or callback frameworks.

Expose IntegrationMethod from methods and controller/state extension types from execution. Internal consumers import defining modules. Remove integration files and the setuptools discovery entry. Retain simulate and Execution_Modal without another orchestration layer. Update supported tests and notebook cells if they contain retired imports. Preserve outputs and do not run long notebooks.

## Compatibility

Complete-job signatures, concrete method names and constructors, numerical algorithms, configured-method serialization, saved archives and result ownership remain unchanged. Extension imports move from integration.core to methods.base or execution._host; state imports move to execution.state and compiled internals to execution._jax_fixed. Direct low-level host invocation remains an internal driver, not an additional public execution API. Historical SDD prose retains past outcomes; update machine-checked source references with a migration note. Historical diagrams remain explicitly historical; current prose and theory paths are updated.

## Verification plan

Capture all 23 deterministic CPU/JAX cases from clean commit 87f85d4 before changes, including all CPU methods, representative GC/FC cases, energy and irregular output times. Compare complete diagnostics as well as times and values exactly. Test import orders in clean interpreters, optional-dependency isolation, custom controllers/providers, reusable methods, consumed runs, state pickling and mocked Modal round trips. Run mypy, all tests, both examples, isolated package build and a clean-wheel simulation. Inspect wheel contents for retired modules, audit notebook imports and validate current/historical SDD references and Spanish reading copies. Compile affected method theory PDFs and inspect changed pages. GPU or live Modal limitations remain explicit.
