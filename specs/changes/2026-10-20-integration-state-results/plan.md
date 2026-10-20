+++
schema_version = 2
kind = "change-plan"
revision = 1
spec_revision = 1
id = "2026-10-20-integration-state-results"
sources = ["src/integration/core.py", "src/integration/state.py", "src/states/history.py", "src/formulations/layout.py", "src/request.py", "src/contracts/problem.py"]
documents = ["specs/current/states.md", "specs/current/results.md", "specs/current/simulation.md"]
open_questions = []
+++
# Integration states and results: implementation plan

## Design

Move request to src/request.py and physical layouts to src/formulations/layout.py. Prevent import cycles through type-only problem imports in formulations. Keep structural StateLayout independent of algorithms. InitialValueProblem accepts initial_state arrays and explicit or canonical layouts, snapshots both, and no longer retains a mutable configuration provider.

Create IntegrationState in integration/state.py; fixed control owns the live instance, adaptive control derives accepted snapshots from its solver, and numerical map signatures remain arrays. States in states/history.py absorbs recording, step metrics and result validation. PhysicalFormulation extracts each physical sample and optional momentum; final energy diagnostics are evaluated in one batch over saved physical data, preserving arithmetic. Retain bulk export for the compiled path. The coordinator reads as prepare, advance, sample, record, finalize; shadow samples never feed the next principal step. States supports construction from completed arrays for archive, custom executor and JAX results. Each run owns its result; final access is read-only and pickling remains supported.

## Compatibility

Migrate all current Python consumers and affected tracked/development notebook source cells while retaining outputs and tracking status. Do not keep State/Solution/IntegrationCollector aliases. Keep numerical archive schema and save_solution/load_solution entry points; they now serialize States. Source identity checks become snapshot equality. Maintain t/states as read-only result conveniences. Historical SDD outcomes remain unchanged; repair moved source references with a migration note. Update relevant model docs, current contracts, memory and Spanish companions, plus TFM content and root PDF.

## Verification plan

Capture deterministic GC/FC runs before edits with nonzero t0, off-grid samples, RK4, GC BM4Implicit and DOP853, energy diagnostics and archives. Require exact equality where numerical arithmetic is unchanged. Check recording isolation, incomplete finalization, immutable output, separate diagnostic axes, arbitrary t0, output-density independence and repeated-run isolation. Run CPU/JAX and archive coverage, complete unit tests, mypy, both examples, build and extracted-wheel imports. Audit notebook source syntax and execute reduced representative cells in memory. Validate SDD records and compile the thesis with latexmk -pdf main.tex.
