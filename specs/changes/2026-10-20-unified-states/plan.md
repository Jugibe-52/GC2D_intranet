+++
schema_version = 2
kind = "change-plan"
revision = 2
spec_revision = 2
id = "2026-10-20-unified-states"
sources = ["src/states/history.py", "src/diagnostics/states/geometry.py", "src/states/generators.py", "src/formulations/layout.py", "src/contracts/formulations/layout.py", "src/problem.py"]
documents = ["specs/current/simulation.md", "specs/current/results.md", "docs/simulation/integration-architecture.md"]
open_questions = []
+++

# Unified states: implementation plan

## Design

Create a states package with State in states/state.py and existing layout implementations in states/layout.py. Keep GCState and FCState as named component tuples, not new storage subclasses. Keep the structural protocols in contracts/configuration.py; states must not import simulation, studies or diagnostics. Adapt that protocol documentation to explain the compatibility initial_state accessor.

Proposed API: State(values, *, layout), State.from_components(*components, layout=...), state.values, state.layout, state.split(), state.positions(), state.velocities(), state.particle_count. Layouts retain pack_components, split, as_blocks and from_blocks for arbitrary arrays. Use existing layout operations rather than duplicating algorithms. State is immutable through the supported API and captures its layout. For FC velocity access inspect the named split result; custom layouts without velocity components raise a clear error.

Place radial_star, ranked_radial_star, square and circle in states/generators.py, independent of the State class. Export these functions explicitly from states.generators; the public import is from states.generators import circle. Remove src/initial_conditions after migrating all supported consumers, without retaining a compatibility package. Extract calculate_area(state_array, *, layout, period=None) into diagnostics geometry support; migrate symplecticity observers and visualization consumers. Geometry labels needed by studies become study metadata rather than attributes on State. Preserve the existing signed-area arithmetic and periodic limitations.

## Compatibility

This is an explicit API migration: retire StateConfiguration, GCInitialConfiguration, FCInitialConfiguration and Area after migrating consumers to states.State and states.generators. Move canonical layout imports from contracts.state_layout to states; do not retain a parallel implementation or an alias-module tree. Preserve named component tuples and structural extension support. InitialValueProblem keeps its initial_configuration argument and Solution keeps its arrays and provenance semantics. Do not rename those unrelated public fields.

Inventory callers relying on empty configurations or set_initial_state and replace their construction sequence. Preserve legacy saved numeric data and archive schemas; inspect serializers for class-path references before removal and retain reading support where required. Migrate tracked supported notebooks and only directly affected local development callers, preserving outputs and user edits. Do not run long notebooks or change tracking. Update detailed docs for each affected model and current contracts only during implementation; historical SDD records retain their outcomes and resolve moved references with explanations.

## Verification plan

AC-001/002 require copy isolation, invalid shape/value handling, named component access, exact pack/split and block round-trips on GC/FC snapshots and histories, and backend-preserving JAX traced operations. AC-003 compares generated coordinates exactly to captured current outputs and area values with existing test tolerances, including periodic seams and orientation. AC-004 uses existing ownership and persistence tests and deterministic before/after GC/FC trajectories with identical inputs; no scientific reference solver is needed for a representation-only change. AC-005 requires consumer/import and documentation audits, archive round-trips, reduced notebook checks and repository checks. Record actual commands and outcomes during implementation, not during preparation.

Source references were updated for the 2026-10-20 states migration; prior recorded outcomes are unchanged.

Source locations were updated on 2026-10-22 after the contract ownership move;
the historical scope, acceptance and recorded checks above are unchanged.
