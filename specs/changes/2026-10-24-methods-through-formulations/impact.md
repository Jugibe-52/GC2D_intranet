# Affected parts

| Area | Change |
| --- | --- |
| `methods/base.py`, classical, HBVM and adaptive methods | Consume functions and initial data; remove raw dynamics/problem resources, single-copy formulation, mirrored nonlinear controls and passive energy state. Preserve numerical equations and temporal control. |
| `formulations/state.py`, exports and workspace consumers | Remove `SingleFormulation`; make `DoubleFormulation` independent and specific to two-copy work. Reuse ordinary array checks without a replacement wrapper. |
| Extended maps, projection helpers and FC splitting | Supply functions/flows explicitly, retain shared compositions and projection equations, and expose optional quadrature data. Move physical flow binding outside methods. |
| `solvers/`, derivative helpers and compiled solver options | Own nonlinear settings and iteration observation once; consolidate validation, differentiation and block packing. Preserve per-equation norms and specialized corrections. |
| Observation contracts, `diagnostics/methods/` and `diagnostics/solvers/` | Own energy accumulation and algebraic observation under their actual processes. Reuse current observers and events; add only the data needed for exact accepted-stage quadrature. |
| `execution/`, JAX kernels, `states/` and public simulation entry points | Bind functions and observers, coordinate separate physical/diagnostic state and sampling, and record unchanged scientific output. Simplify step/compiled-step contracts to remove energy operations. |
| Studies, examples, notebooks and worker payloads | Migrate constructor options, direct preparation and observer registration; verify serialization and existing scientific consumers. |

## Intentional API migration

Retain method/solver class names, `DoubleFormulation`, physical coordinate types,
map helper names and the ordinary three-argument simulation call. Remove
`SingleFormulation` and the old dynamics-taking `new_run` signature. Document
map preparation changes and removal of integrated map-momentum packing; preserve
spatial map behavior and reproduce its energy through the shared observer path.

Replace method-level `track_energy=True` with `EnergyObserver()` registration,
move step callbacks to execution and Newton iteration callbacks to the solver.
Move `newton_absolute_tolerance`, `newton_relative_tolerance`,
`newton_max_iterations`, `newton_jacobian_method`,
`newton_jacobian_relative_step`, `nonlinear_solver` and equivalent HBVM algebraic
fields into the solver. Do not remove identically named adaptive integration
controls. Migrate effective configurations, including explicit-solver precedence
and method defaults, then delete the old argument paths instead of maintaining
both interfaces. Existing configured-object pickles may need reconstruction.

Keep numerical archives, energy labels and saved-state layouts readable without
migration or recomputation. Preserve observer source checks and reject unsupported
callback/backend combinations explicitly. Repeated runs and worker invocations
must start with independent built-in observer and solver work state; explicitly
supplied callbacks keep their caller-owned lifecycle.

## Validation and documentation

Cover function-only method use, workspace ownership, field/map/residual
derivatives, solver defaults and failures, HBVM evaluation order, energy and
observer sampling, compiled/hybrid execution, package boundaries and persistence.
Extend existing relevant tests rather than reproducing implementation details.

During execution update the simulation/component/execution/JAX/API guides,
solver and diagnostic contracts, affected per-model architecture/theory artifacts,
applicable current specifications, project guidance and relevant memory. Audit
tracked experiments/SymPy and ignored development notebooks; migrate actual
consumers and run reduced subsets without saving reduced outputs. Preserve
historical change records, valid user edits and Git tracking policy. Maintain
Spanish Markdown copies. Preparation edits only this four-document set and its
translations, with no results document or implementation changes.
