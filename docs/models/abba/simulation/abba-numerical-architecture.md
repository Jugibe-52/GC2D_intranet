# Canonical ABBA numerical architecture

The implicit ABBA family now has one prepared execution path. ABBA2, ABBA4 and
ABBA6 share the coordinator, state operations, convergence drivers, numerical
records and observation adapters. Their step recipes retain the mathematical
distinction between composing projected maps and projecting a composition.

Read the current model diagrams horizontally for the complete lifecycle and
vertically for each phase's operations and records:

- [ABBA2 complete execution](../../abba2-implicit/simulation/abba2-implicit-simulation-architecture.puml)
  ([PNG](../../abba2-implicit/simulation/abba2-implicit-simulation-architecture.png)).
- [ABBA4 complete execution](../../abba4-implicit/simulation/abba4-implicit-simulation-architecture.puml)
  ([PNG](../../abba4-implicit/simulation/abba4-implicit-simulation-architecture.png)).
- [ABBA6 signed composition](../../abba6-implicit/simulation/abba6-implicit-simulation-architecture.md).
- [ABBA2 midpoint](../../abba2-midpoint/simulation/abba2-midpoint-simulation-architecture.md).

The older `abba-numerical-architecture.puml` is retained as a historical
inventory of the coordinators before this refactor. The model diagrams above
describe the implemented runtime; the `proposed-*` files retain the design
that preceded implementation.

## Reading the implementation

```text
simulate(problem, method, request)
  -> SimulationRunner
  -> method.integrate(problem, request)
  -> prepare_abba(...): PreparedABBA
  -> integrate_abba(prepared, request)
       -> integrate_fixed_grid(..., advance)
            -> state_ops.unpack(t, workspace)
            -> solve_step(t, state, h): tuple[ProjectedMapResult, ...]
            -> state_ops.finish_step(t, h, workspace, projections)
            -> main step: StepResult -> metrics -> optional event
       -> state_ops.extract(output_times, history)
       -> IntegrationData
  -> SimulationRunner validates and constructs Solution
```

Start at a public method, then read `preparation.py`, `runtime.py` and the
selected recipe in `steps.py`. Enter the projection and map modules when the
numerical equations or exact derivative calculations are needed.

| File under `src/simulation/methods/abba/` | Responsibility |
|---|---|
| `order2_implicit.py`, `order4_implicit.py`, `order6_implicit.py` | Public configuration and entry to preparation/runtime |
| `_implicit.py`, `_configuration.py` | Shared option validation and state-dimension metadata |
| `preparation.py` | Validate capabilities, choose the step recipe and bind `PreparedABBA` |
| `runtime.py` | One main/shadow advance adapter and final result assembly |
| `steps.py` | One projected map, a composition of projected maps, or one outer projection |
| `state.py`, `_energy.py` | Bound workspace operations and physical/extended energy handling |
| `records.py` | `ProjectedMapResult`, `StepResult`, typed traces and observer-independent metrics |
| `observations.py` | Optional adapters to the existing public event classes |
| `projection_reduced.py`, `projection_simultaneous.py` | Physical single-map equations and specialized analytic corrections |
| `projection_outer.py` | Physical equations around the complete ABBA4 base composition |
| `projection_extended.py` | Full-diagonal equations, accepted full-map data and implicit tangents |
| `maps/physical.py`, `maps/extended.py` | Unprojected stages and their exact derivatives |
| `../_nonlinear.py` | `SolverOptions`, `SolveStats`, shared Newton and Broyden drivers |
| `../../_fixed.py` | Uniform main grid and independent shadow samples |

The public `InitialValueProblem`, `SimulationRequest`, `simulate` and
`Solution` contracts are unchanged. Dynamics, potential evaluation and initial
state layout remain outside the method package. Their shared contracts are
documented in [the dynamics guide](../../../dynamics/gc2d-h5-import.md).

## Configuration and method identity

The four public classes still define 51 normalized configurations:

| Method | State/energy strategies | Formulations | Solvers | Placements | Total |
|---|---:|---:|---:|---:|---:|
| `ABBA2Midpoint` | 3 | Arithmetic mean | None | One | 3 |
| `ABBA2Implicit` | 3 | 2 | 2 | One | 12 |
| `ABBA4Implicit` | 3 | 2 | 2 | 2 | 24 |
| `ABBA6Implicit` | 3 | 2 | 2 | One | 12 |

The three state/energy strategies are `(physical, False)`,
`(physical, True)` and `(fully_extended, True)`. Full extension always resolves
`track_energy` to `True`. Implicit formulations are `reduced_multiplier` and
`simultaneous_state_multiplier`; nonlinear solvers are `newton` and `broyden`.
Defaults, constructor argument order and tolerance option names remain intact.

| Recipe | Base maps per complete step | Accepted projections |
|---|---:|---:|
| ABBA2: `solve_single_map_step` | 1 | 1 |
| ABBA4: `solve_projected_composition_step` | 3 | 3 |
| ABBA4: `solve_outer_projection_step` | 3 | 1 |
| ABBA6: `solve_projected_composition_step` | 7 | 7 |

ABBA4 selects `after_each_abba_map` by default or
`around_complete_composition` explicitly. Every signed map uses its actual
start time. Negative composition coefficients retain backward substeps.
One exterior projection encloses continuous duplicated copies through all
three base maps; it is a different numerical map from three projected solves.

## Prepared operations and numerical records

`PreparedABBA` stores the initial workspace, bound `solve_step`, `StatePolicy`,
optional event builder, solver options, metadata and recording options. Its
configuration record is frozen; the initial workspace and coefficient metadata
are read-only copies. Preparation checks planar physical dynamics, one-particle
full extension, energy capabilities and Newton derivatives.

`StatePolicy` contains four callables:

1. `initialize(z0, t0)` builds the internal workspace.
2. `unpack(t, workspace)` exposes the numerical state.
3. `finish_step(t, h, workspace, projections)` closes state and energy.
4. `extract(times, history)` returns physical output and energy diagnostics.

For `N` physical particles, the numerical state has `2N` components. Optional
energy tracking appends `N` auxiliary momenta `kappa=k/2` to the workspace.
It uses accepted stage traces and never changes the physical solve. The full
branch evolves one `(x,y,t,k)` state, duplicates it into eight components,
and synchronizes time at outer boundaries and requested output times.

`ProjectedMapResult` describes exactly one accepted projection: input and output
states, signed interval, multiplier, `SolveStats` and a typed trace.
`PhysicalProjectionTrace` contains one or three `PhysicalBaseMapTrace` records.
`ExtendedProjectionTrace` retains the complete selected full base-map result
and coefficients; an exterior composition is still one projection record.
`StepResult` holds the next workspace and the ordered tuple of projections.

Internal arrays are numerical work data. Public `Solution` arrays and observer
snapshots are copied at their respective boundaries.

## Shared convergence, specialized mathematics

`_solve_newton` owns residual validation, convergence decisions, iteration
limits and counters. A formulation supplies its residual and update callback;
the callback owns analytic Jacobian assembly, particle packing and the exact
correction algebra. `_solve_broyden` retains its residual-only secant update.
The drivers return `_NonlinearResult`; the step adapters create `SolveStats`.
A converged initial guess takes zero corrections and one residual evaluation.

| State | Accepted numerical state | Base state | Reduced unknown | Simultaneous unknown |
|---|---:|---:|---:|---:|
| Physical, with or without tracking | `2N` | `4N` | `2N` | `6N` |
| Fully extended, one particle | 4 | 8 | 4 | 12 |

Newton derivatives remain exact and formulation-specific. Broyden does not
request analytic derivatives during its nonlinear iterations. Optional full
observation may evaluate tangents after convergence, with the existing finite
difference fallback when analytic derivatives are unavailable.

## Sampling, diagnostics and observers

The existing fixed-grid scheduler controls time. Main steps advance the
accepted trajectory and append metrics. An off-grid requested sample uses an
independent advance from the preceding main node. It uses the same numerical
recipe and error checks, without changing main states, metrics or events.

`record_completed_step` reads numerical records directly; it never builds an
observer event. `StepMetrics` sums corrections/evaluations, selects residual
and tolerance from the same worst normalized solve, and records the maximum
multiplier norm. Substep columns count projections, so outer ABBA4 has one
column even though its trace contains three base maps.

`observations.py` constructs copied method-specific events only when an
observer is installed and the scheduler accepts a main step. Existing physical
ABBA2, composed ABBA4/ABBA6, exterior ABBA4 and fully extended event contracts
remain available. Shadow steps emit none.

Diagnostic keys and shapes remain compatible, including optional energy
histories. The legacy `newton_*` metric aliases are applied at the final
`IntegrationData` boundary. Study result labels such as
`ABBA4ImplicitSingleProjection` and persisted CSV fields stay unchanged.

## Midpoint and compatibility

`ABBA2Midpoint` uses arithmetic projection and retains its own coordinator.
It shares `maps/physical.py`, `maps/extended.py` and energy helpers. Its full
coordinator now lives in `midpoint_extended.py`.

The old private map/projection modules and `methods/_fully_extended.py` retain
compatibility imports. `composition.py` adapts the old private composition
entry points to the shared recipes for existing numerical tests and callers.
`ABBA4ImplicitSingleProjection(...)` remains a deprecated factory for
`ABBA4Implicit(projection_placement="around_complete_composition", ...)`.
New internal code uses the canonical modules listed above.

## Validation of the refactor

Before changing numerical code, 96 short reference runs covered ABBA2, both
ABBA4 placements and ABBA6, both formulations, Newton/Broyden, all state/energy
strategies, observers and off-grid samples. After the refactor, trajectory
arrays, diagnostics, event data, event map evaluations and full tangents
matched those references exactly.

`tests/test_abba_shared_runtime.py` checks record cardinality, shadow-sample
isolation, observer independence and the common Newton driver's counters and
failure limit. Existing model, full-state and Broyden tests cover the numerical
contracts. Development notebook API validation executes selected cells with
two-step studies and temporary outputs; canonical studies are not run in full.

The completed validation passed 114 focused tests: 101 ABBA tests, four
full-state tests, and nine Broyden/package-layout tests. Syntax checks passed
for all 20 development notebooks (167 code cells). The three notebooks that
use the affected APIs passed selected-cell checks, including short study calls,
diagnostic assertions, temporary CSV persistence and representative plots.
Their original scientific parameters, saved outputs, logs and CSV remain intact.

Type checking passed for all 32 files in the changed-module check. The global
check still reports 19 existing errors in five unchanged diagnostics, studies
and visualization files; those files match the pre-refactor Git revision.
