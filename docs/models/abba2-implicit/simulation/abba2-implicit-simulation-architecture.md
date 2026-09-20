# ABBA2 implicit simulation architecture

[Editable source](abba2-implicit-simulation-architecture.puml) · [Scalable diagram (SVG)](abba2-implicit-simulation-architecture.svg)

The diagram retains the detailed six-phase layout: physical inputs, run
definition, preparation, numerical execution, accepted-step records, and final
result. Read phases horizontally and operations vertically; the cards show
module paths, inputs/outputs and the selected numerical recipe.

The [complete execution diagram](abba2-implicit-simulation-architecture.puml)
([PNG](abba2-implicit-simulation-architecture.png)) follows the implemented
runtime. Follow the preparation, shared control, model step and output branches.
The [family architecture](../../abba/simulation/abba-numerical-architecture.md)
defines the shared records, module responsibilities and full configuration matrix.

## Method instances and integration lifecycle

This method inherits `IntegrationMethod.integrate(problem, request)`, shared by
all 13 public methods. It calls `new_run(problem, request)` to create a fresh
instance of the same numerical class. That instance's `initialize` validates
capabilities and sets its formulation, initial internal state and metadata.
There is no separate context or callback-based method record.

| Operation on the numerical class | Responsibility |
|---|---|
| `initialize(problem, request)` | Initialize this run's resources once; return `None` |
| `advance(t, state, h)` | Execute numerical work and return state, statistics and typed details |
| `build_observation(info, step)` | Construct a method-specific event with independent snapshots |
| `export_history(times, history)` | Extract physical output and auxiliary diagnostics |
| `controller()` | Select fixed or adaptive accepted-step scheduling |

`integrate_method(run)` owns the common loop. Its `IntegrationCollector` saves
requested samples and copied accepted-step metric rows, without retaining
numerical details or events. The method owns the formulation and any live solver.
Analysis and persistence remain in `diagnostics/`; observers remain caller-owned.

Constructor options remain reusable through `simulate`. Per-run resources are
excluded from reconstruction, and initial state and metadata are isolated as
read-only copies. A completed or failed run cannot be integrated again; create a
fresh run. Call `new_run` for low-level access rather than resetting `initialize`.

`FixedStepController` calls `run.advance` with the exact effective duration and
uses independent shortened maps for interior output times. DOP853/Radau's
controller calls their ordinary `advance` on the live solver with an upper step
bound and reads the actual accepted endpoint. Dense sampling retains the backend.

Metric rows follow `step_times`; physical and auxiliary histories follow
`Solution.t`. Common fields are `step_count`, `step_start_times`, `step_times`,
`step_sizes` and `output_interpolation_count`. Shadow work and extra observer-only
work are excluded from accepted numerical counters.

See the [generic architecture](../../../simulation/integration-architecture.md)
for the lifecycle, adaptive semantics and extension guide. Executable contracts
are in `tests/test_method_integration.py` and `tests/test_adaptive_integration.py`.

## One public method, one projected map

`ABBA2Implicit` preserves its public options: `projection_formulation`,
`state_extension`, `track_energy`, `nonlinear_solver`, Newton tolerance names,
iteration limit, progress and observer. Two formulations, two solvers and
three normalized state/energy strategies still yield twelve configurations.

```text
simulate -> SimulationRunner -> ABBA2Implicit.integrate
  -> new_run -> _ABBAImplicitMethod.initialize
  -> integrate_method(run) -> FixedStepController -> run.advance
       -> StatePolicy.unpack
       -> solve_single_map_step -> one projected map
       -> StatePolicy.finish_step
       -> main step: StepResult -> metrics -> optional event
  -> StatePolicy.extract -> IntegrationData -> SimulationRunner -> Solution
```

The public class inherits initialization, advance, observation and export from
`_ABBAImplicitMethod`. `StatePolicy`, projection kernels and accepted-step records
are shared with ABBA4 and ABBA6. The common driver owns the time loop.

## Numerical work

`solve_single_map_step` returns a one-element tuple of `ProjectedMapResult`.
The physical reduced and simultaneous equations live in
`projection_reduced.py` and `projection_simultaneous.py`. They share the
unprojected endpoint-time A--B--B--A stages and exact stage derivatives from
`maps/physical.py`. Full-diagonal equations live in `projection_extended.py`
and use `maps/extended.py`.

The reduced residual is the mapped copy separation plus twice the multiplier.
The simultaneous residual includes both output-map defects and the diagonal
constraint. They retain their specialized analytic Jacobians and correction
packing. `_solve_newton` and `_solve_broyden` in `methods/_nonlinear.py` own
iteration control. The nonlinear result is adapted into solver-neutral
`SolveStats` and the accepted map trace.

| State strategy | Numerical state | Duplicated base state | Reduced unknown | Simultaneous unknown |
|---|---:|---:|---:|---:|
| Physical | `2N` | `4N` | `2N` | `6N` |
| Physical with energy | `2N` | `4N` | `2N` | `6N` |
| Fully extended, one particle | 4 | 8 | 4 | 12 |

Physical tracking transports one `kappa=k/2` per particle from the accepted
stage trace. It never feeds back into the projected physical state. Full
extension evolves `(x,y,t,k)` intrinsically, always enables energy diagnostics,
and synchronizes the accepted time at the outer boundary. Public trajectories
contain physical coordinates; extra energy/state histories remain diagnostics.

## Records, sampling and observations

Every accepted main step creates `StepResult(next_workspace, projections)`.
Its one projection contains `SolveStats` and a `PhysicalProjectionTrace` or
`ExtendedProjectionTrace`. `step_statistics` collects numerical work
directly, regardless of whether an observer exists.

An installed observer receives `ABBA2ImplicitIntegrationStep` for physical
execution or `FullyExtendedImplicitIntegrationStep` for full execution.
`observations.py` copies snapshots and constructs any required full tangent
only when building the event. Optional physical energy remains outside the
observed physical map.

Off-grid output times use shadow advances through the same bound functions.
They do not change main states, diagnostic rows or observer events. A failed
shadow solve still raises an error. `IntegrationData` preserves the existing
diagnostic keys and legacy Newton aliases before the runner builds `Solution`.

## Principal files

| File under `src/simulation/methods/abba/` | Responsibility |
|---|---|
| `order2_implicit.py`, `order4_implicit.py`, `order6_implicit.py` | Concrete order, constructor controls and inherited numerical operations |
| `_implicit.py`, `_configuration.py` | Shared option validation and state-dimension metadata |
| `_implicit.py` | Shared initialization, numerical advance, observation and history export |
| `steps.py` | One projected map, a composition of projected maps, or one outer projection |
| `state.py`, `_energy.py` | Bound workspace operations and physical/extended energy handling |
| `records.py` | `ProjectedMapResult`, `StepResult`, typed traces and observer-independent metrics |
| `observations.py` | Optional adapters to the existing public event classes |
| `projection_reduced.py`, `projection_simultaneous.py` | Physical single-map equations and specialized analytic corrections |
| `projection_outer.py` | Physical equations around the complete ABBA4 base composition |
| `projection_extended.py` | Full-diagonal equations, accepted full-map data and implicit tangents |
| `maps/physical.py`, `maps/extended.py` | Unprojected stages and their exact derivatives |
| `../_nonlinear.py` | `SolverOptions`, `SolveStats`, shared Newton and Broyden drivers |
| `../../integration.py` | Method lifecycle, controller, collection and common coordinator |

The method's mathematical definitions and derivations remain in
[the model theory](../tex/theory.tex). The old `proposed-full-execution`
diagram is retained as the design proposal preceding this implementation.
