# ABBA2 implicit simulation architecture

The [complete execution diagram](abba2-implicit-simulation-architecture.puml)
([PNG](abba2-implicit-simulation-architecture.png)) follows the implemented
runtime. Read its six phases horizontally, then each phase vertically.
The [family architecture](../../abba/simulation/abba-numerical-architecture.md)
defines the shared records, module responsibilities and full configuration matrix.

## One public method, one projected map

`ABBA2Implicit` preserves its public options: `projection_formulation`,
`state_extension`, `track_energy`, `nonlinear_solver`, Newton tolerance names,
iteration limit, progress and observer. Two formulations, two solvers and
three normalized state/energy strategies still yield twelve configurations.

```text
simulate -> SimulationRunner -> ABBA2Implicit.integrate
  -> prepare_abba(..., order=2)
  -> integrate_abba -> integrate_fixed_grid -> advance
       -> StatePolicy.unpack
       -> solve_single_map_step -> one projected map
       -> StatePolicy.finish_step
       -> main step: StepResult -> metrics -> optional event
  -> StatePolicy.extract -> IntegrationData -> SimulationRunner -> Solution
```

The public method only prepares and executes. It no longer dispatches to
separate physical and fully extended integration coordinators. `PreparedABBA`,
`integrate_abba`, `StatePolicy` and the accepted-step records are the same
components used by ABBA4 and ABBA6.

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
`ExtendedProjectionTrace`. `record_completed_step` collects numerical work
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

The method's mathematical definitions and derivations remain in
[the model theory](../tex/theory.tex). The old `proposed-full-execution`
diagram is retained as the design proposal preceding this implementation.
