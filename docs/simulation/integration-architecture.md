# Integration and the four state formulations

All 13 public methods use `IntegrationMethod.new_run` and `integrate_method`.
Dynamics defines physics; formulation defines coordinates; method defines one
numerical step; integration controls accepted intervals and collects output.

The source packages follow that ownership: `src/contracts/` holds the shared
problem, request, state-layout, step and observation types;
`src/formulations/` defines physical and doubled state representations;
`src/methods/` implements numerical steps; and `src/integration/` controls
accepted steps and output collection. `src/simulation/runner.py` validates the
assembled run and returns the immutable `src/solution.py` result. Concrete
initial geometry stays in `src/initial_conditions/`. The `simulation` package
continues to export the established public names through explicit imports and
`__all__`. Internal consumers import directly from the defining modules.

## Public API and imports

Use `from simulation import ABBA4Implicit, BM4Implicit, simulate` for the public
execution API. `methods`, `methods.extended`, `formulations`, and `integration`
also declare their supported exports explicitly. `contracts` exposes types from
its individual submodules; its package initializer deliberately exports no names.
Every exported class or function has one implementation; public re-exports refer
to that same object. Internal modules do not import through the `simulation`
facade. Studies call `simulation.runner.simulate` directly.

The following old routes have been removed after migrating supported consumers:

| Removed route | Current owner |
|---|---|
| `simulation.methods.*` | `methods.*` |
| `simulation.formulations.*` | `formulations.*` |
| `simulation.integration`, `simulation._fixed` | `integration.core`, `integration._fixed` |
| `simulation.configuration`, `.problem`, `.request`, `.observation` | Corresponding `contracts` submodules |
| `simulation._result` | `contracts.result` |
| `simulation.solution` | `solution` |
| `methods.abba.*`, `methods.bm4.*` | `methods.extended` public exports and their defining modules |
| `methods._abba_coefficients` | `methods.extended.coefficients` |
| Full time/momentum projection modules | Removed; use spatial projection with `track_energy=True` for new studies |
| `methods.extended.order4_implicit_single_projection` | `methods.extended.abba_outer` numerical helpers |

No import hooks or `sys.modules` aliases recreate those paths. Deprecated
execution classes, method factories, and full-projection studies have also been
removed; see the [API migration guide](api-migration.md).
`tests/test_package_layout.py` checks the package boundaries and declared exports.

## Execution and state ownership

The `simulate` function owns the stateless execution entry point. `Solution` validates and
copies the physical arrays, including their dimensions, finite values and
packed layout. The entry point then checks that the saved times, state size and
initial state agree with the request and problem. Each check has one owner.

Gauss--Legendre and SDIRK share field validation, analytic particle Jacobians,
centered finite differences and Jacobian selection in
`src/methods/classical/_jacobians.py`. Their stage equations and nonlinear solves
remain defined by each method. GC and FC split maps share the optional momentum
update in `src/formulations/base.py`.

| Formulation | One planar particle | N planar particles |
|---|---:|---:|
| Physical | 2 | 2N |
| Physical with energy | 4 | 4N |
| Spatially duplicated | 4 | 4N |
| Spatially duplicated with energy | 6 | 6N |

Use `track_energy=True` on any public method. Classical methods use
`PhysicalFormulation`; ABBA and BM4 use `DoubledFormulation`. These concrete
objects live in `src/formulations/state.py`. Energy variants append one
time entry and one normalized momentum per particle. Projected methods
store diagonal accepted copies and project spatial variables only. Every
component block has N entries, and `components(state)` exposes a uniform
`(coordinates, N, *sample_axes)` view. The time entries repeat the scalar
integration time. `extended_time` has shape `(N, saved_times)`; `Solution.t`
remains the one-dimensional grid.

`StepResult` and `StepInfo` live in `src/contracts/step.py`. Their states are
internal; `Solution.states` and observer states are physical.
`StepResult.statistics` contains accepted physical work. Details are transient
unless an observer explicitly retains a copied event. `export_history` delegates
to the formulation's common physical and energy extraction.

Tracking does not affect physical stages, Newton/Broyden scales or adaptive error
control. DOP853 and Radau own physical-only SciPy solvers and use eight-point
Gauss quadrature over accepted dense output for passive momentum. This quadrature
has no independent error tolerance; audit it by refinement. Radau's optional
Jacobian always differentiates physical coordinates only. Diagnostic interpolation
work and energy-derivative evaluations are excluded from physical work counters.
BM4 reuses the converged residual's shear states for passive quadrature without
repeating spatial maps. Adaptive energy samples reuse their accepted interval's
endpoint momenta; only interior queries require partial-interval quadrature.

Fixed controllers compute off-grid samples with independent shortened maps;
adaptive controllers evaluate accepted dense output. The formulation validates each
particle time and aligns diagnostic times to the requested grid within round-off.
The coordinator and collector do not inspect clock indices or tracking flags.
Observer snapshots cannot mutate
samples or counters. Repeated runs own fresh solver resources.

The former `fully_extended` selector is rejected. It described spatial, time and
momentum projection in eight duplicated coordinates and is not an energy-monitoring
option. Existing historical full-state diagnostics are readers of that former
record type, not an active simulation path. Canonical model guides and theory
PDFs document the current methods; older rendered diagrams are historical.
