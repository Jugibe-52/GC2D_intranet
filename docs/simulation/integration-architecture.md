# Integration, physical states and temporary workspaces

All 13 public methods use executor-owned preparation through `IntegrationMethod.new_run`.
Execution selects the host or device driver. Dynamics defines physics and coordinate roles; formulation
prepares temporary copies; the method defines numerical operations; execution controls
accepted intervals and collects output.

`problem.py` owns physical initial-value problems; `request.py` owns the interval and output times; `contracts/` owns shared interfaces, step records and array compatibility. `dynamics.coordinates` owns physical structural descriptions, while reusable formulations prepare temporary copies. `methods/` implements numerical steps. `execution/` owns preparation, accepted `IntegrationState` snapshots, sampling and host/device loops; `methods.base` owns numerical preparation. `states/` owns the physical recording container and completed result. `simulation/runner.py` validates that result against the inputs and returns the same object. Internal consumers import directly from each defining module.

Validation follows the same ownership: `InitialValueProblem` checks the dynamics
protocol and compatible initial layout; methods check the Hamiltonian
capability needed for energy tracking and their numerical
options. Methods reuse these validated inputs instead of repeating the same
capability checks. `IntegrationMethod.new_run` owns immutable run snapshots.

Long validation blocks are organized as private preparation functions within
their owning modules. `SimulationRequest` separates interval validation from
output-schedule normalization, and reuses the interval rule in `uniform`.
`States` prepares the physical history and immutable diagnostics separately;
the runner retains responsibility for agreement with the requested initial state
and saved times. Run preparation similarly separates initial-state ownership
and metadata freezing from the method lifecycle.

`InitialConditions` prepares its owned array, layout and particle count in one
private function. `States` shares complete-history checks between construction
and bulk recording without changing dtype; diagnostic preparation validates all
entries before copying arrays. GC/FC parameter preparation retains signed B/eta
and each model's radius constraints. Helpers return values rather than mutating
the object being constructed.

Scalar option rules shared by numerical families live in `methods._validation`.
Callers retain their existing conversion policies, including whether Boolean
values are rejected before conversion to floating point. Newton and Broyden
separate validated entry data from their iterations and reuse residual checks
without repeating a cached evaluation. Checks on convergence, singular systems,
accepted-step continuity, and newly computed non-finite values remain next to
the numerical operation that produces them. These validation helpers do not alter the numerical algorithms.

Gauss--Legendre and SDIRK share constructor normalization in
`methods.classical._configuration`: supplied Newton solvers take precedence,
Boolean tolerances remain accepted, and Jacobian scale is checked before the
iteration count. Other method families retain their own policies.

The downstream packages follow the same ownership convention. Studies share
scalar and sampling rules in `studies._validation` and aligned comparison
records in `studies._comparison_validation`; scientific identity, exact versus
tolerant time alignment, and method-specific diagnostics remain local to each
study family. Diagnostics prepare typed observation records before analysis and
keep sequence and sampling checks at their original state transitions. File
readers retain their format-specific integrity checks. Visualization prepares
validated series before creating figures and shares planar-solution and
animation controls through private package modules, preserving each animator's
frame-count policy. The separately packaged Poincare batch runtime keeps its
validation helpers within the files shipped to workers.

Common comparison preparation separates method-table alignment, per-run records
and sampling/reference constraints. Poincare configuration keeps its stricter
numeric and Python-integer policies in local helpers. Plot preparation includes
every series, even a later one after a valid entry, before creating a figure;
invalid energy, symplecticity, iteration and accuracy inputs therefore leave
existing figures untouched. `tests/test_plot_input_preparation.py` checks this
boundary and the curves produced for valid data.

## Package workflow diagram

![Package responsibilities and execution workflow](package-workflow-architecture.svg)

Open the [SVG](package-workflow-architecture.svg),
[PNG](package-workflow-architecture.png), or
[PlantUML/Graphviz source](package-workflow-architecture.puml).
The previous detailed integration lifecycle is preserved as
[`integration-architecture_old.puml`](integration-architecture_old.puml),
with its [SVG](integration-architecture_old.svg) and
[PNG](integration-architecture_old.png) renderings.

The diagram is a historical rendering, retained without regeneration. Its collector/result split and old module routes have been superseded. The current flow is:

1. Build `InitialValueProblem` from dynamics, a physical array and its layout. Set t0 and output times in `SimulationRequest`.
2. The executor prepares the environment and asks the method for a fresh run and formulation. The controller initializes `IntegrationState(t0, physical_values, momentum)`; `States.prepare(...)` reserves physical output columns.
3. Advance one main interval and retain both accepted endpoint snapshots. Record accepted-step statistics on their own axis.
4. For each requested time in that interval, reuse an endpoint or calculate an independent shadow/dense sample. The sample snapshot supplies physical values and optional momentum; `States.record` copies them.
5. Evaluate energy diagnostics over the saved physical history and auxiliary data. `States.finalize` checks completeness and seals this same object.
6. `Execution.run` returns it to `simulate`, which checks request and initial-data agreement and returns it directly.

The initial state and a saved sample can have identical values while serving different roles. No sample becomes the next main state. Formulations are shared across methods; layout, time control and numerical stepping remain separate responsibilities. JAX retains functional device arrays inside compiled loops and records its exported physical history with `States.record_history`. `Execution.run` owns the single allocation and finalization boundary for both workers.

## Public API and imports

Use `from simulation import ABBA4Implicit, BM4Implicit, simulate` for the public
execution API. `methods`, `methods.extended`, `formulations`, and `execution`
also declare their supported exports explicitly. `contracts` exposes types from
its individual submodules; its package initializer deliberately exports no names.
Every exported class or function has one implementation; public re-exports refer
to that same object. Internal modules do not import through the `simulation`
facade. Studies call `simulation.runner.simulate` directly.

The following old routes have been removed after migrating supported consumers:

| Removed route | Current owner |
|---|---|
| `contracts.execution.Execution` | `execution.Execution` for the executor; `execution.options.ExecutionOptions` for backend choices |
| `simulation.methods.*` | `methods.*` |
| `simulation.formulations.*` | `formulations.*` |
| `simulation.integration`, `simulation._fixed` | `methods.base`, `execution._host` |
| `simulation.configuration`, `.observation` | Corresponding `contracts` submodules |
| `simulation.problem`, `contracts.problem` | `problem` |
| `contracts.execution_options` | `execution.options` |
| `simulation.request`, `contracts.request` | `request` |
| `simulation._result` | `contracts.states.history` |
| `simulation.solution`, `solution` | `states.history` |
| `methods.abba.*`, `methods.bm4.*` | `methods.extended` public exports and their defining modules |
| `methods._abba_coefficients` | `methods.extended.core.composition` |
| Full time/momentum projection modules | Removed; use spatial projection with `track_energy=True` for new studies |
| `methods.extended.order4_implicit_single_projection` | `methods.extended.abba.ABBA4Implicit`; shared projection in `methods.extended.core.projection` |

No import hooks or `sys.modules` aliases recreate those paths. Deprecated
execution classes, method factories, and full-projection studies have also been
removed; see the [API migration guide](api-migration.md).
`tests/test_package_layout.py` checks the package boundaries and declared exports.

## Execution and state ownership

The `simulate` function validates its inputs and delegates each complete job to
`execution.execution.Execution`. The local executor prepares a fresh run through
`new_run`, allocates one recorder, invokes the host or compiled worker and finalizes that same result. Subclasses may perform the whole job
elsewhere. See the [executor contract](execution.md). `States` validates and copies
physical arrays, including dimensions, finite values and packed layout. The entry
point checks saved times, state size and initial data against the request and
problem and returns the same completed object.

`simulate(..., options=ExecutionOptions(...))` optionally selects JAX for every
built-in method. The executor resolves the device, prepares field evaluation and
creates a fresh run. Fixed methods expose a device map used by the common compiled
driver; DOP853/Radau receive prepared field/energy callbacks and retain SciPy control.
Both return NumPy `States`. Omitting options preserves the default CPU controller.
See the [JAX execution contract](jax-execution.md) for scope and limitations.

Gauss--Legendre and SDIRK share field validation, analytic particle Jacobians,
centered finite differences and Jacobian selection in
`src/methods/classical/_jacobians.py`. Their stage equations and nonlinear solves
remain defined by each method. GC and FC split maps share the optional momentum
update in `src/methods/_momentum.py`.

`SingleFormulation(physical_size)` and `DoubleFormulation(physical_size)` retain only a positive physical vector size. They prepare one or two temporary copies through `lift`, `join_copies` and `split_copies`. Shapes are `(d*N,*samples)` and `(copies*d*N,*samples)`; shifted copies remain distinct until the method projects them. NumPy/JAX backends and complete FC velocity blocks are preserved. There is no layout, particle count, tracking flag, clock, momentum, initial state, projection or solver in either representation.

Every accepted snapshot is `IntegrationState(time, values, momentum=None)`: scalar time, physical `values` of shape `(d*N,)`, and optional physical `momentum` of shape `(N,)`. GC remains 2N and FC 4N with or without tracking. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. This generic support does not extend GC numerical methods to FC.

The method initializes zero momentum, prepares copies, selects projection equations and solver, and integrates momentum from accepted stages. `advance(t, Z, h, momentum=kappa)` returns physical `StepResult.state` and separate `StepResult.momentum`. `StepInfo.momentum_before` records the incoming auxiliary. Shadow samples start from the same accepted physical state and momentum. JAX carries separate arrays through the compiled loop.

Methods evaluate saved physical H once. `states.energy.energy_diagnostics` combines supplied H and physical kappa without dynamics evaluation. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. Numerical archives require no migration; old pickled configured objects should be recreated.

Tracking does not affect physical stages, Newton/Broyden scales or adaptive error
control. DOP853 and Radau own physical-only SciPy solvers and use eight-point
Gauss quadrature over accepted dense output for passive momentum. This quadrature
has no independent error tolerance; audit it by refinement. Radau's optional
Jacobian always differentiates physical coordinates only. Diagnostic interpolation
work and energy-derivative evaluations are excluded from physical work counters.
BM4 reuses the converged residual's shear states for passive quadrature without
repeating spatial maps. Adaptive energy samples reuse their accepted interval's
endpoint momenta; only interior queries require partial-interval quadrature.

Fixed controllers reuse accepted endpoint states and compute off-grid samples with
independent shortened maps; adaptive controllers evaluate accepted dense output.
Diagnostic times come directly from each scalar sample time.
The coordinator and `States` do not inspect clock indices or tracking flags.
Observer snapshots cannot mutate
samples or counters. Repeated runs own fresh solver resources.

The former `fully_extended` selector is rejected. It described spatial, time and
momentum projection in eight duplicated coordinates and is not an energy-monitoring
option. Existing historical full-state diagnostics are readers of that former
record type, not an active simulation path. Canonical model guides and theory
PDFs document the current methods; older rendered diagrams are historical.

## Comparison and archive record ownership

The three-, four-, and five-method studies share numerical setup, method
construction, aligned-result validation, and metric reductions through
`studies._comparison`. Their public configuration and result classes remain
available from their existing study modules and the `studies` facade. The family
configurations and results are siblings of neutral common bases; adding a method
does not make one campaign depend on another campaign's private functions. The
three- and four-method runners retain their alternating timing order. The
five-method runner retains reference reuse, optional parallel model campaigns,
and completion logging.

`contracts.studies.comparison` owns immutable reference and accuracy series, execution
records, and comparison summary records. `ComparisonReadView` describes the
shared read interface of live five-method results and loaded CSV views. CSV
loading constructs these concrete records rather than mutable attribute bags;
the existing CSV schema, public loader, and public study imports are preserved.
New numerical consumers read the canonical `residual_evaluations` complete-step
series.

`contracts.studies.gc_energy_bound` owns `GCEnergyBoundResult`;
`contracts.studies.bm4_parallel_recurrence` owns `ParallelBM4RecurrenceConfig`
and `ParallelBM4RecurrenceResult`. The existing
study modules explicitly re-export these classes. Their HDF5/NPZ loaders and
visualizers import the record owners directly, so reading an archive does not
import study orchestration or execute a simulation. Existing energy and parallel
recurrence archive schemas are unchanged. Scalar configuration rules shared with
those records live in `contracts.studies._validation`; study-specific geometry,
reference identity, and execution policies remain in `studies`.

## Problem, options and diagnostic ownership

`problem.InitialValueProblem` associates compatible dynamics and initial values;
`simulation` re-exports that same class. `execution.options.ExecutionOptions`
owns backend/device choices and is also exported by `execution`. These objects
retain their validation and snapshots. Importing the physical problem does not
require the execution package.

`diagnostics.studies.comparison` owns reference RMS floors and physical-energy error
reductions. `diagnostics.studies.bm4_recurrence.recurrence_distances` owns periodic initial
returns. Their input records stay in `contracts.studies.comparison` and
`contracts.studies`, with validation and independent arrays but without
analysis methods or imports from diagnostics. The calculation formulas and
archive schemas are unchanged. See [API migration](api-migration.md).

`studies._validation.require_uncoupled_bm4` applies the policy for new calculations;
archive constructors retain historical coupling metadata. Shared scalar record
validation remains in `contracts.studies._validation`.

`contracts.arrays` applies the shared NumPy/JAX compatibility agreement, including
mixed values, nested sequences and tracers. It selects the array namespace, not
a device or executor, and the NumPy path does not import JAX. This narrow semantic
responsibility is retained rather than introducing a generic utilities package.

## Contract locations

Existing structural interfaces and shared records follow the source hierarchy
under `contracts/`. See the [domain map](contracts.md) for canonical imports.
Implementations remain in their own packages; contract modules do not load them
at runtime. Public facades retain their established exported names.

## Method-owned integration step

Configure fixed methods with `step`, for example `RK4(step=0.03)` (default 0.1
in runtime time units). `SimulationRequest` supplies only the interval and
output times. Complete main steps use the configured value; only the last step
is shortened to reach the final time. The method's `step` remains unchanged.
Independent shadow samples may use shorter durations without moving the main
trajectory. Host and JAX use the same schedule and record effective durations
in `step_sizes`. `advance` remains the numerical operation.

DOP853/Radau instead own `max_step` (default 0.1), an upper bound for their
adaptive controller. Move former request `max_step` arguments to the method:
`step` for fixed methods, `max_step` for adaptive methods. Requests no longer
accept a step argument. A non-divisible interval intentionally gives a different
main grid from the former uniform rescaling policy; old saved results retain
their recorded grids.
