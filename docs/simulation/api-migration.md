# Retired API migration

New code uses one execution function and one public class for each numerical
method. The following interfaces have been removed, including their exports:

| Removed interface | Current interface |
|---|---|
| `contracts.execution.Execution` | `execution.Execution` executes complete jobs; `execution.options.ExecutionOptions` holds backend choices |
| `simulate(..., execution=Execution(backend="jax"))` | `simulate(..., options=ExecutionOptions(backend="jax"))`; `execution` now selects an executor |
| `SimulationRunner().simulate(...)` | `simulate(...)` |
| `ABBA4ImplicitSingleProjection(...)` | `ABBA4Implicit(...)` |
| `ABBA4ImplicitSingleProjectionIntegrationStep` | `ABBA4ImplicitIntegrationStep`, describing one outer projection |
| `abba4_implicit_single_projection_step_particle_jacobians` | `abba4_implicit_step_particle_jacobians` |
| `Trajectory`, `TrajectoryGC`, `TrajectoryFC` | `InitialValueProblem(dynamics, values, layout=...)` |
| `centered_gc_trajectory` | `centered_gc_configuration` |
| Physical `rho`/`eta` metadata on initial configurations and `Area` | Explicit dynamics parameters or study configuration |
| `diagnostics.symplecticity.jacobians` and `.paths` | `diagnostics.methods.jacobians` and `diagnostics.io.paths` |
| `PreparedPotential` and `potential.prepared` | Read-only physical properties and canonical splines owned directly by `Potential` |
| `ScipyPotentialEvaluator`, `JaxPotentialEvaluator`, and `PotentialEvaluator` | `Potential.evaluate` uses NumPy/SciPy; construct `JaxPotential` for JAX and place inputs with `jax.device_put` |
| JAX evaluator grid methods | `evaluate_grid` stays NumPy-only; inherited `electric_field` follows the potential class, also on the full grid |
| Internal `dynamics._jax.JaxDynamics` | `dynamics.jax_dynamics.JaxGuidingCenterDynamics` / `JaxFullCyclotronDynamics` inherit GC/FC equations and only declare JAX coordinates; execution prepares potentials and devices |

The unused BM4 map/Jacobian wrappers, full-state projection implementations,
full-state energy/symplecticity observers, and retired full-state and per-factor
projection comparison studies have been removed. Tests of physical-field and
passive-energy derivatives now exercise the active dynamics directly.

```python

from dynamics import GuidingCenterDynamics
from diagnostics import EnergyObserver
from simulation import ABBA4Implicit, InitialValueProblem, SimulationRequest, simulate

configuration = GuidingCenterDynamics.coordinates.pack_components(x0, y0)
problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=0.3), configuration)
request = SimulationRequest.uniform(t_span=(0.0, 1.0), sample_count=101)
solution = simulate(problem, ABBA4Implicit(step=0.01), request,
                    observers=(EnergyObserver(),))
```

The current method projects only the two spatial copies. Energy tracking is
passive; it does not reproduce the retired projection of time and momentum.
ABBA4 observations now always describe the three continuous unprojected factors
and one outer multiplier. ABBA6 has been removed; previously saved results
retain their original method and projection labels.

New comparison records use the method name `ABBA4Implicit`. Configuration keys
use the `abba4_outer_projection` prefix to distinguish them from historical
three-projection records. Old measurements must retain their original method
identity. Recalculate a study to obtain records under the new interface; do not
relabel historical three-projection or full-state measurements.

Experiment notebooks now use the current interfaces, explicit physical settings,
and four ABBA4 energy-tracking configurations (two projection formulations times
two solvers). The former R6/R8 notebook filenames remain stable, but their
opening notes explain the changed scientific scope. Stale outputs were cleared.
Exact original versioned notebooks and their outputs are archived locally under
`outputs/legacy_notebooks/`, with source commit and content hashes in a manifest;
the same records remain in Git history. Development notebooks remain ignored.

Validation uses short disposable notebook executions. Their reduced settings and
outputs are never saved into the canonical experiment notebooks.

## Study result visualization

Study results retain their data, summaries and numerical analysis. Plotting and
animation now belong to `visualization`: import a function and pass the result
as its first argument. The result methods below have been removed without
compatibility wrappers; their plotting options and return values are unchanged.

| Result and removed method | Function | Import module |
|---|---|---|
| GC symplecticity results: `plot_diagnostics()` | `plot_gc_symplecticity_diagnostics(result)` | `visualization.gc_symplecticity` |
| GC symplecticity results: `plot_solver_diagnostics()` | `plot_gc_solver_diagnostics(result)` | `visualization.gc_symplecticity` |
| GC symplecticity results: `animate(...)` | `animate_gc_symplecticity(result, ...)` | `visualization.gc_symplecticity` |
| `ABBA2MidpointSymplecticityResult.plot_convergence()` | `plot_abba_midpoint_convergence(result)` | `visualization.gc_symplecticity` |
| `ABBASymplecticityResult.plot_defect_floor()` | `plot_abba_defect_floor(result)` | `visualization.gc_symplecticity` |
| `RK4SymplecticityResult.plot_convergence()` | `plot_rk4_convergence(result)` | `visualization.gc_symplecticity` |
| `ABBAComparisonResult.plot_runtime_comparison()` | `plot_abba_runtime_comparison(result)` | `visualization.abba_comparison` |
| `ABBAComparisonResult.plot_trajectory_differences()` | `plot_abba_trajectory_differences(result)` | `visualization.abba_comparison` |
| `ABBAComparisonResult.animate(...)` | `animate_abba_comparison(result, ...)` | `visualization.abba_comparison` |
| `AreaComparisonResult.animate(...)` | `animate_area_comparison(result, ...)` | `visualization.area_comparison` |
| `GeneralizedEnergyResult.plot()` | `plot_generalized_energy_comparison(result)` | `visualization.energy` |
| `TrajectorySymplecticityResult.plot_symplecticity()` | `plot_trajectory_symplecticity_result(result)` | `visualization.trajectory_symplecticity` |
| `TrajectorySymplecticityResult.plot_trajectories(...)` | `plot_trajectory_symplecticity_trajectories(result, ...)` | `visualization.trajectory_symplecticity` |

For example, after running an RK4 symplecticity study:

```python
from visualization.gc_symplecticity import plot_gc_symplecticity_diagnostics

figure, axes = plot_gc_symplecticity_diagnostics(result)
```

This changes presentation ownership only. Study configurations, integration,
saved results and scientific diagnostics retain their existing contracts.

## Extended-family module organization and ABBA6

Import public configurations from `methods.extended`, `methods` or `simulation`. ABBA variants are defined in `methods.extended.abba`, BM4 in `methods.extended.bm4`, and compositions in `methods.extended.core.composition`. Projection, midpoint, Jacobians and numerical records remain in the matching `core` modules; passive energy belongs to `diagnostics.methods.energy`. Event adapters live in `methods.extended.observations`. Retired per-order modules and energy implementations have no forwarding imports.

`ABBA2Implicit`, `ABBA4Implicit` and `BM4Implicit` directly inherit
`ProjectedPalindromicMethod`, which owns preparation, solver validation and
advancement. ABBA4 declares its full tuple and applies one exterior projection.
`ABBA6Implicit`, `ABBA6ImplicitIntegrationStep`, `studies.abba6_accuracy`,
its study exports and `diagnostics.abba6_implicit_step_particle_jacobians` are
removed without aliases. Retained ABBA6 theory and saved results are historical;
do not relabel those trajectories as another method.

## Current state and saved states (2026-10-20)

Initial configurations are now packed NumPy arrays. Use `from dynamics import GuidingCenterDynamics, FullCyclotronDynamics` and `GuidingCenterDynamics.coordinates.pack_components(x, y)` (FC adds vx and vy). External functions in `states.generators` return those arrays. Pass them to `InitialValueProblem(dynamics, values, layout=...)`; canonical GC/FC layouts may be omitted. Replace snapshot-method calls with explicit layout operations on arrays. Physical inputs retain their units, ordering and validation.

`request.SimulationRequest` owns the initial time, integration span and output times; its `simulation` re-export remains supported. `execution.state.IntegrationState` contains current internal time and values. `states.States` records physical samples and becomes the completed result returned by methods, executors and `simulate`. Use `.times`, `.values`, `.components()`, `.positions()` and `.velocities()`; `.t` and `.states` remain read-only aliases. `.initial_state` and `.layout` return independent snapshots. `.select(...)` keeps original sample times and whole-run diagnostics. Step statistics keep their own axes.

Remove imports of `states.State`, `solution.Solution`, `contracts.result.IntegrationData`, `integration.IntegrationCollector`, `states.layout` and `contracts.request`. There are no compatibility modules. `InitialValueProblem.initial_configuration`, result `.source`, `.y` and `.trajectory` are removed. Compare initial snapshots by value rather than object identity. An executor must return complete `States`; `simulate` validates and returns it without creating another container. For changes to diagnostic metadata, use `.with_diagnostics(...)`.

`save_solution` and `load_solution` keep their names and numerical archive schemas. Legacy files remain readable, including those without an initial snapshot. `calculate_area(values, layout=..., period=...)` stays in `diagnostics.states.geometry`. Geometry labels remain study metadata. Fixed shadow samples never replace the main state; adaptive dense output and the numerical algorithms are unchanged. JAX constructs the same `States` at its host boundary.

## Explicit initial, solver and projection objects

See [initial data and numerical components](initial-solvers-formulations.md) for `InitialConditions`, reusable NewtonSolver/BroydenSolver configurations and the two projection classes. Array inputs, method names and stored result formats remain supported. Move method `newton_*` controls into `solver=...`, and energy/step observers into `simulate(..., observers=...)`; the retired keywords have no compatibility aliases.

## Contract responsibilities (2026-10-22)

Import `InitialValueProblem` from `problem` internally or from `simulation`
publicly. Import `ExecutionOptions` from `execution.options` or `execution`.
The old `contracts.problem` and `contracts.execution_options` modules are removed
without aliases. CSV/NPZ/HDF5 schemas are unchanged; old Python pickles embedding
removed module paths are outside this migration.

Records retain their constructors and arrays. Call diagnostics explicitly:

| Previous call | Current call | Import module |
|---|---|---|
| `reference.time_integrated_rms_floor` | `time_integrated_rms_floor(reference)` | `diagnostics.studies.comparison` |
| `reference.final_rms_floor` | `final_rms_floor(reference)` | `diagnostics.studies.comparison` |
| `series.rms_error` | `rms_error(series.errors)` | `diagnostics.studies.comparison` |
| `series.maximum_absolute_error` | `maximum_absolute_error(series.errors)` | `diagnostics.studies.comparison` |
| `series.running_maximum_absolute_error` | `running_maximum_absolute_error(series.errors)` | `diagnostics.studies.comparison` |
| `result.recurrence_distances(period=period)` | `recurrence_distances(result, period=period)` | `diagnostics.studies.bm4_recurrence` |

The removed record calculations have no forwarding methods. Energy plots now
require only the signed `errors` array from their series view and calculate the
reductions through diagnostics. `contracts.arrays` retains its existing imports
and NumPy/JAX behavior. New-work BM4 validation belongs to `studies._validation`;
reading historical coupled records remains supported.

## Diagnostics by observed process (2026-10-22)

Diagnostic implementations now live under `diagnostics.execution`,
`diagnostics.methods`, `diagnostics.formulations`, `diagnostics.states`,
`diagnostics.studies` and `diagnostics.io`. The existing top-level exports of
`diagnostics` retain their names. Direct imports from former flat submodules
must use the new process path; no compatibility modules are provided.
See the [diagnostic package guide](../../src/diagnostics/README.md) for ownership
and import examples. Numerical behavior, archive schemas and output destinations
are unchanged. Working notebook imports have been migrated without replacing
stored results; historical conflict copies retain their original references.

## Contracts organized by source domain

Import pure interfaces and shared records from the matching `contracts` domain;
see the complete [contract map](contracts.md). For example, use
`contracts.methods.base.NumericalMethod`, `contracts.dynamics.coordinates.StateLayout`,
and `contracts.dynamics.protocols.DynamicalSystem`. Common events live in
`contracts.methods.observation`; adaptive, Gauss, ABBA, and BM4 events live beside
their matching method paths. Public `simulation`, `methods`, `formulations`,
`execution`, `dynamics`, and `visualization` exports keep their existing identities.

The old flat contract modules and `dynamics.protocols` are removed, without
compatibility modules. `contracts.execution` is now a domain package containing
`controllers`, not the former execution-configuration module. Implementations such
as `methods.base.IntegrationMethod` and `execution._host.FixedStepController`
stay in their domains. Numerical archives keep their schemas; old Python pickles
with removed module paths are not portable archives. Regenerate those objects
with the current source before process or remote execution.

## Independent state representations

`DoubleFormulation(physical_size)` is the only state formulation. It prepares two temporary physical copies through `lift`, `join_copies` and `split_copies`, preserving NumPy/JAX backends and complete FC velocity blocks. It stores only a positive vector size; it owns no dynamics, layout, time, momentum, solver or projection. Methods using ordinary physical arrays need no formulation; `SingleFormulation` is removed.

Every accepted snapshot is `IntegrationState(time, values)`: scalar time and physical `values` of shape `(d*N,)`. GC remains 2N and FC 4N with or without energy observation. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. Optional quadrature accompanies a sample without storing energy momentum. Generic copy support does not extend GC numerical methods to FC.

Execution supplies `f(t, z)`, independent initial data and an optional Jacobian to the numerical method. `advance(t, z, h)` returns physical `StepResult.state`, numerical work and optional quadrature. `EnergyObserver`, supplied through `simulate(..., observers=(EnergyObserver(),))`, owns a fresh passive accumulator for each run. Accepted steps update it once; each shadow sample starts independently from the accepted left endpoint. JAX carries a separate functional accumulator through the compiled loop.

Execution gives the energy observer H and `-partial_t H`. The observer evaluates saved physical H once; `states.energy.energy_diagnostics` combines supplied H and physical kappa without evaluating dynamics. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. The default initial kappa is zero; `EnergyObserver(initial_momentum=...)` accepts an explicit physical value per particle. Numerical archives require no migration; old pickled configured objects should be recreated.

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

## Physical coordinates and retired paths (2026-10-24)

Dynamics declare `GCCoordinates()` for fixed `(x, y)` blocks or `FCCoordinates()` for fixed `(x, y, vx, vy)` blocks, from `dynamics.coordinates`. `PhysicalCoordinates` is a stateless base with common array operations; dimensions are fixed properties and no names or role-index fields are configured. `split()` returns an ordinary tuple. `JaxGCCoordinates` and `JaxFCCoordinates`, from `dynamics.jax_coordinates`, inherit their respective operations and only select `jnp` through `array_namespace`.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a model from dimension. `StateLayout.snapshot()` explicitly supplies an immutable NumPy convention; `describe_coordinates` validates that snapshot and its dimension. Compatibility compares the snapshot class and equality, even for equally sized arrays. Initial conditions and results own their arrays and conventions without retaining live dynamics or a potential. Custom conventions derive from `PhysicalCoordinates` and implement fixed operations instead of role tables; external adapters may supply these snapshots without inheritance. Packing and views preserve `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)`. The respective JAX classes support tracing; incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. Saved numerical archives retain existing GC/FC schemas and can be interpreted without constructing dynamics or loading a field.

The former GC/FC wrappers and internal state containers are removed without aliases. Maps belong to `methods.extended.maps.GCDoubledMaps` and `methods.splitting.FCSplitMaps`; equations to `methods.projection`; projected observation to `diagnostics.methods.projection`.

## Callable methods, solver controls and observers (2026-10-24)

| Retired configured interface | Current owner |
|---|---|
| `method.new_run(problem, request, evaluated_dynamics=...)` | `method.new_run(f, initial_conditions, request, jacobian=..., jacobian_structure=...)` |
| Method `track_energy=True` | `simulate(..., observers=(EnergyObserver(),))`, imported from `diagnostics` |
| Method `step_observer=callback` | `simulate(..., observers=(callback,))` |
| Method `newton_observer=callback` | `solver=NewtonSolver(iteration_observer=callback)` |
| Method `newton_absolute_tolerance`, `newton_relative_tolerance`, `newton_max_iterations` | Solver `absolute_tolerance`, `relative_tolerance`, `max_iterations` |
| Method `newton_jacobian_method`, `newton_jacobian_relative_step` | Solver `jacobian_method`, `jacobian_relative_step` |
| Method `nonlinear_solver="broyden"` | `solver=BroydenSolver(...)` |
| `SolverOptions`, `SingleFormulation`, method momentum and `diagnose_history` | Removed; use one configured solver, physical arrays and independent energy observation |

Explicit solvers supply every algebraic control. Preserve the effective family defaults when migrating only one old keyword: Gauss/SDIRK use tolerances 1e-14/1e-13 and 20 iterations; ABBA/BM4/HBVM use 1e-13/1e-12 and 12 iterations. HBVM defaults to `backtracking=True`. Set `jacobian_method` explicitly when preserving a prior analytic or finite-difference choice. Physical initial values, numerical coefficients, requested sampling and saved archive schemas are unchanged.

Methods retain only the supplied field, optional derivative and independent numerical data. `DoubleFormulation` remains a small helper for temporary physical copies. Existing callbacks keep their identity and lifecycle; reusing `EnergyObserver` creates fresh built-in state for every run. No wrapper silently translates retired constructor arguments.

## Explicit JAX physical coordinates (2026-10-25)

`PhysicalCoordinates` now consistently uses NumPy. For direct JIT, vmap or
automatic differentiation, use `JaxGCCoordinates` and `JaxFCCoordinates` from `dynamics.jax_coordinates`.
The same shapes, fixed conventions and validation are inherited. `describe_coordinates`
continues producing a structural NumPy snapshot for initial conditions and results.

Direct compiled GC/FC calls use `JaxGuidingCenterDynamics` or
`JaxFullCyclotronDynamics` from `dynamics.jax_dynamics` with `JaxPotential`.
Import the JAX classes from their explicit modules; the base package remains
independent of optional JAX. The base dynamics with a JAX potential alone
are no longer the compiled interface. Normal `simulate` calls need no changes:
`ExecutionOptions(backend="jax")` prepares both JAX coordinates and potentials,
without refitting splines or modifying the original objects. The old internal
`dynamics._jax.JaxDynamics` route remains removed. Saved numerical archives and
physical equations do not change. See the [examples](../dynamics/protocols.md).


## Fixed GC/FC coordinate classes (2026-10-25)

Replace configurable `PhysicalCoordinates(...)` with `GCCoordinates()` or
`FCCoordinates()`. The respective JAX implementations are `JaxGCCoordinates()`
and `JaxFCCoordinates()`; `JaxPhysicalCoordinates` is removed without an alias.

```python
from dynamics.coordinates import GCCoordinates, FCCoordinates
from dynamics.jax_coordinates import JaxGCCoordinates, JaxFCCoordinates

gc = GCCoordinates()
fc = FCCoordinates()
jax_gc = JaxGCCoordinates()
jax_fc = JaxFCCoordinates()
```

The fixed orders are `(x, y)` and `(x, y, vx, vy)`. There are no constructor
fields for dimension, names or indices. Dimension remains a read-only property.
`PhysicalCoordinates` is an abstract, stateless base. Its common operations remain
`split`, `pack_components`, `as_blocks`, `from_blocks` and `particle_count`.
GC and FC implement positions directly; only FC supplies stored velocities.

`CoordinateValues` and the three name fields are removed. The interim
`position_indices` and `velocity_indices` fields are also removed. Use ordinary
tuple unpacking or indexing instead of attributes such as `components.x`.

The structural `StateLayout` contract now includes `snapshot()`, returning an
immutable NumPy `PhysicalCoordinates` implementation. GC/FC snapshots preserve
their explicit class, including from their JAX subclasses. External adapters
provide that convention explicitly; custom coordinate classes implement their
fixed dimension and accessors and use distinct snapshot classes for distinct
conventions. Derive different physical conventions directly from the base;
GC/FC subclasses inherit the same physical meaning and snapshot behavior.
Methods may import shared coordinate types to validate their inputs, but never
physical dynamics equations or objects.

Existing GC/FC numerical archives retain their labels and schemas. Recreate old
configured-object pickles; current coordinate, initial-condition, problem and
result objects still support serialization. No saved arrays or units change.
