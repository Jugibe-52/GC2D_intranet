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
| `diagnostics.symplecticity.jacobians` and `.paths` | `diagnostics.jacobians` and `diagnostics.paths` |
| `PreparedPotential` and `potential.prepared` | Read-only physical properties and canonical splines owned directly by `Potential` |
| `ScipyPotentialEvaluator`, `JaxPotentialEvaluator`, and `PotentialEvaluator` | `Potential.evaluate` uses NumPy/SciPy; construct `JaxPotential` for JAX and place inputs with `jax.device_put` |
| JAX evaluator grid methods | `evaluate_grid` stays NumPy-only; inherited `electric_field` follows the potential class, also on the full grid |
| Internal `dynamics._jax.JaxDynamics` | Shared GC/FC dynamics; execution preparation owns device selection and compilation |

The unused BM4 map/Jacobian wrappers, full-state projection implementations,
full-state energy/symplecticity observers, and retired full-state and per-factor
projection comparison studies have been removed. Tests of physical-field and
passive-energy derivatives now exercise the active dynamics directly.

```python

from formulations.layout import GCStateLayout
from dynamics import GuidingCenterDynamics
from simulation import ABBA4Implicit, InitialValueProblem, SimulationRequest, simulate

configuration = GCStateLayout().pack_components(x0, y0)
problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=0.3), configuration)
request = SimulationRequest.uniform(t_span=(0.0, 1.0), max_step=0.01, sample_count=101)
solution = simulate(problem, ABBA4Implicit(track_energy=True), request)
```

The current method projects only the two spatial copies. Energy tracking is
passive; it does not reproduce the retired projection of time and momentum.
ABBA4 observations now always describe the three continuous unprojected factors
and one outer multiplier. ABBA6 now follows the same structure with seven
unprojected pairs and one outer multiplier; previously saved results with seven
separate projections describe a different numerical map.

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

Import public configurations from `methods.extended`, `methods`, or `simulation`.
Their canonical definitions are `methods.extended.abba` (all ABBA variants) and
`methods.extended.bm4` (both BM4 variants). Coefficients and compositions live in
`methods.extended.core.composition`; projection, midpoint, Jacobians, energy and
numerical records live in the corresponding `core` modules. Event adapters are
in `methods.extended.observations`. Former per-order files and composition
wrappers are removed without forwarding imports.

`ABBA6Implicit` now takes `projection_placement="around_complete_composition"`
and performs one projection around seven unprojected pairs. This replaces the
former seven-solve map. Saved historical ABBA6 results must retain their original
identity and require recomputation for comparisons with the new method.
Its substep work arrays change from `(steps, 7)` to `(steps, 1)`. The ABBA6 event
now owns one multiplier and seven unprojected snapshots, sharing the ABBA4 event
structure. Exact tangents are available through
`diagnostics.abba6_implicit_step_particle_jacobians`.

## Current state and saved states (2026-10-20)

Initial configurations are now packed NumPy arrays. Use `from formulations.layout import GCStateLayout, FCStateLayout` and `GCStateLayout().pack_components(x, y)` (FC adds vx and vy). External functions in `states.generators` return those arrays. Pass them to `InitialValueProblem(dynamics, values, layout=...)`; canonical GC/FC layouts may be omitted. Replace snapshot-method calls with explicit layout operations on arrays. Physical inputs retain their units, ordering and validation.

`request.SimulationRequest` owns the initial time, integration span, maximum step and output times; its `simulation` re-export remains supported. `execution.state.IntegrationState` contains current internal time and values. `states.States` records physical samples and becomes the completed result returned by methods, executors and `simulate`. Use `.times`, `.values`, `.components()`, `.positions()` and `.velocities()`; `.t` and `.states` remain read-only aliases. `.initial_state` and `.layout` return independent snapshots. `.select(...)` keeps original sample times and whole-run diagnostics. Step statistics keep their own axes.

Remove imports of `states.State`, `solution.Solution`, `contracts.result.IntegrationData`, `integration.IntegrationCollector`, `states.layout` and `contracts.request`. There are no compatibility modules. `InitialValueProblem.initial_configuration`, result `.source`, `.y` and `.trajectory` are removed. Compare initial snapshots by value rather than object identity. An executor must return complete `States`; `simulate` validates and returns it without creating another container. For changes to diagnostic metadata, use `.with_diagnostics(...)`.

`save_solution` and `load_solution` keep their names and numerical archive schemas. Legacy files remain readable, including those without an initial snapshot. `calculate_area(values, layout=..., period=...)` stays in `diagnostics.geometry`. Geometry labels remain study metadata. Fixed shadow samples never replace the main state; adaptive dense output and the numerical algorithms are unchanged. JAX constructs the same `States` at its host boundary.

## Explicit initial, solver and projection objects

See [initial data and numerical components](initial-solvers-formulations.md) for
`InitialConditions`, reusable NewtonSolver/BroydenSolver configurations and the
two projection classes. Existing array inputs, method names, tolerance controls,
simulate/executor signatures and stored result formats remain supported.

## Contract responsibilities (2026-10-22)

Import `InitialValueProblem` from `problem` internally or from `simulation`
publicly. Import `ExecutionOptions` from `execution.options` or `execution`.
The old `contracts.problem` and `contracts.execution_options` modules are removed
without aliases. CSV/NPZ/HDF5 schemas are unchanged; old Python pickles embedding
removed module paths are outside this migration.

Records retain their constructors and arrays. Call diagnostics explicitly:

| Previous call | Current call | Import module |
|---|---|---|
| `reference.time_integrated_rms_floor` | `time_integrated_rms_floor(reference)` | `diagnostics.comparison` |
| `reference.final_rms_floor` | `final_rms_floor(reference)` | `diagnostics.comparison` |
| `series.rms_error` | `rms_error(series.errors)` | `diagnostics.comparison` |
| `series.maximum_absolute_error` | `maximum_absolute_error(series.errors)` | `diagnostics.comparison` |
| `series.running_maximum_absolute_error` | `running_maximum_absolute_error(series.errors)` | `diagnostics.comparison` |
| `result.recurrence_distances(period=period)` | `recurrence_distances(result, period=period)` | `diagnostics.bm4_recurrence` |

The removed record calculations have no forwarding methods. Energy plots now
require only the signed `errors` array from their series view and calculate the
reductions through diagnostics. `contracts.arrays` retains its existing imports
and NumPy/JAX behavior. New-work BM4 validation belongs to `studies._validation`;
reading historical coupled records remains supported.
