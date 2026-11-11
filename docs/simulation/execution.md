# Complete integration execution

`Execution` prepares and coordinates one complete local integration. Its canonical
implementation is `execution.execution.Execution`, publicly exported as
`from execution import Execution`.

```python
from execution import Execution
from methods.runge_kutta.rk4 import RK4
from simulation.runner import simulate

executor = Execution()
solution = simulate(problem.with_request(request), RK4(), execution=executor)
```

Omitting `execution` creates a reusable local executor. It prepares a fresh
numerical run, allocates one physical recorder and owns status transitions,
result finalization and execution metadata. Methods own numerical operations;
controllers own accepted steps and sampling; `States` owns physical recording
and the completed result. The executor retains no per-run state and does not
change thread counts, CPU affinity, worker processes or global JAX options.
The default is NumPy/SciPy on the host CPU; a single physical core is not enforced.

## Job and result boundary

`run(problem, method, *, options=None, observers=())` receives:

- `InitialValueProblem`: physical dynamics, initial array, layout and `request`.
- `NumericalMethod`: a reusable provider of fresh numerical runs and capabilities.
- Optional `ExecutionOptions`: backend and device selection for this job.
- `observers`: passive energy configuration and accepted-step callbacks.

The executor reads the interval and saved times from `problem.request`, a
`SimulationRequest` defined in `request.py`. It passes the request directly to
planning and host recording, while numerical preparation receives only physical
functions and initial data. Subclasses override the same two-input signature.

The executor calls `method.new_run`, validates the fresh run once and owns its
running/finished transitions, including kernel preparation and result failures.
It allocates one result with `States.prepare`. The `run_steps` worker in
`execution._host` records streaming samples; the worker in `execution._jax_fixed`
records a validated physical history through `States.record_history`, without a
Python loop over output rows. Both return numerical diagnostics without
sealing the result. Execution combines metadata and diagnostics and calls
`States.finalize` once, applying the same legacy aliases to both paths.
Numerical formulas, nonlinear solves, observations and convergence checks retain
their owners. The executor does not call `simulate`. Failures propagate without
a local fallback, and consumed runs become finished.

Preparation is split into three private functions in the same module:

`EnergyObserver()` records physical H and the positive analytic partial time derivative `hamiltonian_time_derivative` at accepted endpoints and requested physical samples. It does not consume internal stages or integrate momentum. Per-step diagnostics are `energy_start`, `energy_end`, `energy_change`, `energy_time_derivative_start` and `energy_time_derivative_end`, with shape `(steps,N)`. Saved histories `physical_hamiltonian` (alias `hamiltonian`), `energy_drift` and `hamiltonian_time_derivative` have shape `(N,T)`. Energy variation in a time-dependent Hamiltonian is not a numerical conservation error.

DOP853 and Radau are standalone SciPy reference solvers used directly by studies. They are no longer methods in `simulation` or `methods`, and there is no adaptive controller or SciPy/JAX hybrid path in execution. Fixed methods own `step`; requests own the interval and saved times. Host progress is configured with `options=ExecutionOptions(progress=True)`.

`Execution.run` keeps argument validation, status transitions, allocation,
worker selection and finalization visible together. Step preparation and compilation
remain inside its `try/finally`, so preparation failures also finish the consumed
run. Helpers receive explicit inputs and return typed results; no per-run data
is stored on the executor. Method metadata is combined with execution metadata
and then step diagnostics, preserving the existing precedence and aliases.

The returned `states.States` contains physical arrays of shape
`(saved_time_count, physical_state_size)`, saved times, an independent layout and
initial snapshot, and read-only diagnostics. `simulate` calls the executor once,
requires a complete result, checks exact requested times and initial values, and
returns that same object. These checks also apply to executor subclasses.

## Shared fixed-step plan

Before step preparation or result allocation, `Execution.run` calls
`execution._plan.build_fixed_step_plan(request, step)` for the built-in fixed
controller or compiled fixed route. The returned `FixedStepPlan` owns read-only
NumPy arrays: main-step starts, ends and durations; requested output times;
per-step sample offsets; and sample modes and shadow durations. Modes select
the start, end or independent shadow state. Offsets partition output indices
starting at 1 because the initial state is saved separately.
The plan stores O(S + T) scalar metadata for S steps and T outputs, not physical
trajectories; the host route now prepares these sampling arrays in advance.

The builder preserves the last-step adjustment, round-off tolerance and priority
of the right endpoint. Host and JAX consume the same decisions, including empty
sample groups for unsaved steps. Shadow maps start from the accepted left state
and never feed back into the main trajectory. Host step
observers still run once per main step before its samples.

The host fixed controller receives the plan and the JAX preparation function
receives it explicitly as its third argument. `prepare_integration` only places
inputs and obtains the compiled executable; neither worker rebuilds the plan.
Custom controller subclasses retain their own interval and sampling rules.
Standalone `FixedStepController().steps(...)` can build a plan on demand, and
its arbitrary-time `sample(...)` operation remains available.

## Independent backend options

```python
import jax
from execution.options import ExecutionOptions

jax.config.update("jax_enable_x64", True)
solution = simulate(
    problem.with_request(request), RK4(),
    execution=executor,
    options=ExecutionOptions(backend="jax", device="cpu"),
)
```

`Potential.evaluate` uses NumPy/SciPy and `JaxPotential.evaluate` uses JAX.
Inherited electric fields follow the instance; `evaluate_grid` stays NumPy-only.
Neither potential class accepts execution options. Existing study helpers keep
`run_rk4_execution_comparison(..., executions=(options, ...))` and
`run_poincare_star(..., execution=options)` for their backend selections.

## Method preparation contract

`IntegrationMethod.new_run(f, initial_conditions)` prepares a fresh run without
an algebraic solver or Jacobian. `ImplicitIntegrationMethod` in `methods.implicit`
adds `new_run(f, initial_conditions, *, jacobian=None)` and the solver contract;
concrete families retain their defaults. Both bases are exported from `methods`.
Explicit methods use the common base and reject the `jacobian` keyword. There
is no separate explicit base. Execution supplies derivatives only to implicit
configurations, and JAX reads derivative settings only for those runs.

Both paths call `initialize(initial_conditions)` after binding their functions.
Initial data and layout are temporary inputs; the common base retains `f`,
metadata, status and copy count, while execution owns the initial snapshot.
Only implicit runs retain `jacobian`. Supplied derivatives have independent
particle shape `(N,d,d)`; finite differences remain dense. Solver strategies,
Broyden's projection matrix and deferred derivative validation are unchanged.
There is no `jacobian_structure`, `capture_quadrature` or method-level `progress` option.

## Extending execution

`Execution_Modal(Execution)` overrides the same `run` signature, serializes the
complete job and sends it to a deployed Modal Function. Its worker uses
`Execution.run` to prepare and drive the integration in the remote environment.
It returns `States` locally, uses durable call IDs for recovery and leaves later
Backblaze uploads to local persistence. Remote execution supports NumPy/SciPy and
JAX CPU jobs with an appropriately configured image. See the
[Modal guide](modal-execution.md) for deployment, recovery and limitations.
Individual steps, field evaluations and Newton corrections stay in the environment
performing the integration. Executors accept `options`, preserve numerical and
sampling controls, and propagate failures.

## Migration

Replace `method.integrate(problem, request, execution=options)` with
`simulate(problem, method, options=options)` or
`Execution().run(problem, method, options=options)`.
`IntegrationMethod.integrate`, `_integrate_jax`, `CompiledFixedMethod` and its
`methods._compiled` module are removed. `new_run(execution=...)` is also retired:
backend selection belongs to the executor. External integrate-only methods must
implement the preparation protocol. No compatibility aliases are retained.

The older `contracts.execution.Execution` configuration class remains removed. Use `Execution` for complete jobs and `ExecutionOptions` from `execution.options` for backend choices. `simulate` and executors additionally accept `observers=()`; method constructors use `solver=` for algebraic controls. Numerical archive formats remain unchanged. See [API migration](api-migration.md) for removed keywords.

## Unified execution package

`execution.execution.Execution.run` is the single complete local coordinator.
`execution._jax` prepares the optional device environment and contains the cached
SciPy/JAX evaluation adapter. `execution._host` owns host advancement, sampling,
progress and shared step-count rules; `execution._jax_fixed` owns the compiled
device loop and numerical export. `execution.state` defines accepted physical
snapshots. Workers receive an open recorder and never own run status or finalize
results. These are internal implementation functions, not additional executors.

The `integration` package has been removed. Migrate `integration.core.IntegrationMethod`
to `methods.base.IntegrationMethod`, controller imports to `execution`, and
`integration.state.IntegrationState` to `execution.state.IntegrationState`.
The private complete coordinators `integrate_method` and `integrate_fixed` are
also removed; complete callers use `simulate` or `Execution.run`. The former
`execution._jax_run`, `execution._fixed` and `dynamics._jax_host` modules are
retired without compatibility aliases. Saved physical-result formats remain
unchanged; serialized live snapshots using removed paths are not supported archives.

## Explicit initial, solver and projection objects

See [initial data and numerical components](initial-solvers-formulations.md) for `InitialConditions`, reusable NewtonSolver/BroydenSolver configurations and the two projection classes. Array inputs, method names and stored result formats remain supported. Move method `newton_*` controls into `solver=...`, and energy/step observers into `simulate(..., observers=...)`; the retired keywords have no compatibility aliases.

## Visible preparation results

Result assignments follow the [source typing convention](../development/typing.md):
one receiving variable is annotated inline, and new unpacked names are declared
before assignment. Existing optional initializations retain their union with `None`.
`_prepare_backend` returns `tuple[DynamicalSystem, Device | None, str, dict[str, DiagnosticValue]]`;
`resolve_device` returns `jax.Device`. The coordinator checks that the device is
not `None` before passing it to `prepare_step`, which returns `CompiledStep`.
`prepare_integration` returns `PreparedIntegration`, with a `jax.stages.Compiled`
executable and fresh `jax.Array` inputs. Its optional local reference selects
the compiled worker; the default route retains `None`.
Device imports used only for annotations
stay under `TYPE_CHECKING`; ordinary CPU imports still do not load JAX.

## Method-owned integration step

Configure fixed methods with `step`, for example `RK4(step=0.03)` (default 0.1
in runtime time units). `SimulationRequest` supplies only the interval and
output times. Complete main steps use the configured value; only the last step
is shortened to reach the final time. The method's `step` remains unchanged.
Independent shadow samples may use shorter durations without moving the main
trajectory. Host and JAX use the same schedule and record effective durations
in `step_sizes`. `advance` remains the numerical operation.

## Temporal and derivative preparation

`auto` selects a supplied Jacobian without evaluating it, or finite differences when no derivative is supplied. For Newton corrections, `analytic` requires a derivative and `finite_difference` explicitly forces approximation. Broyden continues to use the projection-supplied initial matrix and residual updates without evaluating field derivatives. Shapes and finite values are checked at actual evaluations. A supplied derivative failure propagates; auto does not silently fall back. Gauss, SDIRK, HBVM and projected compositions share `solvers.derivatives.resolve_jacobian_method`. HBVM retains its validation boundary after the first residual. Numerical stage and projection equations are unchanged.

## Common and implicit extension points

The structural `NumericalMethod` protocol requires only
`new_run(f, initial_conditions)`. An external provider can implement exactly
that signature and return a ready IntegrationMethod. To receive an available
analytic field derivative automatically, an implicit extension derives from
`methods.implicit.ImplicitIntegrationMethod`. Execution checks that capability
before accessing the dynamics' Jacobian provider; it does not inspect signatures
or infer implicitness from a solver attribute. A wrapper that only implements
the common protocol must bind any additional dependencies itself.

`FixedStepMethod` describes scheduling independently of implicitness. The
Newton Runge--Kutta base, projected palindromic base and HBVM combine their family
or fixed-step base with ImplicitIntegrationMethod. This shares preparation
without duplicating stages, projection equations or solver defaults. JAX cache
signatures include derivative identity and strategy only for implicit runs;
explicit signatures depend on method configuration and physical size.

## Horizontal transport during execution

`HorizontalTransportObserver(cycle_duration=1.0)` is defined in
`diagnostics.execution.horizontal_transport` and exported from `diagnostics`.
Pass it through `simulate(..., observers=(HorizontalTransportObserver(),))`.
It integrates the absolute physical horizontal velocity on every accepted
complete interval with the composite trapezoidal rule. Internal composition
stages, backward splitting substeps and shadow output samples are excluded.
The width is the largest minus smallest unwrapped x coordinate across the
initial state and all accepted endpoints. It is not the width of folded
Poincare samples. The dimensionless rate is travel / (elapsed cycles * width),
where elapsed cycles = elapsed runtime time / `cycle_duration`.

The final per-particle vectors are `horizontal_travel`, `horizontal_min`,
`horizontal_max`, `horizontal_width` and `horizontal_transport_rate`.
A zero width yields NaN for the rate and is identified by
`horizontal_zero_width="undefined_nan"`. Scalar metadata includes
`horizontal_cycles`, `horizontal_cycle_duration`, `horizontal_observed_steps`,
`horizontal_coordinates="physical_unwrapped"` and
`horizontal_quadrature="trapezoid_accepted_endpoints"`.
NumPy and JAX use the same accepted endpoints and O(particles) accumulation;
no per-step transport history is retained. The cycle-only Modal worker preserves
these five final vectors while discarding other diagnostic arrays.

This is a numerical quadrature, not an exact integral. Both the integral and
sampled width may miss excursions below the integration step; check step
refinement when assessing accuracy. The observer does not change the method,
its physical trajectory, saved output times or other observers.
