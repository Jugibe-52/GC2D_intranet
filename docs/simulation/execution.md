# Complete integration execution

`Execution` prepares and coordinates one complete local integration. Its canonical
implementation is `execution.execution.Execution`, publicly exported as
`from execution import Execution`.

```python
from execution import Execution
from methods.classical.rk4 import RK4
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
`SimulationRequest` defined in `request.py`, and passes it separately to
`method.new_run`. Subclasses override the same two-input signature.

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

- `_prepare_backend` resolves the execution mode and device once, prepares dynamics and binds the hybrid adapter when needed, importing JAX only when selected.
- `_prepare_observers` binds a fresh energy accumulator and preserves callback identity and order.
- `_prepare_numerical_run` supplies the Jacobian and quadrature requirements, validates a fresh run and returns its observation metadata.

`_prepare_observers` reads the request from the supplied problem.
`_prepare_numerical_run` receives `InitialConditions` and `SimulationRequest`
explicitly, alongside the method and prepared dynamics. It obtains the particle
count from the initial conditions and does not receive the complete problem.
`Execution.run` calls `execution._jax_step.prepare_step` directly for fixed JAX
execution. That function checks supported run types and rejects Python callbacks,
progress and iteration observers before copying resources. The local device
reference becomes `PreparedStep.device`. Next, `Execution.run` calls
`execution._jax_fixed.prepare_integration` with the shared temporal plan to
place fresh inputs and complete compilation before allocating the recorder. The returned `PreparedIntegration` holds the executable, inputs and
recording information. The JAX `run_steps(prepared, saved)` only executes,
synchronizes, validates and records. Hybrid evaluation retains the device in
its cached adapter; no global device setting is needed.

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
and never feed back into the main trajectory or energy accumulator. Host step
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

The caller opts into JAX and prepares global precision. Execution resolves the
device once and prepares shared physical dynamics without refitting splines or
mutating the source. Fixed JAX methods use a device-resident map; DOP853/Radau
receive prepared field/energy callbacks while SciPy retains adaptive control.
JAX is never imported by the default path. See the [JAX guide](jax-execution.md)
for supported devices, callbacks and failure contracts.

`Potential.evaluate` uses NumPy/SciPy and `JaxPotential.evaluate` uses JAX.
Inherited electric fields follow the instance; `evaluate_grid` stays NumPy-only.
Neither potential class accepts execution options. Existing study helpers keep
`run_rk4_execution_comparison(..., executions=(options, ...))` and
`run_poincare_star(..., execution=options)` for their backend selections.

## Method preparation contract

`NumericalMethod.new_run(f, initial_conditions, request, *, jacobian=None, jacobian_structure="dense", observe=False, capture_quadrature=False)` receives a physical vector field and independent initial data. Execution resolves dynamics capabilities and supplies any physical Jacobian explicitly. The prepared run retains no `InitialValueProblem` or `DynamicalSystem`. `jacobian_structure` distinguishes dense matrices from independent particle blocks; it does not identify a physical model.

`new_run` returns a fresh ready `IntegrationMethod`, copying constructor configuration and initializing owned physical snapshots. `initialize`, `advance`, `build_observation` and `controller` define its numerical behavior. Methods have no energy accumulator or history diagnosis. Fixed methods share one `advance` implementation across array types. Execution owns JAX compatibility checks and compilation; `NumericalMethod` has no backend declaration. The adaptive implementation retains its local hybrid declaration. CPU extensions can inherit `FixedStepMethod` or supply their own controller.

Execution registers `observers=()` separately from the method. `EnergyObserver()` is immutable configuration for a fresh per-run accumulator; ordinary callbacks keep their supplied identity and caller-owned lifecycle. Execution attaches the original dynamics identity to accepted events and calls each callback once per main step, in registration order. Shadow samples do not emit accepted events. A callback declaring `requires_quadrature=True` requests current-step quadrature without enabling saved energy histories.

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

`methods.base.IntegrationMethod` defines isolated numerical preparation and operations,
while `contracts.methods.base` owns the structural `NumericalMethod` protocol.
`contracts.execution.controllers` owns `StepController`. Import the base with
`from methods import IntegrationMethod`. Controller extensions use
`from execution import FixedStepController, StepController, IntegrationState`.
The SciPy controller remains next to its method-specific live solver adapter.
Complete simulations use `simulate` or `Execution.run`; the private host driver
is not exported as a second public execution entry point.

The `integration` package has been removed. Migrate `integration.core.IntegrationMethod`
to `methods.base.IntegrationMethod`, controller imports to `execution`, and
`integration.state.IntegrationState` to `execution.state.IntegrationState`.
The private complete coordinators `integrate_method` and `integrate_fixed` are
also removed; complete callers use `simulate` or `Execution.run`. The former
`execution._jax_run`, `execution._fixed` and `dynamics._jax_host` modules are
retired without compatibility aliases. Saved physical-result formats remain
unchanged; serialized live snapshots using removed paths are not supported archives.

Simulation preparation in `execution._jax` binds the requested device and shared
GC/FC equations to `JaxPotential` automatically. A simulation caller can keep a
base `Potential`; no manual backend-specific potential construction is required.
Standalone `JaxPotential` retains its own evaluation guards, device coefficient
cache and input placement rules. Numerical JIT kernels remain beside their
algorithms. Physics modules do not import execution; the hybrid SciPy/JAX bridge
belongs to `execution._jax`.

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

DOP853/Radau instead own `max_step` (default 0.1), an upper bound for their
adaptive controller. Move former request `max_step` arguments to the method:
`step` for fixed methods, `max_step` for adaptive methods. Requests no longer
accept a step argument. A non-divisible interval intentionally gives a different
main grid from the former uniform rescaling policy; old saved results retain
their recorded grids.
