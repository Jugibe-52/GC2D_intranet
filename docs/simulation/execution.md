# Complete integration execution

`Execution` prepares and coordinates one complete local integration. Its canonical
implementation is `execution.execution.Execution`, publicly exported as
`from execution import Execution`.

```python
from execution import Execution
from methods.classical.rk4 import RK4
from simulation.runner import simulate

executor = Execution()
solution = simulate(problem, RK4(), request, execution=executor)
```

Omitting `execution` creates a reusable local executor. It prepares a fresh
numerical run, allocates one physical recorder and owns status transitions,
result finalization and execution metadata. Methods own numerical operations;
controllers own accepted steps and sampling; `States` owns physical recording
and the completed result. The executor retains no per-run state and does not
change thread counts, CPU affinity, worker processes or global JAX options.
The default is NumPy/SciPy on the host CPU; a single physical core is not enforced.

## Job and result boundary

`run(problem, method, request, *, options=None)` receives:

- `InitialValueProblem`: physical dynamics, initial array and layout.
- `NumericalMethod`: a reusable provider of fresh numerical runs and capabilities.
- `SimulationRequest`: time interval, step limit and requested saved times.
- Optional `ExecutionOptions`: backend and device selection for this job.

The executor calls `method.new_run`, validates the fresh run once and owns its
running/finished transitions, including kernel preparation and result failures.
It allocates one result with `States.prepare`. The `run_steps` worker in
`execution._host` records streaming samples; the worker in `execution._jax_fixed`
records a validated physical history through `States.record_history`, without a
Python loop over output columns. Both return numerical diagnostics without
sealing the result. Execution combines metadata and diagnostics and calls
`States.finalize` once, applying the same legacy aliases to both paths.
Numerical formulas, nonlinear solves, observations and convergence checks retain
their owners. The executor does not call `simulate`. Failures propagate without
a local fallback, and consumed runs become finished.

The returned `states.States` contains physical arrays of shape
`(physical_state_size, saved_time_count)`, saved times, an independent layout and
initial snapshot, and read-only diagnostics. `simulate` calls the executor once,
requires a complete result, checks exact requested times and initial values, and
returns that same object. These checks also apply to executor subclasses.

## Independent backend options

```python
import jax
from contracts.execution_options import ExecutionOptions

jax.config.update("jax_enable_x64", True)
solution = simulate(
    problem, RK4(), request,
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

`NumericalMethod` requires `new_run(problem, request, *, evaluated_dynamics=None)`
and a `jax_mode` capability: `"unsupported"`, `"device_resident"` or
`"hybrid_scipy_controller"`. `new_run` returns a fresh, ready `IntegrationMethod`
with independent resources, an immutable initial snapshot and numerical metadata.
The base implementation copies dataclass constructor options and calls `initialize`.
The optional evaluated dynamics is an environment-prepared evaluation interface;
the physical problem remains unchanged. CPU extensions can inherit IntegrationMethod
or provide the structural protocol and return a supported prepared run.

`IntegrationMethod` retains `initialize`, `advance`, `build_observation`,
`export_history` and `controller`. The default JAX capability is unsupported.
Built-in fixed methods declare device-resident execution; adaptive methods declare
hybrid control. Numerical family adapters retain explicit supported-type checks,
so declaring a mode alone does not authorize compiling unknown subclass overrides.
Executors use capabilities rather than branches on concrete numerical methods.

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
`simulate(problem, method, request, options=options)` or
`Execution().run(problem, method, request, options=options)`.
`IntegrationMethod.integrate`, `_integrate_jax`, `CompiledFixedMethod` and its
`methods._compiled` module are removed. `new_run(execution=...)` is also retired:
backend selection belongs to the executor. External integrate-only methods must
implement the preparation protocol. No compatibility aliases are retained.

The older `contracts.execution.Execution` configuration class remains removed.
Use `Execution` for the executor and `ExecutionOptions` from
`contracts.execution_options` for backend choices. The signatures of `simulate`,
executors and numerical constructors, archive formats and notebook outputs are
unchanged. Supported direct callers are migrated; no long notebook run is required.

## Unified execution package

`execution.execution.Execution.run` is the single complete local coordinator.
`execution._jax` prepares the optional device environment and contains the cached
SciPy/JAX evaluation adapter. `execution._host` owns host advancement, sampling,
progress and shared step-count rules; `execution._jax_fixed` owns the compiled
device loop and numerical export. `execution.state` defines accepted internal
snapshots. Workers receive an open recorder and never own run status or finalize
results. These are internal implementation functions, not additional executors.

`methods.base.IntegrationMethod` defines isolated numerical preparation and operations,
alongside the structural `NumericalMethod` protocol. Import the base with
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

See [initial data and numerical components](initial-solvers-formulations.md) for
`InitialConditions`, reusable NewtonSolver/BroydenSolver configurations and the
two projection classes. Existing array inputs, method names, tolerance controls,
simulate/executor signatures and stored result formats remain supported.
