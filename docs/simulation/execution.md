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

Omitting `execution` creates a local executor. It prepares a fresh numerical run,
selects the shared integration driver and supplies execution metadata. Methods own
their numerical operations; controllers own accepted steps and sampling; `States`
owns physical recording and the completed result. The executor retains no per-run
state and can be reused. It does not change thread counts, CPU affinity, worker
processes or global JAX options. The default is NumPy/SciPy on the host CPU; a
single physical core is not enforced.

## Job and result boundary

`run(problem, method, request, *, options=None)` receives:

- `InitialValueProblem`: physical dynamics, initial array and layout.
- `NumericalMethod`: a reusable provider of fresh numerical runs and capabilities.
- `SimulationRequest`: time interval, step limit and requested saved times.
- Optional `ExecutionOptions`: backend and device selection for this job.

The executor calls `method.new_run` and starts `integrate_method` for host control
or `integrate_fixed` for a prepared device map. Execution metadata enters the driver
before finalization, avoiding a second physical-history copy. Numerical formulas,
nonlinear solves, observations and convergence checks keep their existing owners.
The executor does not call `simulate`, which would re-enter the same boundary.
Failures propagate without a local fallback, and consumed runs become finished.

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
