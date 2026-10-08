# Complete integration execution

`Execution` is the local executor for one complete numerical integration. Its
canonical implementation is `execution.execution.Execution`; the public package
explicitly re-exports it as `from execution import Execution`.

```python
from execution import Execution
from methods.classical.rk4 import RK4
from simulation.runner import simulate

executor = Execution()
solution = simulate(problem, RK4(), request, execution=executor)
```

Omitting `execution` creates a local executor. It delegates to the method's
existing integration implementation, including its full time loop, potential
evaluations, nonlinear solves, observations and diagnostics. It keeps no run
state, so the same executor can serve successive integrations. It does not set
thread counts, CPU affinity, worker processes or JAX global options. The default
remains NumPy/SciPy on the host CPU; a single physical core is not enforced.

## Job and result boundary

`run(problem, method, request, *, options=None)` receives:

- `InitialValueProblem`: physical dynamics and the initial configuration.
- `NumericalMethod`: the integration algorithm and its numerical parameters.
- `SimulationRequest`: time interval, step limit and requested saved times.
- Optional `ExecutionOptions`: the existing backend/device selection for this job.

It returns `contracts.result.IntegrationData`, containing `t`, physical `states`
and integration diagnostics. States use the existing component-major layout,
with shape `(physical_state_size, saved_time_count)`. `simulation.runner.simulate`
validates the inputs, calls the executor once and constructs `Solution`.
`Solution` owns immutable output storage and structural validation; `simulate`
checks alignment with the requested times and initial state, including results
from executor subclasses.

The execution path is `simulate -> Execution.run -> method.integrate`.
Numerical steps remain in the methods and integration packages. The executor
does not call `simulate`, which would recurse into the same execution boundary.
Exceptions propagate to the caller without retrying through a local fallback.

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

The caller opts into JAX and prepares its global configuration. The executor
forwards the immutable choice to the existing numerical implementation. JAX is
never imported by the default execution path. See the [JAX execution guide](jax-execution.md)
for device support and the fixed versus adaptive integration distinction.

The lower-level method API retains its `execution` keyword for backend options:
`method.integrate(problem, request, execution=options)`. It accepts
`ExecutionOptions`, not an executor. `Potential` accepts no execution options:
`evaluate` and coordinate-based `electric_field` select NumPy or JAX from their
arguments, while full-grid evaluations remain NumPy-only. Simulation preparation
places arrays on the selected device and compiles the same GC/FC dynamics used
by the NumPy path. Existing study helpers also retain their backend-selection
arguments: `run_rk4_execution_comparison(..., executions=(options, ...))` and
`run_poincare_star(..., execution=options)`.

## Extending execution

`Execution_Modal(Execution)` overrides the same `run` signature, transfers the
complete job to a deployed Modal Function, and returns `IntegrationData` to the
local machine. It uses durable call IDs for recovery and leaves any subsequent
Backblaze upload to local persistence. Remote execution supports NumPy/SciPy and
JAX CPU jobs with a suitably configured image. See the [Modal executor guide](modal-execution.md) for the
separate deployment example, recovery contract and limitations. Individual steps,
potential evaluations and Newton corrections stay within the environment
performing the integration.
Executors must accept the optional `options` keyword, preserve the requested
method and sampling contract, and propagate failures.

## Migration

The old `contracts.execution.Execution` configuration class and its module have
been removed. Use `Execution` from `execution` for the executor and
`ExecutionOptions` from `contracts.execution_options` for backend choices.
Replace `simulate(..., execution=Execution(backend="jax"))` with
`simulate(..., options=ExecutionOptions(backend="jax"))`.
No compatibility import aliases are provided. Development notebook consumers
are migrated locally; versioned experiment notebooks remain outside this change.
