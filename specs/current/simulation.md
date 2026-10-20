+++
schema_version = 1
kind = "spec"
id = "simulation"
sources = ["src/simulation/runner.py", "src/contracts/problem.py", "src/request.py", "src/execution/execution.py", "src/execution/_jax_run.py", "src/methods/base.py"]
documents = ["docs/simulation/integration-architecture.md", "docs/simulation/execution.md", "docs/simulation/jax-execution.md"]

[[requirements]]
id = "SIM-001"
tests = ["tests/test_core.py::SimulationTests.test_problem_and_request_validate_initial_state_and_time_span", "tests/test_state_ownership.py::StateOwnershipTests.test_problem_and_prepared_formulations_own_state_and_layout_snapshots"]

[[requirements]]
id = "SIM-002"
tests = ["tests/test_core.py::SimulationTests.test_problem_and_request_validate_initial_state_and_time_span", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_request_revalidates_normalized_endpoints"]

[[requirements]]
id = "SIM-003"
tests = ["tests/test_execution.py::ExecutionTests.test_default_and_explicit_executor_produce_the_same_solution", "tests/test_execution.py::ExecutionTests.test_simulation_dispatches_one_complete_job_to_the_subclass", "tests/test_execution.py::ExecutionTests.test_local_executor_owns_preparation_and_drives_the_controller", "tests/test_execution.py::ExecutionTests.test_executor_failure_propagates_without_local_fallback", "tests/test_execution.py::ExecutionTests.test_external_provider_prepares_a_run_without_an_integration_entry_point", "tests/test_execution.py::ExecutionTests.test_controller_preparation_failure_consumes_only_the_fresh_run", "tests/test_jax_methods.py"]

[[requirements]]
id = "SIM-004"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_invalid_method_output", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_output_times_that_do_not_match_the_request", "tests/test_execution.py::ExecutionTests.test_subclass_results_still_undergo_simulation_validation"]
+++

# Simulation boundary

The public call is `simulate(problem, method, request, *, execution=None,
options=None) -> States`. This baseline covers orchestration, not a particular
integrator's accuracy or convergence. Detailed ownership remains in the
[integration architecture](../../docs/simulation/integration-architecture.md)
and [executor contract](../../docs/simulation/execution.md).

## SIM-001: Validated problem snapshot

`InitialValueProblem(dynamics, initial_state, layout=...)` binds compatible dynamics, a raw physical array and a structural layout. The initial vector must be finite, one-dimensional and contain complete particle blocks. The layout dimension matches the dynamics. State and layout snapshots are owned, with no retained provider identity. When omitted, the layout is selected for canonical GC/FC dimensions; other dimensions require an explicit layout. The initial time belongs exclusively to `request.SimulationRequest` and may be nonzero. Generators in `states.generators` return arrays. See [state contracts](states.md).

## SIM-002: Complete, increasing output schedule

`SimulationRequest` requires finite, increasing span endpoints and a finite,
positive, non-Boolean maximum step. Output times are a finite, strictly
increasing vector of at least two samples, including both endpoints. Times use
the dynamics' time convention (dimensionless characteristic periods for the
measured-potential model).

Endpoint roundoff may be snapped using
`16*eps*max(1, abs(t0), abs(tf))`. The normalized schedule must remain strictly
increasing. The request stores an owned, read-only schedule. This tolerance
validates the request; it does not relax the runner's output equality contract.

## SIM-003: One complete execution job

`simulate` validates its orchestration inputs and delegates one complete job to
`Execution.run`. Omitting the executor selects local execution; omitting options
selects NumPy/SciPy without importing JAX. The executor prepares the environment
and a fresh numerical run, selects the shared integration driver, and supplies
execution metadata before result construction. Fixed JAX methods use a compiled
device map; DOP853/Radau retain SciPy control with prepared JAX field evaluations.
Methods own numerical operations and controllers own accepted intervals/sampling.

`NumericalMethod` requires `new_run(problem, request, *, evaluated_dynamics=None)`
and `jax_mode` (unsupported, device_resident or hybrid_scipy_controller). Prepared
runs are fresh ready IntegrationMethod instances with independent resources.
Methods no longer expose integrate or _integrate_jax; CompiledFixedMethod and
methods._compiled are removed. A capability declaration does not bypass supported
compiled-type checks. Custom providers use the preparation contract; complete
external jobs use the executor extension boundary. Configurations remain reusable,
consumed runs finish on success or failure, and errors propagate without a local
fallback. simulate and executor signatures and numerical controls are unchanged.


The numerical base is defined in methods.base and exported as methods.IntegrationMethod.
Execution contains its own private host and compiled JAX loops, fixed control,
sampling and current snapshots. Controller/state extension types are exported
from execution. There is no separate integration package or compatibility import.

## SIM-004: Output agreement with the request and initial state

Every execution, including custom executors, must return completed `States`. The runner validates exact saved times, physical state size, layout dimension, independent initial snapshot and first saved column against the request and problem. It returns the same result object without conversion to another container. Incompatible output raises `ValueError`. Structural storage guarantees are specified by [SOL-001 through SOL-003](results.md).
