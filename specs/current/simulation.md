+++
schema_version = 1
kind = "spec"
id = "simulation"
sources = ["src/simulation/runner.py", "src/contracts/problem.py", "src/contracts/request.py", "src/execution/execution.py"]
documents = ["docs/simulation/integration-architecture.md", "docs/simulation/execution.md", "docs/simulation/jax-execution.md"]

[[requirements]]
id = "SIM-001"
tests = ["tests/test_core.py::SimulationTests.test_problem_and_request_validate_initial_state_and_time_span", "tests/test_state_ownership.py::StateOwnershipTests.test_problem_and_prepared_formulations_own_state_and_layout_snapshots"]

[[requirements]]
id = "SIM-002"
tests = ["tests/test_core.py::SimulationTests.test_problem_and_request_validate_initial_state_and_time_span", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_request_revalidates_normalized_endpoints"]

[[requirements]]
id = "SIM-003"
tests = ["tests/test_execution.py::ExecutionTests.test_default_and_explicit_executor_produce_the_same_solution", "tests/test_execution.py::ExecutionTests.test_simulation_dispatches_one_complete_job_to_the_subclass", "tests/test_execution.py::ExecutionTests.test_local_executor_passes_explicit_backend_options_to_the_method", "tests/test_execution.py::ExecutionTests.test_executor_failure_propagates_without_local_fallback"]

[[requirements]]
id = "SIM-004"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_invalid_method_output", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_output_times_that_do_not_match_the_request", "tests/test_execution.py::ExecutionTests.test_subclass_results_still_undergo_simulation_validation"]
+++

# Simulation boundary

The public call is `simulate(problem, method, request, *, execution=None,
options=None) -> Solution`. This baseline covers orchestration, not a particular
integrator's accuracy or convergence. Detailed ownership remains in the
[integration architecture](../../docs/simulation/integration-architecture.md)
and [executor contract](../../docs/simulation/execution.md).

## SIM-001: Validated problem snapshot

`InitialValueProblem` binds compatible dynamics and initial configuration
protocols. Its packed initial state is finite, one-dimensional, and contains
complete blocks for a positive number of particles. The layout's state
dimension matches the dynamics. The problem owns state and layout snapshots;
later edits to the original configuration do not change them. The source
configuration remains provenance, not the live owner of the problem's state.

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

`simulate` validates its orchestration inputs and delegates the complete job
to `Execution.run`. Omitting `execution` selects local execution. Omitting
backend options selects the default NumPy/SciPy path. Backend options are
separate from the executor and are forwarded without moving numerical methods
into the orchestration layer. A custom executor's failure propagates; the runner
does not silently repeat the integration locally.

## SIM-004: Output agreement with the request and initial state

Every execution result, including one from a custom executor, is converted to a
`Solution` and undergoes its structural checks. Saved times equal the requested
output times exactly, the physical state dimension equals the problem's initial
state size, and the first saved column equals that initial state exactly.
Incompatible output raises `ValueError`. Shapes and storage guarantees are
specified by [SOL-001 through SOL-003](results.md).

The built-in provider is `states.State`, an immutable instantaneous snapshot with an explicit layout; custom mutable structural providers remain supported. Generators live in `states.generators`. See [state contracts](states.md).
