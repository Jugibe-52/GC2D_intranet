+++
schema_version = 1
kind = "spec"
id = "simulation"
sources = ["src/contracts/methods/functions.py", "src/contracts/methods/quadrature.py", "src/diagnostics/methods/energy.py", "src/solvers/derivatives.py", "src/formulations/state.py", "src/states/initial.py", "src/solvers/nonlinear.py", "src/methods/projection.py", "src/simulation/runner.py", "src/problem.py", "src/request.py", "src/execution/execution.py", "src/execution/_jax.py", "src/dynamics/jax_dynamics.py", "src/dynamics/jax_coordinates.py", "src/methods/base.py", "src/execution/options.py", "src/contracts/arrays.py", "src/contracts/methods/base.py", "src/contracts/execution/controllers.py", "src/contracts/dynamics/protocols.py", "src/contracts/methods/maps.py", "src/contracts/methods/projection.py"]
documents = ["docs/simulation/initial-solvers-formulations.md", "docs/simulation/integration-architecture.md", "docs/simulation/execution.md", "docs/simulation/jax-execution.md"]

[[requirements]]
id = "SIM-001"
tests = ["tests/test_problem_request.py", "tests/test_independent_formulations.py", "tests/test_initial_solvers_formulations.py", "tests/test_core.py::SimulationTests.test_problem_and_request_validate_initial_state_and_time_span", "tests/test_state_ownership.py::StateOwnershipTests.test_problem_and_prepared_formulations_own_state_and_layout_snapshots"]

[[requirements]]
id = "SIM-002"
tests = ["tests/test_core.py::SimulationTests.test_problem_and_request_validate_initial_state_and_time_span", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_request_revalidates_normalized_endpoints"]

[[requirements]]
id = "SIM-003"
tests = ["tests/test_function_methods.py", "tests/test_solver_configuration.py", "tests/test_energy_observation.py", "tests/test_execution.py::ExecutionTests.test_default_and_explicit_executor_produce_the_same_solution", "tests/test_execution.py::ExecutionTests.test_result_preparation_and_finalization_failures_consume_only_the_run", "tests/test_compiled_step_contract.py::CompiledStepContractTests.test_fixed_and_hybrid_runs_finalize_the_allocated_result_once", "tests/test_execution.py::ExecutionTests.test_simulation_dispatches_one_complete_job_to_the_subclass", "tests/test_execution.py::ExecutionTests.test_local_executor_owns_preparation_and_drives_the_controller", "tests/test_execution.py::ExecutionTests.test_executor_failure_propagates_without_local_fallback", "tests/test_execution.py::ExecutionTests.test_external_provider_prepares_a_run_without_an_integration_entry_point", "tests/test_execution.py::ExecutionTests.test_controller_preparation_failure_consumes_only_the_fresh_run", "tests/test_jax_methods.py"]

[[requirements]]
id = "SIM-004"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_invalid_method_output", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_output_times_that_do_not_match_the_request", "tests/test_execution.py::ExecutionTests.test_subclass_results_still_undergo_simulation_validation"]
+++

# Simulation boundary

The public call is `simulate(problem, method, *, execution=None,
options=None, observers=()) -> States`. This baseline covers orchestration, not a particular
integrator's accuracy or convergence. Detailed ownership remains in the
[integration architecture](../../docs/simulation/integration-architecture.md)
and [executor contract](../../docs/simulation/execution.md).

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

## SIM-001: Validated problem snapshot

`InitialValueProblem(dynamics, initial_state, request=..., layout=...)` associates compatible
dynamics with `states.InitialConditions` or a packed physical array. The initial
vector is finite, one-dimensional and contains complete particle blocks; its
layout dimension matches the dynamics. InitialConditions owns copies of the
values and layout, remains immutable after pickle, and returns independent data.
A supplied object already carries its layout, so an additional layout argument
is rejected. Raw arrays normalize through the same owner, using the declared dynamics coordinates when omitted; otherwise an explicit layout is required.
`initial_conditions` exposes the immutable owner and `initial_state` returns a
fresh array. The initial time belongs exclusively to `request.SimulationRequest`
and may be nonzero. Geometry generators still return arrays. See the
[state contracts](states.md).

`request` references a validated `request.SimulationRequest`. Omitting the keyword
creates an independent uniform request from 0 to 1 with 11 samples.
`with_request(request)` reuses dynamics and the initial-value owner in another
problem without mutating the source. `simulate` and executors receive the problem
and method; numerical methods still receive the request separately at preparation.

## SIM-002: Complete, increasing output schedule

`SimulationRequest` requires finite, increasing span endpoints. Output times are a finite, strictly
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
selects NumPy/SciPy without importing JAX. Execution owns environment preparation,
fresh-run validation, running/finished transitions, one States allocation and
one finalization. Internal host and compiled workers fill the open recorder and
return numerical diagnostics. Fixed JAX uses bulk recording; DOP853/Radau retain
SciPy control with JAX field callbacks prepared in execution._jax. Methods own
numerical operations and controllers own accepted intervals/sampling. Physics
modules have no dependency on execution. Standalone JaxPotential remains usable
with its own evaluation guards and caches.

Private backend, observer and numerical-run preparation helpers live in
`execution.execution`. Backend preparation resolves the mode and device once
and binds the prepared dynamics directly; other helpers read `problem.request`.
They return typed results without storing job state on the executor.
`Execution.run` retains the lifecycle and recording boundary and calls
`execution._jax_step.prepare_step` directly inside its `try/finally`, before
result allocation. Step preparation owns fixed-JAX observation restrictions and
the supported-run guard; the compiled worker uses `kernel.device`.
Optional imports, observer behavior and metadata precedence are preserved.
See the [execution guide](../../docs/simulation/execution.md).

`NumericalMethod.new_run(f, initial_conditions, request, *, jacobian=None, jacobian_structure="dense", observe=False, capture_quadrature=False)` receives a physical vector field and independent initial data. Execution resolves dynamics capabilities and supplies any physical Jacobian explicitly. The prepared run retains no `InitialValueProblem` or `DynamicalSystem`. `jacobian_structure` distinguishes dense matrices from independent particle blocks; it does not identify a physical model.

`new_run` returns a fresh ready `IntegrationMethod`, copying constructor configuration and initializing owned physical snapshots. `initialize`, `advance`, `build_observation` and `controller` define its numerical behavior. Methods have no energy accumulator or history diagnosis. Fixed methods share one `advance` implementation across array types. Execution owns JAX compatibility checks and compilation; `NumericalMethod` has no backend declaration. The adaptive implementation retains its local hybrid declaration. CPU extensions can inherit `FixedStepMethod` or supply their own controller.

Execution registers `observers=()` separately from the method. `EnergyObserver()` is immutable configuration for a fresh per-run accumulator; ordinary callbacks keep their supplied identity and caller-owned lifecycle. Execution attaches the original dynamics identity to accepted events and calls each callback once per main step, in registration order. Shadow samples do not emit accepted events. A callback declaring `requires_quadrature=True` requests current-step quadrature without enabling saved energy histories.

Configurations remain reusable; consumed runs finish after success or failure and errors propagate without a local fallback. Algebraic controls belong exclusively to `solver=...`; execution accepts the separate `observers=()` argument.


The numerical base is defined in methods.base and exported as methods.IntegrationMethod.
Execution contains its own private host and compiled JAX loops, fixed control,
sampling and current snapshots. Controller/state extension types are exported
from execution. There is no separate integration package or compatibility import.

## SIM-004: Output agreement with the request and initial state

Every execution, including custom executors, must return completed `States`. The runner validates exact saved times, physical state size, coordinate order and roles, independent initial snapshot and first saved column against the request and problem. It returns the same result object without conversion to another container. Incompatible output raises `ValueError`. Structural storage guarantees are specified by [SOL-001 through SOL-003](results.md).

## Configurable nonlinear resolution

Local implicit methods accept immutable `solver=` configurations. Explicit objects
supply their tolerances and limits; omission preserves existing defaults. Newton
supports Gauss, SDIRK, HBVM and projections; Broyden supports ABBA/BM4 projections.
Unsupported solver classes and combinations fail explicitly. Each solve has its
own work state. ABBA accepts ReducedProjection/SimultaneousProjection or legacy
names; their shared equations own 2N/6N unknowns, residuals and reconstruction in
both CPU and JAX paths. Physical scales, work counters, HBVM norms/backtracking,
particle blocks and independent shadow samples retain their numerical meaning.
See [component interface](../../docs/simulation/initial-solvers-formulations.md).

## Uncoupled BM4 and independent C

Both BM4 constructors omit coupling_frequency and bind uncoupled CPU/JAX maps.
Their weights, endpoint clock and projections stay unchanged. Independent C
remains available through methods.extended.maps.HarmonicCoupling. Legacy calculation
rates must be zero; stored coupled results retain their original metadata.
See [the migration guide](../../docs/models/extended/simulation/uncoupled-bm4.md).

## Explicit palindromic operations

All ABBA/BM4 presets inherit PalindromicMethod. Composition is a public tuple[Operation, ...] type alias. Recipes contain explicit A/B/T/C operations with signed full-step coefficients; only T advances the clock. Generic PalindromicImplicit and PalindromicMidpoint accept validated recipes for experiments. Existing presets merge consecutive T operations and keep their mathematical maps, grouped observations and solver controls. Fixed recipes are tuple literals on the method classes, shared by variants without dataclass fields. Preparation validates tuples once; names belong to the method. Observation adapters reconstruct groups from elementary traces without replay. Elementary ABBA2/ABBA4/BM4 counts are 5/15/30; clock rounding may differ from separate additions. Generic C requires an explicit operation and HarmonicCoupling; BM4 stays uncoupled. See [the operation architecture](../../docs/models/extended/simulation/palindromic-operations.md). ABBA2Implicit, ABBA4Implicit and BM4Implicit directly inherit ProjectedPalindromicMethod, including preparation and advancement. ABBA4 declares a complete tuple and retains one exterior projection. ABBA6 and ABBA-specific intermediate bases are removed.

## Preparation result types

Source result assignments follow the [typing convention](../../docs/development/typing.md).
Backend preparation returns `tuple[DynamicalSystem, Device | None, str, dict[str, DiagnosticValue]]`;
device resolution returns `jax.Device`. The coordinator narrows the optional
device before fixed-step preparation. Type-only imports preserve optional JAX
loading and existing execution behavior.

## Input owners and array compatibility

`problem.InitialValueProblem` owns the validated physical problem and remains
exported from `simulation`. Backend/device configuration belongs to
`execution.options.ExecutionOptions`, re-exported from `execution`. The retired
contract module paths have no aliases. Import order and new-object serialization
are checked by `tests/test_package_layout.py`.

`contracts.arrays` retains the shared NumPy/JAX compatibility rules, including
mixed arrays, nested sequences and tracers, without device selection or eager
JAX imports for generic formulation operations. Physical coordinate operations
instead select an explicit library: `PhysicalCoordinates` uses NumPy and
`JaxGCCoordinates` and `JaxFCCoordinates` from `dynamics.jax_coordinates` override only the
`array_namespace: ClassVar[ModuleType]` attribute with `jnp` instead of `np`.
The module is selected at class definition, without a property or per-operation
input inspection. JAX classes are imported from their explicit modules; base
imports stay independent of JAX. Shared validation, fixed conventions and shapes
are unchanged.

`dynamics.jax_dynamics.JaxGuidingCenterDynamics` and `JaxFullCyclotronDynamics`
inherit their complete physical implementations and only declare JAX coordinates.
Shared methods use `self.coordinates` and its namespace, including GC Jacobians
and FC split flows. Execution prepares these exact classes with `JaxPotential`,
reusing existing physical data and splines without repeating gyroaveraging or
mutating the source. Already compatible objects and bounded compilation caches
are reused. Arbitrary subclass overrides remain unsupported by JAX execution.
Direct compiled calls require both the JAX dynamics class and `JaxPotential`.
Initial conditions and results retain structural NumPy layouts and array storage.
`tests/test_shared_array_dynamics.py` and `tests/test_potential_execution.py`
cover explicit backends, inherited methods, numerical equivalence and preparation.

## Domain contract ownership

Existing pure interfaces live under `contracts` in the same domain/subdirectory
hierarchy as `src`; shared records follow their consuming process or method family.
See the [contract map](../../docs/simulation/contracts.md). Implementations and
numerical operations remain in their original domains. Public facades re-export
canonical contracts, and internal consumers import their defining modules.

## Independent state formulation and method preparation

`DoubleFormulation(physical_size)` is the only state formulation. It prepares two temporary physical copies through `lift`, `join_copies` and `split_copies`, preserving NumPy/JAX backends and complete FC velocity blocks. It stores only a positive vector size; it owns no dynamics, layout, time, momentum, solver or projection. Methods using ordinary physical arrays need no formulation; `SingleFormulation` is removed.

Every accepted snapshot is `IntegrationState(time, values)`: scalar time and physical `values` of shape `(d*N,)`. GC remains 2N and FC 4N with or without energy observation. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. Optional quadrature accompanies a sample without storing energy momentum. Generic copy support does not extend GC numerical methods to FC.

Execution supplies `f(t, z)`, independent initial data and an optional Jacobian to the numerical method. `advance(t, z, h)` returns physical `StepResult.state`, numerical work and optional quadrature. `EnergyObserver`, supplied through `simulate(..., observers=(EnergyObserver(),))`, owns a fresh passive accumulator for each run. Accepted steps update it once; each shadow sample starts independently from the accepted left endpoint. JAX carries a separate functional accumulator through the compiled loop.

Execution gives the energy observer H and `-partial_t H`. The observer evaluates saved physical H once; `states.energy.energy_diagnostics` combines supplied H and physical kappa without evaluating dynamics. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. The default initial kappa is zero; `EnergyObserver(initial_momentum=...)` accepts an explicit physical value per particle. Numerical archives require no migration; old pickled configured objects should be recreated.
