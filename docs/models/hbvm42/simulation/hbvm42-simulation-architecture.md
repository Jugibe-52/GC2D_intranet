# HBVM42: state formulations and execution

Four Gauss quadrature stages and a rank-two Legendre solve.

The canonical theoretical source is [theory.tex](../tex/theory.tex), with its
compiled [theory.pdf](../tex/theory.pdf). The physical-state contract below
supersedes earlier diagrams showing a duplicated clock or full time/momentum projection.

## Responsibilities

| Component | Owns |
|---|---|
| Dynamics | Physical vector field, Hamiltonian and required derivatives |
| Formulation | Size-checked temporary spatial copies |
| Method | Stages, signed coefficients, spatial projection and passive quadrature |
| Integration | Accepted intervals, sampling, observation and output collection |

## State contract

`SingleFormulation(physical_size)` and `DoubleFormulation(physical_size)` retain only a positive physical vector size. They prepare one or two temporary copies through `lift`, `join_copies` and `split_copies`. Shapes are `(d*N,*samples)` and `(copies*d*N,*samples)`; shifted copies remain distinct until the method projects them. NumPy/JAX backends and complete FC velocity blocks are preserved. There is no layout, particle count, tracking flag, clock, momentum, initial state, projection or solver in either representation.

Every accepted snapshot is `IntegrationState(time, values, momentum=None)`: scalar time, physical `values` of shape `(d*N,)`, and optional physical `momentum` of shape `(N,)`. GC remains 2N and FC 4N with or without tracking. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. This generic support does not extend GC numerical methods to FC.

The method initializes zero momentum, prepares copies, selects projection equations and solver, and integrates momentum from accepted stages. `advance(t, Z, h, momentum=kappa)` returns physical `StepResult.state` and separate `StepResult.momentum`. `StepInfo.momentum_before` records the incoming auxiliary. Shadow samples start from the same accepted physical state and momentum. JAX carries separate arrays through the compiled loop.

Methods evaluate saved physical H once. `states.energy.energy_diagnostics` combines supplied H and physical kappa without dynamics evaluation. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. Numerical archives require no migration; old pickled configured objects should be recreated.

## Energy and nonlinear work

The four converged stage states and Gauss weights integrate the passive momentum. Its value never enters the Legendre coefficient solve.

The stored momentum is physical `kappa`, initialized at zero. Its derivative is
`-partial_t H`; splitting sums are normalized by one half. The diagnostic is
`H(t, z) + kappa - H(t0, z0)`. It measures a balance, not conservation of the
time-dependent physical Hamiltonian. Dynamics must implement
[`HamiltonianSystem`](../../../dynamics/protocols.md) when tracking is enabled.
It extends `DynamicalSystem` with `hamiltonian` and
`extended_momentum_derivative`, including an explicit zero derivative for an
autonomous Hamiltonian. With tracking disabled, only `DynamicalSystem` is
required for this energy choice; method-specific capabilities still apply.

Only spatial coordinates enter a Hairer constraint. The reduced multiplier has
2N components; the ABBA simultaneous spatial solve has 6N unknowns. Clock and
momentum never enlarge these roots or affect their stopping scale. Tracking also
leaves classical physical solves and adaptive acceptance decisions unchanged.
Passive energy quadrature and optional observer reconstruction are extra work
outside the physical solver counters; reported wall time still includes them.

## Lifecycle and output

Scalar constructor controls use typed private helpers in `methods._validation`.
HBVM's local Jacobian resolver selects `auto` from dynamics capabilities without
probing analytic derivatives. Resolution still occurs during each physical step,
after the initial predictor and residual evaluations; preparation and evaluation
counts are unchanged. Convergence and linear-solve guards remain in the algorithm.

By default, `simulate(problem, method, request)` delegates to `Execution.run`,
which prepares a fresh run via `new_run` and allocates one physical recorder.
The host worker advances and samples through the method's controller. Each
`advance` returns a physical state and separate momentum, work counters and accepted details.
The controller owns `IntegrationState(time, values, momentum=None)`. Requested samples are
extracted into physical coordinates and copied into `States`; step counters keep
their separate axis. The method evaluates the saved physical Hamiltonian; the formulation combines it with stored momentum for output.
Execution finalizes the same recorder once and closes the run on success or
failure. Configured methods remain reusable with isolated run resources.

Fixed methods use independent shortened maps for off-grid samples. Adaptive
methods retain one live SciPy solver whose state is always physical. Their
energy quadrature follows accepted dense output and cannot affect the error norm.
Radau Jacobians are physical-sized even when energy tracking is enabled.

Observers receive the physical map and independent snapshots. Their shapes do
not change with tracking. All energy histories have shape `(N, saved_times)`,
including `extended_time` even for a single particle. Diagnostic arrays
`extended_time`, `extended_momentum`,
`physical_hamiltonian`, `generalized_energy` and `generalized_energy_error` follow
`States.t`; nonlinear and runtime work arrays follow `step_times`.
`extended_momentum_normalization` is `physical_kappa` and `energy_error` is the
maximum absolute sampled balance error over all particles.

## Migration and verification

`state_extension="fully_extended"` no longer runs a time/momentum projection.
It raises explicit migration guidance. Use `track_energy=True` with spatial
projection. The historical full-state symplecticity study is retired because it
measured a different map; existing saved artifacts can still be read.

Tests in `tests/test_state_formulations.py` cover all 13 methods, both tracking
settings, particle batches, physical-only observers, adaptive control, per-particle
time alignment and energy normalization. Model tests retain order, projection,
Jacobian and nonlinear-solver checks. The pre-change physical trajectories are
also compared with the migrated implementations on short nonautonomous runs.

## Optional JAX execution

This model supports `simulate(..., options=ExecutionOptions(backend="jax", device="cpu"))`
(or a configured `device="gpu"`) with explicit float64. The complete fixed time
loop, particle stages, applicable nonlinear/projection solves and passive energy
quadrature remain on that device until final NumPy `States` export. Its
coefficients, physical-state tolerance scaling and sampling conventions are
unchanged. JAX rejects unconverged solves explicitly. Diagnostics report
`execution_mode="device_resident"`. Python step observers and progress reporting
require the default CPU path.

Import `ExecutionOptions` from `execution.options`. Omission preserves existing
SciPy/NumPy execution. See the [shared execution contract](../../../simulation/jax-execution.md)
for supported physical systems, solver details, timing scope and validation.

## Complete integration executor

`simulate(problem, method, request, execution=Execution())` delegates the complete
job to `Execution.run`. Import the executor with `from execution import Execution`.
The executor prepares the requested environment and a fresh numerical run, allocates one recorder, invokes the host or compiled worker and finalizes the result once. It owns run status on success and failure. It does not change thread counts, affinity
or global JAX settings. Subclasses can execute the same job elsewhere, while
`simulate` validates the completed `States`. Backend choices remain separate in
`options=ExecutionOptions(...)`. See the [executor contract](../../../simulation/execution.md).
Numerical stages, projection equations and step controllers retain their own
implementations inside this execution boundary. The base class is
`methods.base.IntegrationMethod`; time loops and sampling live inside `execution`,
with no separate integration package.

`Execution_Modal` now implements this boundary for remote NumPy/SciPy CPU
integrations. Results return to the local machine for validation and subsequent
persistence; see the [Modal executor guide](../../../simulation/modal-execution.md).

## Initial-state ownership

Dynamics declare immutable `PhysicalCoordinates(component_names, position_names=(), velocity_names=())` from `dynamics.coordinates`. GC declares `(x,y)`; FC declares `(x,y,vx,vy)`. Position and velocity roles are explicit. The structural protocol is `contracts.dynamics.coordinates.StateLayout`; external implementations require no inheritance.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a physical model from dimension 2 or 4. It checks dimension, component order and position/velocity roles, even for equally sized arrays. Initial conditions and results retain an immutable structural description and owned arrays, without retaining live dynamics or a potential. Generic packing, component views and role access support `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)` while preserving NumPy/JAX tracing. Incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. Saved numerical archives retain existing GC/FC schemas and can be interpreted without constructing dynamics or loading a field.

## Compiled fixed-step contract

Fixed-step methods inherit `IntegrationMethod` and declare `jax_mode="device_resident"`.
The executor resolves the device and prepares the dynamics; a numerical family
adapter binds the method's controls into `contracts.methods.compiled.CompiledStep`.
The executor starts the shared `execution/_jax_fixed.py` loop, which knows no
concrete numerical method. Each method retains its compiled stages and statistics.
See [JAX execution](../../../simulation/jax-execution.md).

## Potential evaluation boundary

`Potential.evaluate` uses NumPy/SciPy; `JaxPotential` inherits its representation
and overrides only `evaluate` with JAX. Simulation preparation binds the shared
GC/FC equations to JAX potentials sharing the existing splines, including the
gyroaveraged field, without refitting or mutating the source dynamics. Inherited
electric fields follow the potential class; `evaluate_grid` remains NumPy-only. See the
[shared potential contract](../../../dynamics/jax-potential-evaluation.md).

Dynamics declare immutable `PhysicalCoordinates(component_names, position_names=(), velocity_names=())` from `dynamics.coordinates`. GC declares `(x,y)`; FC declares `(x,y,vx,vy)`. Position and velocity roles are explicit. The structural protocol is `contracts.dynamics.coordinates.StateLayout`; external implementations require no inheritance.

## Initial data and numerical components

The public problem constructor also accepts `states.InitialConditions`; existing
packed-array callers remain supported. See the [component interface](../../../simulation/initial-solvers-formulations.md)
for ownership, array shapes and unchanged request/result boundaries.

This method accepts `solver=NewtonSolver(...)`; omission preserves all previous
stopping controls. Method-specific equations and Jacobians remain here; shared
iteration belongs to `solvers`. Broyden is not enabled for this method.
HBVM retains its matrix infinity norm and Newton backtracking.

## State representation and step preparation

See the state contract above and the [representation operations](../../../simulation/initial-solvers-formulations.md#state-representations).

## Integration step

This method owns its positive `step` (default 0.1 in runtime time units).
Complete steps retain that size; only the final accepted step is shortened.
`advance` receives the effective duration without modifying the configured value.
See the [shared scheduling contract](../../../simulation/execution.md#method-owned-integration-step)
for independent output sampling and the common CPU/JAX grid.
