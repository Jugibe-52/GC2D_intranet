# GaussLegendre4: state formulations and execution

Two coupled Gauss stages, solved in physical coordinates.

The canonical theoretical source is [theory.tex](../tex/theory.tex), with its
compiled [theory.pdf](../tex/theory.pdf). The four-formulation convention below
supersedes earlier diagrams showing a duplicated clock or full time/momentum projection.

## Responsibilities

| Component | Owns |
|---|---|
| Dynamics | Physical vector field, Hamiltonian and required derivatives |
| Formulation | Internal coordinates, spatial copies and physical/energy extraction |
| Method | Stages, signed coefficients, spatial projection and passive quadrature |
| Integration | Accepted intervals, sampling, observation and output collection |

## State contract

| Planar one-particle formulation | Internal coordinates | Dimension |
|---|---|---:|
| Physical | `(x, y)` | 2 |
| Physical with energy | `(x, y, t, kappa)` | 4 |
| Duplicated | `(x1, y1, x2, y2)` | 4 |
| Duplicated with energy | `(x1, y1, x2, y2, t, kappa)` | 6 |

GaussLegendre4 uses the **physical** rows. `track_energy=False` is the default;
`track_energy=True` enables the energy row. With N planar particles its internal
dimensions are 2N / 4N. There are N time entries and N energy momenta.
Classical FC runs use their actual physical size 4N, giving 4N / 6N.
All components remain component-major; energy states append N times, then N
normalized momenta. Every component block has the same particle dimension.
The time entries are copies of the integration time; physical maps still receive
one scalar time. The formulation owns clock validation and output alignment.
The physical output always has its original size.
Accepted duplicated copies are equal. They separate only inside the numerical map.

`PhysicalFormulation` and `DoubledFormulation` are constructed directly from the
problem, initial time and tracking flag. They are defined in
`src/formulations/state.py`. BM4 additionally uses directly bound
`GCDoubledMaps` for its spatial direct/adjoint stages; its legacy configuration
factory is only a compatibility entry point.

## Energy and nonlinear work

The two converged Gauss stage states supply the passive momentum quadrature.

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

Scalar constructor controls use the typed private helpers in
`methods._validation`. The method retains its option names, normalization order
and accepted scalar inputs. Checks for invalid intermediate states, singular
systems and convergence remain next to the numerical operations.

The field and Jacobian helpers in `src/methods/classical/_jacobians.py` are
shared with SDIRK: field validation, centered finite differences, analytic
particle-block validation and automatic Jacobian selection. The coupled Gauss
stage equations and corrections remain in `gauss_legendre.py`.

`simulate(problem, method, request)` creates a fresh run via `new_run`, validates
its formulation and calls the shared `integrate_method`. Each `advance` returns
an internal state, small work counters and method-specific accepted details.
The common collector retains samples and counters; the formulation extracts the
physical trajectory and diagnostic histories. Run resources are isolated.

Fixed methods use independent shortened maps for off-grid samples. Adaptive
methods retain one live SciPy solver whose state is always physical. Their
energy quadrature follows accepted dense output and cannot affect the error norm.
Radau Jacobians are physical-sized even when energy tracking is enabled.

Observers receive the physical map and independent snapshots. Their shapes do
not change with tracking. All energy histories have shape `(N, saved_times)`,
including `extended_time` even for a single particle. Diagnostic arrays
`extended_time`, `extended_momentum`,
`physical_hamiltonian`, `generalized_energy` and `generalized_energy_error` follow
`Solution.t`; nonlinear and runtime work arrays follow `step_times`.
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
quadrature remain on that device until final NumPy `Solution` export. Its
coefficients, physical-state tolerance scaling and sampling conventions are
unchanged. JAX rejects unconverged solves explicitly. Diagnostics report
`execution_mode="device_resident"`. Python step observers and progress reporting
require the default CPU path.

Import `ExecutionOptions` from `contracts.execution_options`. Omission preserves existing
SciPy/NumPy execution. See the [shared execution contract](../../../simulation/jax-execution.md)
for supported physical systems, solver details, timing scope and validation.

## Complete integration executor

`simulate(problem, method, request, execution=Execution())` delegates the complete
integration to `Execution.run`. Import the executor with `from execution import
Execution`. It uses the existing local method implementation and does not
configure runtime resources. A subclass can replace the whole integration while
`simulate` retains final `Solution` construction and validation. Backend choices
remain separate in `options=ExecutionOptions(...)`. See the
[executor contract](../../../simulation/execution.md) for the extension interface.
The numerical stages, projection equations and integration controllers described
above remain inside this execution boundary.

`Execution_Modal` implements this boundary for remote NumPy/SciPy or JAX CPU
integrations. Results return to the local machine for validation and subsequent
persistence; see the [Modal executor guide](../../../simulation/modal-execution.md).

The matched Poincare star study selects the two-stage fourth-order method with
`studies.poincare_rho_sweep.RhoStarConfig(method="GaussLegendre4", ...)`.
Its analytic Newton solver uses the same explicit controls as the BM4Implicit study:
absolute tolerance `1e-12`, relative tolerance `1e-11`, at most 40 corrections and
Jacobian relative step `cbrt(eps)`. The method's constructor defaults are unchanged.
Set `coupling_frequency=0`, since the physical formulation has no doubled copies.
JAX CPU float64 execution saves the initial state and integer-cycle returns with
scalar two-stage and solver metadata; per-step diagnostics are excluded from these
cycle-only archives. Folded coordinates are postprocessing of the retained
unwrapped positions. No independent trajectory-accuracy reference is implied.
The radial Poincare study also supports explicit additional initial positions via
`studies.poincare_gap_probes.GapProbeSeeds` and `prepare_gap_probes`. These use
the same canonical method, physical dynamics, complete-step grid and JAX/Modal
execution as the original star. Probe archives retain their own seed geometry
and IDs and are saved separately; viewers append their coordinates after checking
the field fingerprint, scientific settings and aligned cycle times. The eight
gap probes have IDs 41–48, with two positions per gap selected at rho=0.30 and
reused for all eleven rho values. See the
[batch and persistence protocol](../../../simulation/modal-execution.md#additional-particles-inside-saved-poincare-gaps).

## Initial-state ownership

`InitialValueProblem` captures the validated physical state, particle count, and
independent layout at construction. A later edit to the original initial-state
provider does not change this run's formulation. Physical layouts are owned by
`contracts.state_layout`; compatible external providers need no inheritance from
initial-condition classes. Built-in dynamics keep their physical parameters
immutable. See the shared [layout and ownership contract](../../../dynamics/protocols.md#physical-layouts-and-problem-ownership).

## Shared Newton control and iteration callbacks

The stage equations, predictors, and method-specific corrections remain local.
`methods._nonlinear._solve_newton` owns the common stopping rule, correction
limit, residual evaluation count, and optional iteration callback. Tolerances,
initial evaluations, and acceptance thresholds retain their existing meaning;
HBVM keeps its distinct matrix residual norm and backtracking algorithm.

The optional keyword-only `newton_observer` receives
`contracts.nonlinear.NewtonIteration` records from each already evaluated
iterate, including the initial guess and final root. Gauss unknowns concatenate
its two physical stage vectors. SDIRK exposes one physical stage vector and its
zero-based `stage_index`; `time` and `duration` identify the complete physical
step. Every array in the record is an independent read-only copy. No additional
numerical evaluations are made. The callback is local to the method instance;
its exceptions propagate without modifying other runs.

Direct advances and off-grid output solves emit iterates. Physical `map_state`
callbacks provided to step observers omit Newton callbacks, preventing later
diagnostic replay from appending integration history. Compiled JAX and remote
Modal execution reject these optional Python callbacks.

Analytic particle corrections use `methods._linear._solve_particle_systems`
with matrices `(N,d,d)` and vectors `(N,d)`. Its explicit singleton
right-hand-side axis preserves batch semantics under NumPy 1.x and 2.x.


### Shared comparison records

The three-, four-, and five-method comparison campaigns share method construction,
alignment checks, and metric reductions in `studies._comparison`; their public
study entry points remain unchanged. Reference, accuracy, summary, and execution
records belong to `contracts.comparison`, and CSV readers reconstruct those same
record types without importing study orchestration. Energy-bound and parallel
BM4 archive records belong to `contracts.study_results`, with explicit re-exports
from their established study modules. See the
[record ownership contract](../../../simulation/integration-architecture.md#comparison-and-archive-record-ownership).

## Compiled fixed-step contract

Fixed-step methods inherit `CompiledFixedMethod` and provide a JAX adapter that
returns `contracts.compiled.CompiledStep`. The shared `integration/jax_fixed.py`
loop knows no concrete numerical method; each method owns its compiled stages
and statistics. See [JAX execution](../../../simulation/jax-execution.md).
