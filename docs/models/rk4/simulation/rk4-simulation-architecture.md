# RK4: state formulations and execution

Four classical Runge--Kutta stages.

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

RK4 uses the **physical** rows. `track_energy=False` is the default;
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

The passive momentum uses the same four stage states and weights as the physical RK4 update.

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

By default, `simulate(problem, method, request)` delegates to `Execution.run`,
which prepares a fresh run via `new_run` and allocates one physical recorder.
The host worker advances and samples through the method's controller. Each
`advance` returns an internal state, work counters and accepted details.
The controller owns `IntegrationState(time, values)`. Requested samples are
extracted into physical coordinates and copied into `States`; step counters keep
their separate axis. The formulation diagnoses the saved physical history.
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

### Study presentation

RK4 study results hold numerical data and summaries. Import
`plot_gc_symplecticity_diagnostics`, `plot_rk4_convergence`, or
`animate_gc_symplecticity` from `visualization.gc_symplecticity` and pass the
completed result as the first argument. The corresponding result methods
`plot_diagnostics()`, `plot_convergence()`, and `animate()` have been removed.
Generalized-energy result plots similarly use
`visualization.energy.plot_generalized_energy_comparison(result)`. These
presentation functions preserve their former options and outputs; integration
and persistence are unchanged. See the
[migration table](../../../simulation/api-migration.md#study-result-visualization)
for the other study result views.

## JAX execution

```python
import jax
from contracts.execution_options import ExecutionOptions
from methods.classical.rk4 import RK4
from simulation.runner import simulate

jax.config.update("jax_enable_x64", True)
solution = simulate(
    problem, RK4(track_energy=True), request,
    options=ExecutionOptions(backend="jax", device="cpu"),
)
# Use device="gpu", device_index=0 on a configured JAX GPU installation.
```

`ExecutionOptions()` or omission preserves the existing SciPy/NumPy CPU path. The JAX
extra is optional; CPU integration never imports it. JAX requires explicit
float64 configuration and available hardware, with no automatic fallback.
`ExecutionOptions` describes resources separately from physical data and RK4 controls.

Both routes use the canonical RK4 stages and passive quadrature in
`methods/classical/_rk4_core.py`. The default controller retains its Python
lifecycle. For JAX, `Execution.run` prepares a fresh run and selects the common
fixed-step compiled driver in `execution/_jax_fixed.py`. Both paths call the same
GC/FC dynamics methods and physical equations in `dynamics/_equations.py`.
Execution preparation binds JAX field evaluation through `JaxPotential`, reusing
the existing SciPy spline coefficients without refitting.

The driver compiles the complete time loop using `jax.lax.scan`. Particle
operations are batched within each stage; stages and successive time steps
remain sequential. The accepted state and requested output buffer stay on one
device. Storage scales with requested samples, rather than all internal steps.
Off-grid samples use independent shortened RK4 maps from the accepted interval
start, with the same endpoint tolerance and priority as the CPU controller.
They never change the subsequent main trajectory. Every accepted and sampled
state contributes to a finite-value check, including unsaved main states.

Tracking uses the same four physical stage states for `-partial_t H` on the
device. Physical Hamiltonian histories are also evaluated there, then the
standard formulation constructs energy diagnostics after one synchronized
transfer of the completed output. The result remains an immutable NumPy
`States` with identical layouts, sampling and diagnostic definitions.
Execution metadata reports backend, device, index and the JAX device kind.

The compiled route supports the built-in `GuidingCenterDynamics` and
`FullCyclotronDynamics`, including gyroaveraging and time-dependent fields.
It rejects custom dynamics, method subclasses, `step_observer` callbacks and
`progress=True` explicitly. Use the CPU path for Python observation/progress.
All other built-in fixed methods share this driver; DOP853/Radau use an explicit
hybrid path. See the [execution guide](../../../simulation/jax-execution.md).
This route uses one device, not multi-GPU sharding.

### Reproducible test notebooks

The local, Git-ignored directory `notebooks/developements/rk4_execution/`
contains `calculation.ipynb` and `visualisation.ipynb`. The default short test
uses 256 particles in a seeded synthetic periodic field, 200 steps, passive
energy tracking and three alternating timing repetitions. It deliberately
tests backend equivalence rather than the long-time method-comparison protocol.
The calculation delegates composition to `studies.rk4_execution`, asserts
trajectory and energy agreement and saves all trajectories, parameters, raw
timings, software versions and field samples through canonical States archives.
The visualisation delegates plots to `visualization.rk4_execution` and only
loads saved data. Both derive the same destination with `solution_destination`:
bucket storage is the default; `storage="local"` is an explicit alternative.

First-call timing includes device preparation and compilation when required.
Warmed timings include complete integration, transfers, energy diagnostics and
States creation; input field preparation is excluded. Changing particle count
or output shapes can trigger recompilation. Reusing a kernel can make another
first call warm. CPU timing results do not establish GPU performance.

`tests/test_jax_rk4.py` compares GC/FC trajectories and energy diagnostics,
irregular and sparse samples, independent particles, a known time-dependent
shear with fourth-order convergence, optional imports, and explicit capability
failures. GPU equivalence runs only when a GPU is available. The study test
round-trips canonical archives and keeps visualisation independent of integration.

## Migration and verification

RK4 returns the canonical in-memory `States` through `simulation.runner`.
Optional persistence is a subsequent operation:
`diagnostics.persistence.save_solution(solution, destination, metadata=..., potential=...)`.
Omitting `destination` and supplying `experiment_path` and `run_id` selects the
configured data bucket by default. Local storage requires an explicit local
destination, optionally built by `solution_destination(..., storage="local")`.
The destination selects a local result directory or a configured rclone bucket
prefix. `load_solution` restores the GC/FC layout, diagnostics and optional
sampled potential independently of RK4, allowing a separate visualisation notebook
to consume a completed calculation. The small `studies.persistence_demo` example
stores the base field, `rho`, full RK4 request and synthetic-field recipe.
See the [persistence contract](../../../../src/diagnostics/README.md) for the
manifest, supported formats and transfer-failure behavior.

`state_extension="fully_extended"` no longer runs a time/momentum projection.
It raises explicit migration guidance. Use `track_energy=True` with spatial
projection. The historical full-state symplecticity study is retired because it
measured a different map; existing saved artifacts can still be read.

Tests in `tests/test_state_formulations.py` cover all 13 methods, both tracking
settings, particle batches, physical-only observers, adaptive control, per-particle
time alignment and energy normalization. Model tests retain order, projection,
Jacobian and nonlinear-solver checks. The pre-change physical trajectories are
also compared with the migrated implementations on short nonautonomous runs.

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

`Execution_Modal` implements this boundary for remote NumPy/SciPy or JAX CPU
integrations. Results return to the local machine for validation and subsequent
persistence; see the [Modal executor guide](../../../simulation/modal-execution.md).

The matched Poincare star study selects this canonical implementation with
`studies.poincare_rho_sweep.RhoStarConfig(method="RK4", ...)`. It uses the same
physical field, geometry, radial Hamiltonian normalization and complete-step grid
as the implicit-method studies. Set `coupling_frequency=0`, since RK4 has no
doubled copies; Newton controls do not participate in its explicit stages.
JAX CPU float64 execution retains the initial state and integer-cycle returns,
optionally adding folded coordinates after integration. Its scalar metadata must
not report nonlinear solves or projection. These cycle-only archives exclude
per-step arrays and do not certify independent trajectory accuracy.
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

`InitialValueProblem(dynamics, initial_state, layout=...)` captures a validated
physical array, particle count and independent layout. Later edits to the input
array do not change the prepared formulation. Physical layouts belong to
`formulations.layout`; compatible structural layouts require no inheritance. Built-in dynamics keep their physical parameters
immutable. See the shared [layout and ownership contract](../../../dynamics/protocols.md#physical-layouts-and-problem-ownership).


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

Fixed-step methods inherit `IntegrationMethod` and declare `jax_mode="device_resident"`.
The executor resolves the device and prepares the dynamics; a numerical family
adapter binds the method's controls into `contracts.compiled.CompiledStep`.
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

Initial data are packed NumPy arrays, including the output of `states.generators`. `formulations.layout` supplies GC/FC layouts; `request.SimulationRequest` owns the initial time and output schedule. Controllers manage the current internal `IntegrationState`; `states.States` stores physical samples and becomes the completed result. Several methods can share a formulation. See the [migration guide](../../../simulation/api-migration.md).

## Initial data and numerical components

The public problem constructor also accepts `states.InitialConditions`; existing
packed-array callers remain supported. See the [component interface](../../../simulation/initial-solvers-formulations.md)
for ownership, array shapes and unchanged request/result boundaries.
