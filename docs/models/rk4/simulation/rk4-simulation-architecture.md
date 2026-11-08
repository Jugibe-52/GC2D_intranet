# RK4: state formulations and execution

Four classical Runge--Kutta stages.

The canonical theoretical source is [theory.tex](../tex/theory.tex), with its
compiled [theory.pdf](../tex/theory.pdf). The physical-state contract below
supersedes earlier diagrams showing a duplicated clock or full time/momentum projection.

![Functions, numerical method and observers](rk4-simulation-architecture.svg)

## Responsibilities

| Component | Owns |
|---|---|
| Dynamics | Physical vector field, Hamiltonian and required derivatives |
| Formulation | Optional doubled spatial copies |
| Method | Physical stages, projection equations and optional quadrature data |
| Solver | Nonlinear tolerances, iteration limits, derivatives and callbacks |
| EnergyObserver | Endpoint H and analytic positive partial_t H |
| Execution | Function injection, accepted intervals, sampling, observation and output collection |

## State contract

`DoubleFormulation(physical_size)` is the only state formulation. It prepares temporary copies through `lift`, `join_copies` and `split_copies`, preserving NumPy/JAX arrays and complete component blocks. Runge--Kutta and HBVM methods use the physical vector directly and instantiate no formulation. The doubled representation stores only a positive physical size; it contains no dynamics, solver or diagnostic state.

Every accepted snapshot is `IntegrationState(time, values)`: scalar time and a physical vector of shape `(d*N,)`. GC remains 2N and FC 4N with or without energy observation. Temporary doubled GC work has 4N entries; generic doubled FC work has 8N. These workspaces never become accepted states. This generic copy support does not extend GC numerical methods to FC.

`IntegrationMethod.new_run(f, initial_conditions)` prepares a fresh run without
an algebraic solver or Jacobian. `ImplicitIntegrationMethod` in `methods.implicit`
adds `new_run(f, initial_conditions, *, jacobian=None)` and the solver contract;
concrete families retain their defaults. Both bases are exported from `methods`.
Explicit methods use the common base and reject the `jacobian` keyword. There
is no separate explicit base. Execution supplies derivatives only to implicit
configurations, and JAX reads derivative settings only for those runs.

Both paths call `initialize(initial_conditions)` after binding their functions.
Initial data and layout are temporary inputs; the common base retains `f`,
metadata, status and copy count, while execution owns the initial snapshot.
Only implicit runs retain `jacobian`. Supplied derivatives have independent
particle shape `(N,d,d)`; finite differences remain dense. Solver strategies,
Broyden's projection matrix and deferred derivative validation are unchanged.
There is no `jacobian_structure`, `capture_quadrature` or method-level `progress` option.

`EnergyObserver()` records physical H and the positive analytic partial time derivative `hamiltonian_time_derivative` at accepted endpoints and requested physical samples. It does not consume internal stages or integrate momentum. Per-step diagnostics are `energy_start`, `energy_end`, `energy_change`, `energy_time_derivative_start` and `energy_time_derivative_end`, with shape `(steps,N)`. Saved histories `physical_hamiltonian` (alias `hamiltonian`), `energy_drift` and `hamiltonian_time_derivative` have shape `(N,T)`. Energy variation in a time-dependent Hamiltonian is not a numerical conservation error.

## Energy and nonlinear work

Specialized historical generalized-energy studies explicitly use stage diagnostics under `diagnostics.methods.stage_energy` and `studies._generalized_energy`. They retain their signed quadrature, physical kappa normalization and archive labels. Generic execution has no momentum accumulator. `States.k` and `States.err` remain readers for historical diagnostics and return `None` when absent. Old numerical archives remain readable; recreate configured-object pickles after the API change.

## Lifecycle and output

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
from diagnostics.methods.energy import EnergyObserver
from execution.options import ExecutionOptions
from methods.runge_kutta.rk4 import RK4
from simulation.runner import simulate

jax.config.update("jax_enable_x64", True)
solution = simulate(
    problem.with_request(request), RK4(),
    observers=(EnergyObserver(),),
    options=ExecutionOptions(backend="jax", device="cpu"),
)
# Use device="gpu", device_index=0 on a configured JAX GPU installation.
```

`ExecutionOptions()` or omission preserves the existing SciPy/NumPy CPU path. The JAX
extra is optional; CPU integration never imports it. JAX requires explicit
float64 configuration and available hardware, with no automatic fallback.
`ExecutionOptions` describes resources separately from physical data and RK4 controls.

Preparation explicitly compiles the complete time loop using `jax.lax.scan`
before `run_steps` executes it. Particle
operations are batched within each stage; stages and successive time steps
remain sequential. The accepted state and requested output buffer stay on one
device. Storage scales with requested samples, rather than all internal steps.
Off-grid samples use independent shortened RK4 maps from the accepted interval
start, with the same endpoint tolerance and priority as the CPU controller.
They never change the subsequent main trajectory. Every accepted and sampled
state contributes to a finite-value check, including unsaved main states.

DOP853 and Radau are standalone SciPy reference solvers used directly by studies. They are no longer methods in `simulation` or `methods`, and there is no adaptive controller or SciPy/JAX hybrid path in execution. Fixed methods own `step`; requests own the interval and saved times. Host progress is configured with `options=ExecutionOptions(progress=True)`.

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
`diagnostics.io.persistence.save_solution(solution, destination, metadata=..., potential=...)`.
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
It raises explicit migration guidance. Use `observers=(EnergyObserver(),)` with spatial
projection. The historical full-state symplecticity study is retired because it
measured a different map; existing saved artifacts can still be read.

Tests cover physical-only accepted states, isolated preparation, independent shadow samples, particle-block Jacobians, endpoint-only energy evaluation, analytic time differentiation, and CPU/JAX agreement. Model tests retain convergence, projection and nonlinear-work checks. Independent SciPy reference tests cover the retired adaptive API boundary.

## Complete integration executor

`simulate(problem, method, execution=Execution())` delegates the complete
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

Dynamics declare `GCCoordinates()` for fixed `(x, y)` blocks or `FCCoordinates()` for fixed `(x, y, vx, vy)` blocks, from `dynamics.coordinates`. `PhysicalCoordinates` is a stateless base with common array operations; dimensions are fixed properties and no names or role-index fields are configured. `split()` returns an ordinary tuple. `JaxGCCoordinates` and `JaxFCCoordinates`, from `dynamics.jax_coordinates`, inherit their respective operations and only select `jnp` through `array_namespace`.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a model from dimension. `StateLayout.snapshot()` explicitly supplies an immutable NumPy convention; `describe_coordinates` validates that snapshot and its dimension. Compatibility compares the snapshot class and equality, even for equally sized arrays. Initial conditions and results own their arrays and conventions without retaining live dynamics or a potential. Custom conventions derive from `PhysicalCoordinates` and implement fixed operations instead of role tables; external adapters may supply these snapshots without inheritance. Packing and views preserve `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)`. The respective JAX classes support tracing; incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. States stores time rows `(T,d*N)`, with `states[i]` at `t[i]`; component accessors retain `(N,T)`. New solution archives use schema 2; the reader converts schema 1 columns explicitly. Archives can be interpreted without constructing dynamics or loading a field.

## Compiled fixed-step contract

Fixed-step methods use the same `advance` implementation with NumPy or JAX
arrays. They contain no JAX imports, backend declaration or alternate stages.
Execution owns compatibility checks and device selection; `execution._jax_step`
binds an isolated prepared method to `contracts.methods.compiled.CompiledStep`.
Potential, dynamics and solvers own their internal `_as_jax()` preparation.
`Execution.run` first builds a shared fixed-step and sampling plan for host
and JAX execution. It passes this plan to `execution._jax_fixed.prepare_integration`
to place inputs and compile the complete loop before `run_steps` executes the
returned `PreparedIntegration` and records its results.
See [JAX execution](../../../simulation/jax-execution.md).

## Potential evaluation boundary

`Potential.evaluate` uses NumPy/SciPy; `JaxPotential` inherits its representation
and overrides only `evaluate` with JAX. Simulation preparation constructs
`JaxGuidingCenterDynamics` or `JaxFullCyclotronDynamics` from
`dynamics.jax_dynamics`; these subclasses only declare `JaxGCCoordinates` or `JaxFCCoordinates`
and inherit all physical methods. It supplies JAX potentials sharing the existing
splines, including the
gyroaveraged field, without refitting or mutating the source dynamics. Inherited
electric fields follow the potential class; `evaluate_grid` remains NumPy-only. See the
[shared potential contract](../../../dynamics/jax-potential-evaluation.md).

Dynamics declare `GCCoordinates()` for fixed `(x, y)` blocks or `FCCoordinates()` for fixed `(x, y, vx, vy)` blocks, from `dynamics.coordinates`. `PhysicalCoordinates` is a stateless base with common array operations; dimensions are fixed properties and no names or role-index fields are configured. `split()` returns an ordinary tuple. `JaxGCCoordinates` and `JaxFCCoordinates`, from `dynamics.jax_coordinates`, inherit their respective operations and only select `jnp` through `array_namespace`.

## Initial data and numerical components

The public problem constructor also accepts `states.InitialConditions`; existing
packed-array callers remain supported. See the [component interface](../../../simulation/initial-solvers-formulations.md)
for ownership, array shapes and unchanged request/result boundaries.

## State representation and step preparation

See the state contract above and the [representation operations](../../../simulation/initial-solvers-formulations.md#state-representations).

## Integration step

This method owns its positive `step` (default 0.1 in runtime time units).
Complete steps retain that size; only the final accepted step is shortened.
`advance` receives the effective duration without modifying the configured value.
See the [shared scheduling contract](../../../simulation/execution.md#method-owned-integration-step)
for independent output sampling and the common CPU/JAX grid.

## Tableau-driven implementation

`RK4` now inherits its numerical stage calculations from `ExplicitRungeKutta`
in `methods.runge_kutta`. The concrete recipe declares its immutable
`ButcherTableau(A, B, C)` and metadata. The shared `RungeKutta` base owns
physical step results and passive stage quadrature; solver and execution
responsibilities remain separate. See the [Runge–Kutta family contract](../../../simulation/runge-kutta.md)
for coefficient validation, generic construction and import migration.
