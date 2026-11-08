# JAX execution across numerical methods

Every built-in numerical method can run through the same executor backend selection.
`simulate` delegates the complete job to an `Execution` (local by default),
independently of these options; see the [executor contract](execution.md).

```python
import jax
from execution.options import ExecutionOptions
from diagnostics import EnergyObserver
from methods.palindromic.bm4_implicit import BM4Implicit
from simulation.runner import simulate

jax.config.update("jax_enable_x64", True)
solution = simulate(
    problem.with_request(request), BM4Implicit(),
    observers=(EnergyObserver(),),
    options=ExecutionOptions(backend="jax", device="cpu"),
)
# On a configured JAX accelerator: device="gpu", device_index=0.
```

Omitting `options` or supplying `ExecutionOptions()` retains the established
SciPy/NumPy path. JAX is optional (`pip install -c constraints.txt -e '.[jax]'`)
and is never imported by default CPU execution. GPU operation also requires a
compatible accelerator installation. Float64 and device availability are checked
explicitly, including repeated runs with cached resources. No hardware fallback
or change of numerical method is performed.

## Execution scope

DOP853 and Radau are standalone SciPy reference solvers used directly by studies. They are no longer methods in `simulation` or `methods`, and there is no adaptive controller or SciPy/JAX hybrid path in execution. Fixed methods own `step`; requests own the interval and saved times. Host progress is configured with `options=ExecutionOptions(progress=True)`.

The fixed methods keep state, nonlinear work and output buffers on one selected
device until completion. `execution_mode="device_resident"` records this path.
Particles are processed in parallel within each stage; dependent stages,
Newton corrections and successive time steps remain sequential. This is
single-device execution, not CPU process pools or multi-GPU sharding.

All routes return the same immutable NumPy `States`, with component-major
physical states and canonical energy histories. Initial field preparation,
gyroaveraging and spline fitting stay on CPU. `Potential` owns the canonical
splines. Preparation creates `JaxGuidingCenterDynamics` or
`JaxFullCyclotronDynamics` with `JaxGCCoordinates` or `JaxFCCoordinates` and `JaxPotential`,
sharing those knots and coefficients without refitting, repeating gyroaveraging
or mutating the source. The subclasses only declare coordinates; all physical
methods are inherited. Built-in GC dynamics are supported by every
method. FC dynamics are supported by Runge--Kutta and HBVM methods;
ABBA/BM4 retain their existing planar-GC restriction. Custom dynamics and method
subclasses are rejected rather than silently replacing their equations.

## Nonlinear and projection contracts

All fixed routes call the same numerical `advance` functions, including stage
assembly, compositions, residuals and quadrature. The methods use array-preserving
operations; they do not import JAX or choose devices. `JaxNewtonSolver` and
`JaxBroydenSolver` inherit ordinary solver controls and expose the same solve
inputs. Their iteration loops and linear algebra run on the selected device.

Analytic derivatives use independent particle blocks. Dense finite-difference matrices retain cross-particle terms. Centered differences differentiate the
same complete field or map on both routes, probing every global coordinate.
JAX vectorizes these probes, recorded as
`finite_difference_batching="vectorized_global_columns"`. HBVM counts the same
logical field evaluations on both routes: five predictor calls, four calls per
residual and, for differences, `8 * state_size` calls per correction. This
replaces the former JAX-only simultaneous particle-column approximation, whose
counts were lower for multiple particles. The derivative target, increment,
accepted residual and physical equations are unchanged.

These complete finite-difference matrices require O(N²) storage. Analytic
declared particle derivatives keep the small independent linear systems,
including both Legendre coefficients per particle in HBVM.

Convergence is still global across the batch: tolerances use the original
physical-state infinity norm, and particles share the original residual stopping
criterion. HBVM retains its matrix row-sum residual norm and its damping limit.
Negative composition coefficients, stage evaluation times, reduced and
simultaneous projection equations remain unchanged. Shadow samples do not contribute accepted-step nonlinear counters.

Broyden retains the original **global** rank-one Jacobian update and dense solve.
Independent per-particle secant updates would change that iteration. Its field
calculations are batched, but Broyden therefore still needs O(N²) matrix storage;
Newton's analytic particle blocks are generally a better scaling choice when
that solver is scientifically appropriate. No solver is changed automatically.

The JAX driver checks every accepted and sampled state for finite values and
tracks nonlinear convergence even for unsaved main steps and shadow samples.
Failed solves raise an error without returning a successful `States`.
Nonlinear histories retain their existing names, shapes and legacy Newton aliases.
Floating-point reassociation may change the correction count at a tolerance
boundary; equivalence means compatible trajectories and converged residuals,
not universal bitwise equality.

## Lifecycle, callbacks and measurement

Execution selects the device and delegates physical conversion to
`Potential._as_jax()` and dynamics `_as_jax()`. These methods reuse explicit JAX
classes, immutable samples, fitted splines and the effective GC potential.
Bounded identity caches reuse prepared components without changing their sources.
Ordinary solvers prepare their optional JAX counterparts through `_as_jax()`;
all tolerances and strategies come from the original configuration.

`execution._jax_step.prepare_step` snapshots a supported fixed method, prepares
its solver, binds its residual closures and packages its ordinary `advance`
result. It contains no stage equations or per-family kernels. The shared
`NumericalMethod` contract requires no backend declaration. Compatibility checks
reject unknown overrides before compilation. The retired method-owned dispatch,
options and alternate JAX-step modules have no aliases.

JAX compiles the same fixed-step numerical maps and a physical-only accepted-state loop. A prepared step returns `(state, statistics, converged)`. Execution evaluates endpoint H and analytic positive partial_t H on the device, records requested histories and accepted-step diagnostics, then exports NumPy States. Initial data are explicit inputs to step/integration preparation. Python callbacks, iteration observers and progress reporting require host execution.

`Execution.run` calls `execution._jax_fixed.prepare_integration` before recorder
allocation and the JAX worker, passing the common `FixedStepPlan` already built
by `execution._plan.build_fixed_step_plan`. Preparation places fresh inputs with `jax.device_put`, and describes their shapes, dtypes,
weak types and sharding with `jax.ShapeDtypeStruct`. `_compiled_integrator`
binds the numerical map and optional energy functions to `_integrate`, applies
`jax.jit`, and completes `.lower(*signature).compile()` without running the
trajectory. The step and `lax.scan` loop are compiled together. JAX documents
this [explicit compilation interface](https://docs.jax.dev/en/latest/aot.html).

A bounded cache retains at most 16 executables, keyed by the map, energy
functions and input signatures, including placement. Fresh initial values and
sampling arrays belong to each `PreparedIntegration`, outside the cache.
Changing particle, step or sample counts, input types, device or static controls
can require another executable. `run_steps(prepared, saved)` executes it,
synchronizes with `jax.device_get`, validates finite values and convergence,
and records physical history and diagnostics. Both preparation and execution
stay inside the executor's lifecycle guard; failures finish the consumed run.
The prepared object is private and process-local; remote workers compile locally.

Tests cover physical-only accepted states, isolated preparation, independent shadow samples, particle-block Jacobians, endpoint-only energy evaluation, analytic time differentiation, and CPU/JAX agreement. Model tests retain convergence, projection and nonlinear-work checks. Independent SciPy reference tests cover the retired adaptive API boundary.

Measure the first call separately from warmed calls. Complete `simulate` timings
include input/output transfers, diagnostics and immutable result construction,
and are already synchronized at the NumPy boundary. Compare the same physical
problem, numerical settings, saved times, energy options and particle count.
Compilation can repeat when shapes, methods or static controls change. GPU
speedups must be measured on the intended GPU; CPU JAX results do not establish
them. The RK4 calculation/visualisation notebook pair remains a reproducible
256-particle example of this timing protocol.

## Validation
