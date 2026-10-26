# JAX execution across numerical methods

Every built-in numerical method can run through the same executor backend selection.
`simulate` delegates the complete job to an `Execution` (local by default),
independently of these options; see the [executor contract](execution.md).

```python
import jax
from execution.options import ExecutionOptions
from diagnostics import EnergyObserver
from methods.extended.bm4 import BM4Implicit
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

| Methods | JAX work | Host work during integration |
|---|---|---|
| ExplicitEuler, RK4 | Batched fields, stages and complete fixed time loop | Final result export |
| GaussLegendre4, SDIRK4, HBVM42 | Fields, Jacobians, batched Newton solves, passive energy and time loop | Final result export |
| ABBA2Midpoint, BM4Midpoint | Signed composition, arithmetic projection, energy and time loop | Final result export |
| ABBA2Implicit, ABBA4Implicit, BM4Implicit | Composition, residuals, Newton/Broyden, Hairer projection, energy and time loop | Final result export |
| DOP853, Radau | Batched field and energy evaluations | SciPy adaptive controller, stages, rejection, dense output and Radau Newton/LU |

The fixed methods keep state, nonlinear work and output buffers on one selected
device until completion. `execution_mode="device_resident"` records this path.
Particles are processed in parallel within each stage; dependent stages,
Newton corrections and successive time steps remain sequential. This is
single-device execution, not CPU process pools or multi-GPU sharding.

DOP853 and Radau deliberately retain their existing SciPy numerical solvers.
They publish `execution_mode="hybrid_scipy_controller"`,
`adaptive_controller="scipy"`, and `backend="scipy"`, alongside the selected
`execution_backend="jax"` and device metadata. Each field evaluation crosses
the NumPy/device boundary. A user-supplied Radau Jacobian retains its original
host callback semantics. This route can be slower than CPU for small batches;
it does not claim that the complete adaptive solver runs on GPU.

All routes return the same immutable NumPy `States`, with component-major
physical states and canonical energy histories. Initial field preparation,
gyroaveraging and spline fitting stay on CPU. `Potential` owns the canonical
splines. Preparation creates `JaxGuidingCenterDynamics` or
`JaxFullCyclotronDynamics` with `JaxGCCoordinates` or `JaxFCCoordinates` and `JaxPotential`,
sharing those knots and coefficients without refitting, repeating gyroaveraging
or mutating the source. The subclasses only declare coordinates; all physical
methods are inherited. Built-in GC dynamics are supported by every
method. FC dynamics are supported by the classical and adaptive methods;
ABBA/BM4 retain their existing planar-GC restriction. Custom dynamics and method
subclasses are rejected rather than silently replacing their equations.

## Nonlinear and projection contracts

All fixed routes call the same numerical `advance` functions, including stage
assembly, compositions, residuals and quadrature. The methods use array-preserving
operations; they do not import JAX or choose devices. `JaxNewtonSolver` and
`JaxBroydenSolver` inherit ordinary solver controls and expose the same solve
inputs. Their iteration loops and linear algebra run on the selected device.

Analytic derivatives retain declared independent particle blocks; dense
functions retain cross-particle terms. Centered differences differentiate the
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
simultaneous projection equations, and physical-kappa normalization remain
unchanged. Shadow samples do not contribute accepted-step nonlinear counters.

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
not universal bitwise equality or identical adaptive step grids.

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

The compiled worker receives `contracts.methods.compiled.CompiledStep`, which binds a physical map, explicit functions and numerical controls on the resolved device. Its call returns `(state, quadrature, statistics, valid)`. Execution owns the optional energy accumulator independently: it evaluates rates with `vmap` and reduces signed weighted stages in their original order with `lax.scan`, preserving the quadrature reduction scale. No host callback or complete stage history is required. `execution._jax_fixed` owns scheduling, sampling, convergence checks and bulk physical export. `Execution.run` owns recorder allocation, run status and finalization. Prepared maps snapshot controls; exact-type compatibility checks reject unsupported method overrides before tracing.

`IntegrationMethod.new_run` receives prepared `f`, independent initial data and an optional Jacobian; it retains no dynamics object, ExecutionOptions or energy functions. Configured instances remain reusable. The fixed driver uses `lax.scan` for accepted steps and independent shortened maps for off-grid output. A shortened energy sample starts from the same accepted left accumulator and never changes it. Only requested states, auxiliary samples and small accepted-step statistics are retained. Bounded caches reuse compatible compiled functions; `JaxPotential` reuses concrete device coefficients without refitting splines or persisting traced constants.

Fixed JAX methods require `progress=False` and `solver.iteration_observer=None`. `observers=(EnergyObserver(),)` is supported as built-in configuration; arbitrary Python callbacks are rejected before compilation and remain available through the CPU controller. Hybrid adaptive methods retain normal progress and callback contracts because their controller already resides on the host.

Measure the first call separately from warmed calls. Complete `simulate` timings
include input/output transfers, diagnostics and immutable result construction,
and are already synchronized at the NumPy boundary. Compare the same physical
problem, numerical settings, saved times, energy options and particle count.
Compilation can repeat when shapes, methods or static controls change. GPU
speedups must be measured on the intended GPU; CPU JAX results do not establish
them. The RK4 calculation/visualisation notebook pair remains a reproducible
256-particle example of this timing protocol.

## Validation

`tests/test_jax_methods.py` covers every fixed and adaptive method, GC/FC,
analytic and finite-difference Jacobians, Newton/Broyden, both projection
formulations, harmonic coupling, energy tracking, irregular/sparse samples,
run isolation, custom Radau Jacobians, observers on the hybrid path and explicit
failure cases. All families have a GPU equivalence test which skips when no
accelerator is available. Existing method tests continue to validate order and
geometric identities of the canonical CPU algorithms; RK4 additionally has an
exact-solution fourth-order test for its compiled route.
