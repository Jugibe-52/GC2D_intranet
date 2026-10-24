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
    problem, BM4Implicit(), request,
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
| ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Implicit | Composition, residuals, Newton/Broyden, Hairer projection, energy and time loop | Final result export |
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
splines. Preparation binds the shared equations to `JaxPotential` instances
sharing those knots and coefficients without refitting or mutating the source. Built-in GC dynamics are supported by every
method. FC dynamics are supported by the classical and adaptive methods;
ABBA/BM4 retain their existing planar-GC restriction. Custom dynamics and method
subclasses are rejected rather than silently replacing their equations.

## Nonlinear and projection contracts

The device kernels reuse the canonical Gauss/SDIRK/HBVM tableaux, ABBA/BM4
composition coefficients and GC coupling matrices. RK4 shares its actual stage
and quadrature function with the CPU implementation. A common device Newton
controller handles convergence, iteration limits and HBVM backtracking.

Newton solves exploit independent particle blocks. Analytic GC Jacobians use
the potential Hessian; finite differences use the configured centered increment.
A finite-difference probe perturbs the same component in every independent
particle simultaneously. It obtains the diagonal particle blocks without
forming an O(N²) finite-difference matrix. HBVM's field-evaluation counters count
these actual batched calls, so its finite-difference call counts can be lower
than the CPU implementation's per-global-column probes. The result records
`finite_difference_batching="independent_particle_columns"` for that selection.

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

Fixed methods inherit `IntegrationMethod` and declare `jax_mode="device_resident"`;
adaptive methods declare `"hybrid_scipy_controller"`. The executor resolves the
device and prepares shared GC/FC dynamics, then calls a typed family adapter in
`methods._jax_dispatch`. The adapter binds numerical kernels and immutable controls;
explicit, classical implicit and composition methods have separate option types.
Layouts and dynamics preserve the array backend, so compiled kernels use the same
field, energy and analytic-Jacobian equations as NumPy. Device preparation belongs
to execution. Arithmetic projection carries no nonlinear controls. Kernel selection
is complete before tracing, with explicit rejection of unsupported subclass overrides.

The compiled worker receives `contracts.methods.compiled.CompiledStep`, which binds a physical map, explicit functions and numerical controls on the resolved device. Its call returns `(state, quadrature, statistics, valid)`. Execution owns the optional energy accumulator independently: it evaluates rates with `vmap` and reduces signed weighted stages in their original order with `lax.scan`, preserving the quadrature reduction scale. No host callback or complete stage history is required. `execution._jax_fixed` owns scheduling, sampling, convergence checks and bulk physical export. `Execution.run` owns recorder allocation, run status and finalization. Prepared maps snapshot controls; exact-type adapters reject unsupported method overrides before tracing.

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
