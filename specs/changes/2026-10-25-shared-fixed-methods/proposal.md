# Shared algorithms with prepared components

## Ownership and scope

Cover the existing fixed-step paths: Euler, RK4, Gauss--Legendre, SDIRK, HBVM,
ABBA/BM4 and generic palindromic compositions, retaining their current GC/FC
support. Keep formulas, tableaux, composition order, projection equations and
physical reconstruction in methods. Methods continue receiving functions and
independent initial data; they must not recover dynamics from callable owners.

Execution selects the array implementation and device, prepares components and
owns scheduling, compilation, sampling and finalization. A method does not
choose JAX, hold a device or provide a second JAX advancement algorithm.
Preserve the public method constructors and ordinary simulate calls.

Leave `methods/adaptive/scipy.py`, DOP853 and Radau algorithms, their controls,
sampling and existing hybrid execution unchanged. Shared infrastructure must
continue serving that route. Its existing JAX declaration may remain local to
the adaptive classes; it must not be required by the common method contract.

## Components prepare their own representations

Add internal `_as_jax()` preparation to supported potential and dynamics
classes. Reuse the existing explicit JAX classes. Dynamics prepares both its
source and effective potentials, sharing immutable samples and fitted splines
without repeating gyroaveraging or normalization. Preparation is reusable,
imports JAX lazily and never changes the source object or global JAX settings.

Add `JaxNewtonSolver` and `JaxBroydenSolver` in an explicitly imported optional
solver module. Ordinary solvers prepare those representations through
`_as_jax()`, preserving every configured control and supported combination.
Both implementations of each solver expose the same solve inputs and numerical
result meaning. Solvers own iteration, derivative approximation, linear solves,
convergence, damping and accounting; methods supply residuals and equation-specific
derivative assembly, corrections and norms. Keep one source of configuration,
fresh work arrays per solve and no persistent traced arrays.

The problem retains its original dynamics and initial data; standard potential,
dynamics and solver configurations are sufficient for either execution path.
Do not introduce a new restriction banning existing explicit JAX objects or
external CPU systems. Conversion is an optional capability of supported
components, not a requirement on every `DynamicalSystem`.

## Shared method operations

Use the same numerical step and residual functions for NumPy and JAX. Replace
forced NumPy conversions, in-place writes and extraction of runtime Python
scalars in those functions with array-preserving expressions and small existing
array helpers. Static coefficient construction and host-side input validation
may still use NumPy. Shapes and physical snapshots retain their existing roles.

Keep stage assembly in methods and algebraic iteration in solvers. Residual
callbacks must themselves work with the prepared arrays. Express fixed recipes
with static loops where suitable; encapsulate any required compiled iteration
outside the algorithm. Reuse `contracts.arrays` and backend-neutral records;
do not introduce a universal mathematical language or a physical service bundle.

Preserve numerical guards. Compiled kernels may return validity/convergence
data for execution to report as failures; they must not silently accept invalid
states. Observation and counters must work without mutable Python objects inside
compiled numerical steps. This includes HBVM counters and composition traces.

Retire fixed-method `jax_mode`, `methods._jax_dispatch`,
`methods._jax_options`, `methods.classical._jax_steps` and
`methods.extended.core._jax_projection` after migrating their consumers.
An execution-owned adapter may compile the common prepared step and handle
records, but may not reimplement stages or select alternate per-method formulas.
Keep one compatibility boundary, explicit unsupported-case errors, bounded
compilation reuse and the device-resident time loop. Never substitute a built-in
algorithm for an unknown override.

## Independent numerical choices

`ReducedProjection` solves for mu in R^(2N); `SimultaneousProjection` solves
for (u, v, mu) in R^(6N). These formulations keep their guesses, residuals,
Jacobian assembly and reconstruction. NumPy/JAX selection must not change the
chosen equations, Newton/Broyden algorithm or solver controls. Existing ABBA
choices and reduced-only BM4 behavior remain valid. See the
[equations and solver guide](../../../docs/simulation/initial-solvers-formulations.md).

For example, the caller retains the same method configuration:

```python
method = ABBA4Implicit(
    step=0.02,
    projection_formulation=ReducedProjection(),
    solver=NewtonSolver(
        absolute_tolerance=1e-13,
        relative_tolerance=1e-12,
        max_iterations=12,
    ),
)
result = simulate(problem, method, request,
                  options=ExecutionOptions(backend="jax", device="cpu"))
```

Retain component-major physical states (GC 2N, FC 4N), temporary doubled
workspaces, scalar time, physical units and Hamiltonian normalization. Preserve
signed composition weights, T operations, projection placement, family defaults,
derivative targets, dense versus declared particle blocks, full Broyden updates,
HBVM's matrix infinity norm and its first-residual-before-Jacobian behavior.
Passive energy, its signed quadrature and independent shadow samples remain
unchanged. Requested algebraic tolerances are not measured trajectory errors.

This proposal is ready for review. Concrete private callable signatures and
record representations can be refined within these boundaries. If sharing an
algorithm requires changing its numerical behavior, dropping supported features
or modifying the adaptive methods, revise the proposal before proceeding.
