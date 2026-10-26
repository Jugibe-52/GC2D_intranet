# Initial data, projection equations and algebraic solvers

`InitialConditions` owns the initial physical values and their coordinate layout.
`InitialValueProblem` associates those values with compatible dynamics and a temporal request. The
integration method supplies its step equations, a projection formulation builds
the selected constraint system, and an algebraic solver finds its root. Execution
continues to own the complete run and returns the same completed `States`.

## Constructing physical input

```python
from states import InitialConditions
from simulation import InitialValueProblem, SimulationRequest, simulate
from dynamics import GuidingCenterDynamics

initial = InitialConditions.gc(x=[0.2, 0.7], y=[0.3, 0.6])
request = SimulationRequest.uniform(t_span=(0.2, 1.2))
problem = InitialValueProblem(GuidingCenterDynamics(potential), initial, request=request)
```

The GC factory packs `[x, y]` blocks; `InitialConditions.fc` packs `[x, y, vx, vy]`.
Each component is a nonempty one-dimensional array of N particles. Values must be
finite and remain in the supplied physical units. `InitialConditions(values,
layout=...)` supports external structural layouts. Public `values`, `layout`,
`components()`, `positions()` and `velocities()` return independent data;
`particle_count` is immutable. GC has no stored velocity components.

A problem stores this immutable owner as `initial_conditions`; its existing
`initial_state` property still returns a fresh packed array. Pass no second layout
when supplying InitialConditions. The existing array constructor remains supported
and normalizes through the same owner. Geometry generators still return arrays;
wrap them with an explicit layout when using the object interface. The request
alone owns time. Internal integration snapshots and saved physical histories retain
their existing roles. Saved result archive formats are unchanged.

## State representations

`DoubleFormulation(physical_size)` is the only state formulation. It prepares two temporary physical copies through `lift`, `join_copies` and `split_copies`, preserving NumPy/JAX backends and complete FC velocity blocks. It stores only a positive vector size; it owns no dynamics, layout, time, momentum, solver or projection. Methods using ordinary physical arrays need no formulation; `SingleFormulation` is removed.

Every accepted snapshot is `IntegrationState(time, values)`: scalar time and physical `values` of shape `(d*N,)`. GC remains 2N and FC 4N with or without energy observation. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. Optional quadrature accompanies a sample without storing energy momentum. Generic copy support does not extend GC numerical methods to FC.

Execution supplies `f(t, z)`, independent initial data and an optional Jacobian to the numerical method. `advance(t, z, h)` returns physical `StepResult.state`, numerical work and optional quadrature. `EnergyObserver`, supplied through `simulate(..., observers=(EnergyObserver(),))`, owns a fresh passive accumulator for each run. Accepted steps update it once; each shadow sample starts independently from the accepted left endpoint. JAX carries a separate functional accumulator through the compiled loop.

Execution gives the energy observer H and `-partial_t H`. The observer evaluates saved physical H once; `states.energy.energy_diagnostics` combines supplied H and physical kappa without evaluating dynamics. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. The default initial kappa is zero; `EnergyObserver(initial_momentum=...)` accepts an explicit physical value per particle. Numerical archives require no migration; old pickled configured objects should be recreated.

## Choosing equations and solver independently

```python
from methods.projection import ReducedProjection, SimultaneousProjection
from solvers import NewtonSolver, BroydenSolver
from methods import ABBA4Implicit, GaussLegendre4

solver = NewtonSolver(absolute_tolerance=1e-12,
                      relative_tolerance=1e-11, max_iterations=40)
method = ABBA4Implicit(step=0.02, projection_formulation=ReducedProjection(), solver=solver)
result = simulate(problem, method)
gauss = GaussLegendre4(step=0.02, solver=solver)
```

`ReducedProjection` solves for mu in R^(2N); `SimultaneousProjection` solves for (u, v, mu) in R^(6N). Both implement `contracts.methods.projection.ProjectionFormulation` and own the initial guess, residual, Jacobian assembly and physical reconstruction. The method supplies its doubled composition and derivative; these equations belong to methods and work with NumPy/JAX. ABBA accepts either class or the existing `reduced_multiplier`/`simultaneous_state_multiplier` names; BM4 retains reduced equations. Gauss stage equations remain method-specific. The `methods.ProjectionFormulation` and `simulation.ProjectionFormulation` exports remain the string-name type alias; import the equation protocol from `methods.projection`.

NewtonSolver supports Gauss, SDIRK, HBVM, ABBA and BM4. BroydenSolver supports the
existing ABBA/BM4 projection routes. Explicit methods have no algebraic solver;
SciPy adaptive methods retain their own internal algorithms. The built-in classes
are immutable configurations with no per-run work arrays. Unsupported solver
classes, subclass overrides and unsupported method/solver combinations fail before
backend preparation. Each solve is independent, including repeated simulations.

`solver=` is the sole owner of `absolute_tolerance`, `relative_tolerance`, `max_iterations`, `jacobian_method`, `jacobian_relative_step`, `backtracking` and `iteration_observer`. The method-level `newton_*`, `nonlinear_solver` and `SolverOptions` routes are removed. Omission selects the family default: Gauss/SDIRK use 1e-14, 1e-13 and 20 iterations; ABBA/BM4/HBVM use 1e-13, 1e-12 and 12 iterations. HBVM defaults to backtracking. An explicit solver owns its complete configuration, including these defaults. Relative thresholds use physical state scales. HBVM retains its matrix infinity norm and evaluates the first residual before its Jacobian. Requested tolerances bound the accepted residual, not the trajectory error.

Shared implementations are `solvers.nonlinear`, `solvers.linear` and optional
`solvers.jax_nonlinear` (with iteration primitives in `solvers._jax`).
`NewtonSolver._as_jax()` prepares `JaxNewtonSolver`; `BroydenSolver._as_jax()`
prepares `JaxBroydenSolver`. Both inherit the original immutable configuration
and expose the same solve inputs and counters. Compiled nonconvergence returns
a flag to execution, which raises before exporting a completed result. Methods
supply equations, derivative assembly and norms; solvers own iteration,
differences, linear solves, damping and convergence. Fixed `advance` functions
are shared across array types, including the 2N and 6N projection equations. The private `methods._nonlinear`, `methods._linear` and
`methods._jax_common` modules are removed; internal callers import the canonical
implementations. No optional JAX dependency is imported by the default CPU path.
See [execution](execution.md) for lifecycle and backend ownership and
[projection mathematics](../models/abba4-implicit/tex/simultaneous-formulation.tex)
for the equivalence of the two equations.

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

## Time rows in saved results (2026-10-26)

`States.states` and its alias `values` have shape `(samples, state_size)`.
Use `states[i]` for the complete physical state at `t[i]`, `states[:, j]`
for one component through time, and `zip(t, states)` to iterate samples.
Construction and `record_history` require this orientation, even for square
arrays. Each physical vector keeps its component-major order.

`positions()`, `velocities()`, `components()` and `split()` still return
component arrays of shape `(particles, samples)`. The coordinate classes,
Hamiltonian evaluation, geometry functions, study reference arrays and
specialized CSV/NPZ/HDF5 study formats retain their component-first contracts.
Pass `solution.states.T` when those consumers need a packed history.
`select(indices)` selects rows; independent diagnostic axes are unchanged.

`save_solution` writes schema 2 time rows. `load_solution` converts schema 1
columns using the archive version, never shape inference. Old files are not
rewritten. Completed States pickles also carry a history version; older
States pickles with the supported restoration function decode as columns.
Compiled kernels may keep component-first work arrays and transpose once
at the recording boundary. Physical integration and tolerances are unchanged.
