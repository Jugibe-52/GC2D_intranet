# Initial data, projection equations and algebraic solvers

`InitialConditions` owns the initial physical values and their coordinate layout.
`InitialValueProblem` associates those values with compatible dynamics. The
integration method supplies its step equations, a projection formulation builds
the selected constraint system, and an algebraic solver finds its root. Execution
continues to own the complete run and returns the same completed `States`.

## Constructing physical input

```python
from states import InitialConditions
from simulation import InitialValueProblem, SimulationRequest, simulate
from dynamics import GuidingCenterDynamics

initial = InitialConditions.gc(x=[0.2, 0.7], y=[0.3, 0.6])
problem = InitialValueProblem(GuidingCenterDynamics(potential), initial)
request = SimulationRequest.uniform(t_span=(0.2, 1.2))
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

`SingleFormulation(physical_size)` and `DoubleFormulation(physical_size)` retain only a positive physical vector size. They prepare one or two temporary copies through `lift`, `join_copies` and `split_copies`. Shapes are `(d*N,*samples)` and `(copies*d*N,*samples)`; shifted copies remain distinct until the method projects them. NumPy/JAX backends and complete FC velocity blocks are preserved. There is no layout, particle count, tracking flag, clock, momentum, initial state, projection or solver in either representation.

Every accepted snapshot is `IntegrationState(time, values, momentum=None)`: scalar time, physical `values` of shape `(d*N,)`, and optional physical `momentum` of shape `(N,)`. GC remains 2N and FC 4N with or without tracking. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. This generic support does not extend GC numerical methods to FC.

The method initializes zero momentum, prepares copies, selects projection equations and solver, and integrates momentum from accepted stages. `advance(t, Z, h, momentum=kappa)` returns physical `StepResult.state` and separate `StepResult.momentum`. `StepInfo.momentum_before` records the incoming auxiliary. Shadow samples start from the same accepted physical state and momentum. JAX carries separate arrays through the compiled loop.

Methods evaluate saved physical H once. `states.energy.energy_diagnostics` combines supplied H and physical kappa without dynamics evaluation. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. Numerical archives require no migration; old pickled configured objects should be recreated.

## Choosing equations and solver independently

```python
from methods.projection import ReducedProjection, SimultaneousProjection
from solvers import NewtonSolver, BroydenSolver
from methods import ABBA4Implicit, GaussLegendre4

solver = NewtonSolver(absolute_tolerance=1e-12,
                      relative_tolerance=1e-11, max_iterations=40)
method = ABBA4Implicit(step=0.02, projection_formulation=ReducedProjection(), solver=solver)
result = simulate(problem, method, request)
gauss = GaussLegendre4(step=0.02, solver=solver)
```

`ReducedProjection` solves for mu in R^(2N). `SimultaneousProjection` solves for
(u, v, mu) in R^(6N). Both implement `contracts.methods.projection.ProjectionFormulation`
and own the initial guess, residual, Jacobian assembly and physical reconstruction.
The method supplies its doubled composition and derivative; these equations belong to methods. Their array operations serve NumPy and JAX. ABBA accepts either
class or the existing `reduced_multiplier`/`simultaneous_state_multiplier` strings;
BM4 retains the reduced equations. Gauss stage equations remain method-specific.
The older `methods.ProjectionFormulation` and `simulation.ProjectionFormulation`
exports remain the string-name type alias; import the equation protocol from
`methods.projection` when defining its type.

NewtonSolver supports Gauss, SDIRK, HBVM, ABBA and BM4. BroydenSolver supports the
existing ABBA/BM4 projection routes. Explicit methods have no algebraic solver;
SciPy adaptive methods retain their own internal algorithms. The built-in classes
are immutable configurations with no per-run work arrays. Unsupported solver
classes, subclass overrides and unsupported method/solver combinations fail before
backend preparation. Each solve is independent, including repeated simulations.

An explicit `solver=` supplies its tolerances and iteration limit. When omitted,
all existing method defaults and tolerance keywords retain their meaning. The
relative threshold uses the physical state scale, not the projection multiplier.
Newton accepts a residual function and a Jacobian or specialized update operation;
Broyden receives its initial Jacobian estimate. Reuse cached residual payloads and
particle blocks. HBVM retains its matrix infinity norm and damped Newton updates;
other methods retain full Newton. Requested tolerances bound the accepted residual,
not the trajectory error.

Shared implementations are `solvers.nonlinear`, `solvers.linear` and optional
`solvers._jax`. The private `methods._nonlinear`, `methods._linear` and
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
