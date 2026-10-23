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

`SingleFormulation(layout, particle_count, track_energy=False)` and
`DoubleFormulation(...)` describe one or two complete physical copies. They are
defined in `formulations.state` and exported by `formulations` and `simulation`.
The layout is copied; no problem, initial values, clock, dynamics or solver is
retained. The same classes support GC, FC and external structural layouts.

For N particles with d physical components, `lift(z)` takes `(d*N, *samples)`
and returns `(copies*d*N, *samples)`. `join_copies(*values)` packs explicitly
supplied copies, including `(z+mu, z-mu)` for implicit projection.
`split_copies` reads them without projecting. Invalid dimensions or mismatched
sample axes raise `ValueError`. NumPy outputs own independent copy blocks;
JAX inputs remain functional arrays, including during tracing.

`pack(spatial, time, momentum)` appends auxiliaries to already prepared copies.
Time and momentum each have shape `(N, *samples)`; time broadcasts to that shape.
`lift` does not append them. Single/double sizes are d*N/2*d*N, plus 2*N with
tracking. FC duplication includes positions and velocities, giving 8*N/10*N
entries. This does not extend GC-only numerical methods to FC.

```python
import numpy as np
from formulations import DoubleFormulation
from formulations.layout import FCStateLayout

representation = DoubleFormulation(FCStateLayout(), particle_count=2,
                                   track_energy=True)
z = FCStateLayout.pack_components([0.2, 0.7], [0.3, 0.6],
                                 [0.1, 0.2], [-0.1, -0.2])
copies = representation.lift(z)
internal = representation.pack(copies, 0.2, np.zeros(2))
assert copies.shape == (16,) and internal.shape == (20,)
```

Each method initializes its run from the problem and request, selects the
representation and sets passive momentum to zero. It prepares each step's
copies, applies projection and integrates momentum from its accepted stages.
Shared method helpers `_initialize_state`, `_momentum_rate` and `_finish_state`
preserve energy normalization and physical-only solver scales. Existing implicit
method `solver=None` selects configured defaults; representations have no solver.

Accepted doubled snapshots contain equal copies. `physical(state)` reads the
first copy under that precondition; it never averages an off-diagonal input.
Methods evaluate the saved Hamiltonian once or use values from the compiled
kernel. `extract_history(..., energy=values)` and `diagnose_history` combine
those supplied values with stored momentum for output; they evaluate no dynamics.
Energy-tracked export requires energy values. Specialized GC/FC split-map
interfaces retain caller-triggered preparation and map operations, reusing
the generic copy helpers.

## Choosing equations and solver independently

```python
from formulations import ReducedProjection, SimultaneousProjection
from solvers import NewtonSolver, BroydenSolver
from methods import ABBA4Implicit, GaussLegendre4

solver = NewtonSolver(absolute_tolerance=1e-12,
                      relative_tolerance=1e-11, max_iterations=40)
method = ABBA4Implicit(step=0.02, projection_formulation=ReducedProjection(), solver=solver)
result = simulate(problem, method, request)
gauss = GaussLegendre4(step=0.02, solver=solver)
```

`ReducedProjection` solves for mu in R^(2N). `SimultaneousProjection` solves for
(u, v, mu) in R^(6N). Both implement `contracts.formulations.projection.ProjectionFormulation`
and own the initial guess, residual, Jacobian assembly and physical reconstruction.
The method supplies its doubled composition and derivative; formulations do not
import methods. Their array operations serve NumPy and JAX. ABBA accepts either
class or the existing `reduced_multiplier`/`simultaneous_state_multiplier` strings;
BM4 retains the reduced equations. Gauss stage equations remain method-specific.
The older `methods.ProjectionFormulation` and `simulation.ProjectionFormulation`
exports remain the string-name type alias; import the equation protocol from
`formulations.projection` when defining its type.

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
