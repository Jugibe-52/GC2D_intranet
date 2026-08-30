# ABBA2 implicit simulation architecture

This document explains the companion
[`abba2-implicit-simulation-architecture.puml`](abba2-implicit-simulation-architecture.puml)
diagram. It follows one `ABBA2Implicit` simulation from public input assembly,
through either supported projection formulation, to the read-only `Solution`
returned to the caller. It also locates the shared-time and fully duplicated
state extensions without confusing their state spaces with a nonlinear solve.

The diagram is a runtime-path view, not a complete package inventory. Some boxes
are concrete Python classes, while others name a distinct algorithmic role
implemented by private helpers in the same module. The distinction is stated
explicitly below.

## The three main parts

The horizontal diagram has three principal regions. The small inherited boundary
at the far left is an input to these regions rather than a fourth phase.

| Region | Main question | Starts with | Produces |
|---|---|---|---|
| **1. Run assembly** | What physical problem, method, and time request will be run? | Dynamics, initial configuration, method parameters, and output schedule | A validated `InitialValueProblem`, `ABBA2Implicit`, and `SimulationRequest` |
| **2. Implicit ABBA integration** | How is each numerical step advanced and projected back to a physical state? | The assembled run | Requested states plus nonlinear-solver diagnostics |
| **3. Result boundary** | How is internal numerical output validated and exposed safely? | `IntegrationData` | A read-only public `Solution` |

The complete main path is:

```text
dynamics + initial configuration
            |
            v
InitialValueProblem + SimulationRequest + ABBA2Implicit
            |
            v
SimulationRunner -> ABBA2Implicit.integrate(...)
                 -> fixed grid -> projected ABBA steps
            |
            v
IntegrationData -> Return to SimulationRunner -> Solution
```

## Inherited boundary: guiding-centre dynamics

### `GuidingCenterDynamics`

**File:** [`src/dynamics/gc.py`](../../../../src/dynamics/gc.py)

`GuidingCenterDynamics` is the concrete structural implementation used in this
path. It does not explicitly inherit the protocol, but it provides every required
member: `state_dimension = 2`, `vector_field(...)`, and
`particle_vector_field_jacobians(...)`. The dashed hollow-triangle relationship
in the diagram denotes this structural protocol implementation.

Its potential, gyroaverage construction, and field internals remain outside the
simulation diagram. This boundary consumes only the public dynamics capability.

The state convention is component-major:

```text
[x_1, ..., x_N, y_1, ..., y_N]
```

Both Newton and Broyden enter through the same
`GuidingCenterJacobianSystem` protocol. Only Newton evaluates the exact particle
Jacobians; Broyden evaluates the vector field and residual only.

### `GuidingCenterJacobianSystem`

**File:** [`src/dynamics/protocols.py`](../../../../src/dynamics/protocols.py)

`GuidingCenterJacobianSystem` inherits the `DynamicalSystem` protocol and adds
`particle_vector_field_jacobians(t, state)`. The method returns one exact `2 x 2`
spatial Jacobian for each independent guiding-centre particle. The implicit ABBA
coordinator narrows the general problem dynamics to this capability.

### `DynamicalSystem`

**File:** [`src/dynamics/protocols.py`](../../../../src/dynamics/protocols.py)

`DynamicalSystem` is the general runtime-checkable dynamics protocol. It requires
`state_dimension` and `vector_field(t, state)`. `InitialValueProblem` is typed
against this broad contract so general numerical methods need not know which
physical system they receive.

The inherited boundary is therefore read from the concrete implementation on
the left, through the guiding-centre-specific capability, to the general
dynamical-system contract on the right. These arrows express type relationships,
not successive runtime calls.

## Part 1: run assembly

The subsections below mirror the exact horizontal order of the diagram. Within
each vertical state box, the configuration is described before its layout.

### `InitialValueProblem`

**File:** [`src/simulation/problem.py`](../../../../src/simulation/problem.py)

`InitialValueProblem` is a frozen dataclass that binds one dynamics instance to
one initial configuration. Freezing prevents reassignment of those two fields;
the referenced dynamics and configuration objects are not deep-copied.

#### `__post_init__()`

Construction performs the structural checks that make the pair safe to pass to
a numerical method:

1. `dynamics` must implement `DynamicalSystem`.
2. `initial_configuration` must implement `InitialConfiguration`.
3. `initial_configuration.layout` must implement `StateLayout`.
4. An initial state must be present.
5. The initial state must be a finite one-dimensional vector.
6. The layout and dynamics must declare the same `state_dimension`.

#### `initial_state`

Returns a fresh copy of the validated initial state. Numerical methods therefore
do not receive the configuration's owned array directly.

#### `particle_count`

Asks `initial_configuration.layout` to interpret the packed state and return
its particle count.

`InitialValueProblem` appears first because it is the public problem object whose
dependencies are expanded by the boxes to its right. This is an architectural
reading order, not the literal construction order in user code: a concrete
initial configuration must already exist when the problem is instantiated.

Two solid arrows enter `InitialValueProblem`. `DynamicalSystem` is drawn on its
left, so that arrow points right; `InitialConfiguration` is drawn on its right,
so that arrow points left. Both have exactly the same meaning: the corresponding
object is supplied to the constructor and retained in a dataclass field. Neither
arrow denotes inheritance.

### State contracts

This vertical box contains the two runtime-checkable protocols consumed by
`InitialValueProblem` and `Solution`.

#### `InitialConfiguration`

**File:** [`src/simulation/configuration.py`](../../../../src/simulation/configuration.py)

`InitialConfiguration` is a runtime-checkable protocol, not a concrete initial
condition. It is therefore marked `«protocol»` just like `DynamicalSystem` and
`NumericalMethod`; **State contracts** is the architectural name of the
containing box. A concrete implementation owns the optional initial physical
state and exposes a separate layout object that interprets packed arrays.

| Member | Responsibility |
|---|---|
| `initial_state` | Returns an independent initial-state copy, or `None` if unset. |
| `layout` | Returns the `StateLayout` used to interpret physical arrays. |

The hollow diamond between `InitialConfiguration` and `StateLayout` represents
that composed capability. The configuration owns state storage, not the packed
memory rules. Physical parameters such as the potential and gyroaverage radius
remain in the dynamics object.

#### `StateLayout`

**File:** [`src/simulation/configuration.py`](../../../../src/simulation/configuration.py)

`StateLayout` is the independent runtime-checkable state-interpretation
protocol:

| Member | Responsibility |
|---|---|
| `state_dimension` | Declares the number of physical components per particle. |
| `validate_packed_state_layout(state)` | Checks one packed state or a complete history. |
| `split(state)` | Returns physical component blocks while preserving sample axes. |
| `particle_count(state)` | Computes the number of represented particles. |
| `positions(state)` | Returns the planar `x` and `y` position blocks. |

Every state layout currently supported by the simulation core is planar, so
position extraction belongs to this base contract. `pack_components(...)`,
`as_blocks(...)`, and `from_blocks(...)` are useful implementation operations,
but the generic protocol does not require them.

### Guiding-center initial state

This vertical box is the concrete guiding-centre specialization of the two
contracts immediately to its left.

#### `GCInitialConfiguration`

**File:** [`src/initial_conditions/gc.py`](../../../../src/initial_conditions/gc.py)

`GCInitialConfiguration` is a real Python subclass of `StateConfiguration`. It:

- owns the optional initial-state copy through its base class;
- returns the shared stateless `GCStateLayout` through `layout`;
- supplies `from_components(x=..., y=...)`; and
- structurally conforms to `InitialConfiguration` without inheriting the
  protocol.

This is the concrete configuration normally supplied when constructing a
guiding-centre `InitialValueProblem`. Its dashed line with a hollow triangle
denotes structural implementation of the protocol.

#### `GCStateLayout`

**File:** [`src/initial_conditions/gc.py`](../../../../src/initial_conditions/gc.py)

`GCStateLayout` inherits the common packed-layout implementation and defines:

- `state_dimension = 2`;
- component order `[x_1, ..., x_N, y_1, ..., y_N]`;
- `split(...) -> GCState(x, y)`; and
- `positions(...)` as required by `StateLayout`.

It structurally conforms to `StateLayout` without inheriting that protocol. As
with `GCInitialConfiguration`, this is shown by a dashed line with a hollow
triangle.

### Shared state implementation

This vertical box contains reusable storage and packed-layout behavior. It is
shown after the guiding-centre specialization because the diagram follows the
requested visual order; the hollow inheritance triangles still point from the
concrete guiding-centre classes to these base classes.

#### `StateConfiguration`

**File:** [`src/initial_conditions/base.py`](../../../../src/initial_conditions/base.py)

`StateConfiguration` is an abstract initial-state storage base. Its abstract
`layout` property prevents instantiation until a concrete subclass supplies a
state interpretation.

It owns an optional initial-state copy and provides:

- `set_initial_state(...)`, which validates and stores one finite flat state;
- `initial_state`, which returns an independent copy; and
- compatibility forwarding methods, while supported simulation code accesses
  `configuration.layout` directly.

The class does not inherit `InitialConfiguration`; concrete configurations
complete that protocol structurally by exposing the required properties.

#### `PackedStateLayout`

**File:** [`src/initial_conditions/base.py`](../../../../src/initial_conditions/base.py)

`PackedStateLayout` contains reusable component-major splitting, reshaping,
validation, packing, and particle counting. It stores neither an initial state
nor physical parameters. A concrete layout supplies `state_dimension` and the
physical meaning of each component block.

### `SimulationRequest`

**File:** [`src/simulation/request.py`](../../../../src/simulation/request.py)

`SimulationRequest` is a frozen, method-independent description of the time
domain and sampling request.

Its three fields are:

- `t_span = (t_0, t_f)`: finite and strictly increasing integration interval.
- `max_step`: positive upper bound for the main integration step.
- `output_times`: finite, strictly increasing saved times that include both
  endpoints of `t_span`.

#### `__post_init__()`

Validates all temporal values, normalizes endpoint values within floating-point
tolerance, copies `output_times`, and marks it read-only.

#### `SimulationRequest.uniform(...)`

Convenience constructor that creates `sample_count` uniformly spaced output
times over `t_span`. These output times do not define the method's internal main
grid; that distinction is handled by `integrate_fixed_grid(...)`.

### `SimulationRunner` and `simulate(...)`

**File:** [`src/simulation/runner.py`](../../../../src/simulation/runner.py)

`SimulationRunner` is the public orchestration boundary. The module-level
`simulate(problem, method, request)` function is a convenience facade that
creates a runner and calls `SimulationRunner.simulate(...)`.

`InitialValueProblem`, `NumericalMethod`, and `SimulationRequest` all point into
`SimulationRunner` because all three are arguments of that public method. The
problem and request arrive from the left; the method protocol is placed to the
right so the remaining diagram can expand the selected implementation before
entering the integration region.

The diagram shows the return phase of that same `simulate(...)` call as
`Return to SimulationRunner` inside the result boundary. This is an algorithmic
continuation, not a second Python class or runner instance. Separating the entry
and return phases keeps the runtime path horizontal without drawing a long
backward cycle across the integration boxes.

#### `SimulationRunner.simulate(...)`

The method:

1. validates the public `problem`, `method`, and `request` objects;
2. calls `method.integrate(problem, request)`;
3. checks that returned times exactly equal the requested output times;
4. checks state shape, finiteness, and preservation of the initial state;
5. asks `initial_configuration.layout` to validate the complete packed history;
   and
6. constructs the public `Solution` with the original configuration as its
   initial-state source.

This boundary prevents a numerical method from returning malformed or
method-specific state layouts to user code.

### `NumericalMethod`

**File:** [`src/simulation/methods/base.py`](../../../../src/simulation/methods/base.py)

`NumericalMethod` is the runtime-checkable protocol consumed by
`SimulationRunner`. It requires one operation:

```python
integrate(problem: InitialValueProblem, request: SimulationRequest) -> IntegrationData
```

The runner therefore depends on a method capability rather than on
`ABBA2Implicit` specifically. Other numerical methods can enter through the same
interface.

### ABBA method subpackage

All ABBA-specific numerical methods now live under
[`src/simulation/methods/abba/`](../../../../src/simulation/methods/abba/).
The top-level import path remains stable while the classes use the new naming
convention: callers import `ABBA2Implicit`, `ABBA2Midpoint`, and the other ABBA
method classes from `simulation` or `simulation.methods`.

| Module | Responsibility |
|---|---|
| `__init__.py` | Reexports the public ABBA method classes as one family. |
| `_core.py` | Implements the generic endpoint-time A-B-B-A map shared by midpoint and implicit projection. |
| `_projection_common.py` | Builds the shared displaced stages, exact stage tangents, and accepted-step record. |
| `_projection_reduced.py` | Implements formulation 1: the reduced projection-multiplier solve. |
| `_projection_simultaneous.py` | Implements formulation 2: the simultaneous output--multiplier solve. |
| `_implicit.py` | Owns shared implicit configuration, formulation dispatch, run coordination, fixed-grid hand-off, and diagnostics. |
| `order2_midpoint.py` | Implements arithmetic-mean projection using the shared A-B-B-A core. |
| `order2_implicit.py` | Implements `ABBA2Implicit` and selects its projection solver from `projection_formulation`. |
| `order4_implicit.py` | Composes three signed reduced implicit steps into `ABBA4Implicit`. |
| `order4_implicit_single_projection.py` | Applies one reduced projection around a complete fourth-order composition. |
| `order6_implicit.py` | Implements the seven-substep sixth-order composition. |
| `extensions/shared_time.py` | Implements the shared `(t,k)` lift as `ABBA2SharedTimeExtendedImplicit`. |
| `extensions/fully_extended.py` | Exposes `ABBA2FullyExtendedImplicit` and `ABBA4FullyExtendedImplicit`. |
| `methods/_fully_extended.py` | Implements the neutral `R^8` full-state kernel shared by ABBA and BM4 wrappers. |

Method-independent utilities remain outside the folder. In particular,
`methods/_nonlinear.py` is shared by ABBA and BM4 nonlinear solvers, while
`simulation/_fixed.py` is shared by numerical methods outside the ABBA family.

### `_ABBAImplicitConfig`

**File:**
[`src/simulation/methods/abba/_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py)

`_ABBAImplicitConfig` is a private frozen dataclass that owns shared nonlinear-solver
configuration. It is a configuration base, not a complete numerical method:
it deliberately has no `integrate(...)` method. Concrete descendants implement
that operation and thereby satisfy `NumericalMethod` structurally.

Its configurable fields are:

| Field | Meaning | Default |
|---|---|---:|
| `newton_absolute_tolerance` | Absolute contribution to the nonlinear stopping threshold | `1e-13` |
| `newton_relative_tolerance` | State-scaled contribution to the threshold | `1e-12` |
| `newton_max_iterations` | Maximum nonlinear corrections per step | `12` |
| `nonlinear_solver` | Either `"newton"` or `"broyden"` | `"newton"` |
| `progress` | Enables the terminal progress indicator | `False` |
| `step_observer` | Optional callback for accepted main-grid steps | `None` |

#### `_ABBAImplicitConfig.__post_init__()`

Normalizes and validates positive tolerances, the positive iteration limit, and
the nonlinear-solver name.

### `ABBA2Implicit`

**File:**
[`src/simulation/methods/abba/order2_implicit.py`](../../../../src/simulation/methods/abba/order2_implicit.py)

`ABBA2Implicit` explicitly inherits `_ABBAImplicitConfig` and adds both the public
`projection_formulation` field and `integrate(...)`. Its default is
`"reduced_multiplier"`; `"simultaneous_state_multiplier"` selects the
algebraically equivalent output--multiplier solve. `_step_solver_for(...)`
maps the semantic identifier to the corresponding private solver and
`integrate(...)` delegates to `_integrate_projected_abba(...)`.

Because `ABBA2Implicit` supplies `integrate(problem, request)`, the dashed
hollow-triangle relation from this concrete class to `NumericalMethod` denotes
structural protocol implementation. The solid hollow-triangle relation from
`ABBA2Implicit` to `_ABBAImplicitConfig` denotes actual Python inheritance.

### Projection formulations: one map, two solve vectors

For one guiding-centre particle, both formulations accept and return the same
physical state `z in R^2` and traverse the same duplicated ABBA stage map on
`(u,v) in R^4`.

| Formulation | Nonlinear unknown | Newton block | Meaning of `R^6` |
|---|---|---:|---|
| `reduced_multiplier` | `mu in R^2` | `2 x 2` | Not used |
| `simultaneous_state_multiplier` | `(u_f,v_f,mu) in R^6` | `6 x 6` | Algebraic solve vector only |

The simultaneous formulation's `R^6` vector is not a trajectory state. Once
its root is found, the same projected `z_(n+1)` is returned. Both branches emit
`projection_formulation` in diagnostics and in
`ABBA2ImplicitIntegrationStep.formulation_name`.

### State-extension classes

The extension classes are separate numerical-method subclasses because they
change the internal state carried through the split flow:

| Class | Accepted internal state | Base splitting state | Momentum | Relationship to the physical map |
|---|---|---|---|---|
| `ABBA2SharedTimeExtendedImplicit` | `(z,t,kappa) in R^4` | `(u,v,t,k) in R^6` | `kappa=k/2` | Same `z` trajectory as `ABBA2Implicit` with the selected formulation |
| `ABBA2FullyExtendedImplicit` | `(z,t,k) in R^4` | `(Z_1,Z_2) in R^8` | Direct `k` | Distinct full-state projected ABBA2 map |
| `ABBA4FullyExtendedImplicit` | `(z,t,k) in R^4` | `(Z_1,Z_2) in R^8` | Direct `k` | Fourth-order full-state composition |

Here `Z=(z,t,k)`. The shared-time extension inherits `ABBA2Implicit`, reuses
its `projection_formulation`, and asks the same coordinator to advance the
additional `(t,kappa)` values. The full-state classes instead inherit the
neutral `_FullyExtendedImplicitMethod` kernel. All three extensions currently
require exactly one particle.

The table distinguishes the two unrelated six-dimensional objects: the
simultaneous formulation has a temporary `R^6` Newton unknown, whereas
`ABBA2SharedTimeExtendedImplicit` has a genuine `R^6` splitting state.

## Part 2: implicit ABBA integration

This region makes the numerical execution order explicit. Read its principal
boxes from left to right:

```text
coordinator -> outer time loop -> formulation selector
            -> reduced or simultaneous nonlinear solve
            -> shared A-B-B-A map -> accepted physical state
            -> outer-loop continuation -> IntegrationData
```

There are two nested loops:

1. The **outer time loop** advances the physical trajectory over the uniform
   main grid and, when necessary, computes independent shadow samples.
2. The **inner nonlinear loop** solves one projected ABBA step. The reduced
   branch corrects `mu`; the simultaneous branch corrects `(u_f,v_f,mu)`.

Most boxes are algorithmic roles rather than separate public Python classes.
Their files are deliberately separated by responsibility:

- [`_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py) coordinates
  the complete run;
- [`_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py)
  implements formulation 1;
- [`_projection_simultaneous.py`](../../../../src/simulation/methods/abba/_projection_simultaneous.py)
  implements formulation 2;
- [`_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py)
  retains only their shared stage and tangent kernels; and
- [`_core.py`](../../../../src/simulation/methods/abba/_core.py) applies the
  projection-independent A-B-B-A stage map.

The outer scheduler remains the shared utility
[`src/simulation/_fixed.py`](../../../../src/simulation/_fixed.py).

### Integration coordinator: `_integrate_projected_abba(...)`

**File:**
[`src/simulation/methods/abba/_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py)

This private function coordinates the complete method run.

Before stepping, it verifies that:

- the dynamics implements `GuidingCenterJacobianSystem`;
- the dynamics is planar with `state_dimension == 2`; and
- when Newton is selected, an exact vector-field Jacobian can be evaluated at
  the initial state.

For `ABBA2SharedTimeExtendedImplicit`, it additionally requires one concrete
`GuidingCenterDynamics` particle, initializes the accepted internal state as
`(z,t,kappa)`, and checks that its carried time stays aligned with the fixed
grid. Ordinary `ABBA2Implicit` runs carry only `z`.

It then defines the `advance(t, state, step, step_index, observe)` callback used
by the fixed-grid scheduler. Each callback invocation solves one projected ABBA
step. Accepted main-grid calls record nonlinear diagnostics and may emit an
observation; shadow calls return a state without altering the main diagnostic
history.

After the scheduler finishes, the coordinator packages saved states and these
diagnostics into `IntegrationData`:

- main `step_count`;
- `nonlinear_solver`, `projection_formulation`, and `state_extension`;
- nonlinear iteration and residual-evaluation counts;
- residual norms and stopping tolerances;
- projection-multiplier norms; and
- configured absolute tolerance, relative tolerance, and iteration limit.

Shared-time runs additionally store `extended_time`, `extended_kappa`, and
`extended_momentum_normalization="kappa_equals_k_over_2"`; only their physical
`z` history enters `IntegrationData.states`.

The `newton_*` diagnostic names remain alongside the general `nonlinear_*`
names for compatibility. The coordinator does not itself perform the temporal
or nonlinear iteration: it creates the callback and hands control to the outer
time loop.

### Outer time loop — entry: `integrate_fixed_grid(...)`

**File:** [`src/simulation/_fixed.py`](../../../../src/simulation/_fixed.py)

The scheduler separates the numerical trajectory from the requested sampling
schedule. At the `FixedGrid` box it performs these actions:

1. Chooses the smallest uniform main-step count whose internal step does not
   exceed `request.max_step`.
2. Builds the uniform main grid from `t_0` to `t_f`.
3. For each required interval, it invokes the coordinator's `advance(...)`
   callback with a main or shadow step.

The callback then enters the one-step setup shown in the next box. The effects
of the returned main or shadow state are described later under
**Outer time loop — continuation**, matching its position near the right-hand
side of the diagram.

### Formulation selection

**Files:**
[`src/simulation/methods/abba/_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py)
with
[`_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py)
and
[`_projection_simultaneous.py`](../../../../src/simulation/methods/abba/_projection_simultaneous.py)

`_step_solver_for(...)` selects `_solve_reduced_multiplier_step(...)` for
`reduced_multiplier` or `_solve_simultaneous_state_multiplier_step(...)` for
`simultaneous_state_multiplier`. Both helpers receive the same dynamics,
`(t_n,z_n,h)`, stopping controls, and nonlinear-solver choice. Both return the
same `_ProjectedStep` shape, so the coordinator and fixed-grid scheduler remain
independent of the selected algebraic formulation.

### Reduced branch: `_solve_reduced_multiplier_step(...)`

**File:**
[`src/simulation/methods/abba/_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py)

This is the default step solver. One invocation receives:

- the dynamics;
- the current main or shadow start time `t_n`;
- the packed physical state `z_n`;
- the requested step duration `h`; and
- the nonlinear tolerances, iteration limit, and solver selection.

Before starting the inner loop, it validates `z_n`, initializes the projection
multiplier as

\[
\mu_0 = 0,
\]

and computes the stopping threshold

\[
\tau = \operatorname{atol} + \operatorname{rtol}\,
\max\left(1, \lVert z_n \rVert_\infty\right).
\]

This setup occurs once per attempted main or shadow advance. The following
residual evaluation and correction may occur several times within that same
advance.

### Inner nonlinear loop — duplicated input

**File:**
[`src/simulation/methods/abba/_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py)

At the start of nonlinear iteration `k`, `_evaluate_displaced_stages(...)` uses
the current multiplier to create two displaced copies of the same physical state:

\[
u_0 = z_n + \mu_k,
\qquad
v_0 = z_n - \mu_k.
\]

This displacement is projection-specific. When the correction produces
`mu_(k+1)`, the backward arrow returns here so both copies are reconstructed
before the four stages are reevaluated.

### Inner nonlinear loop — shared A-B-B-A map

**File:**
[`src/simulation/methods/abba/_core.py`](../../../../src/simulation/methods/abba/_core.py)

`_evaluate_unprojected_stages(...)` accepts any `DynamicalSystem`; it needs only
`vector_field(...)`, not guiding-centre Jacobians. With initial copies `u_0` and
`v_0`, vector field `f`, initial time `t_n`, final time `t_n+h`, and step `h`,
it performs the endpoint-time shears:

\[
\begin{aligned}
u_1 &= u_0 + \frac{h}{2}f(t_n,v_0), \\
v_1 &= v_0 + \frac{h}{2}f(t_n,u_1), \\
v_f &= v_1 + \frac{h}{2}f(t_n+h,u_1), \\
u_f &= u_1 + \frac{h}{2}f(t_n+h,v_f).
\end{aligned}
\]

The core then records the unprojected copy separation

\[
d_k = u_f-v_f.
\]

The private `_ABBAStages` record retains `u_initial`, `v_initial`, `u_first`,
`v_final`, `u_final`, and this separation in its `residual` field. The name of
that field is intentionally generic because the projection layer replaces it
with the complete reduced residual. `_checked_vector_field(...)` guards every
stage evaluation against a changed shape or non-finite value.

#### Reuse by `ABBA2Midpoint`

[`order2_midpoint.py`](../../../../src/simulation/methods/abba/order2_midpoint.py) calls the
same core with

\[
u_0=v_0=z_n.
\]

It does not introduce a multiplier or nonlinear solve. It accepts the arithmetic
mean `(u_f+v_f)/2` and records `||d||_inf` as the copy-separation diagnostic.
The dashed arrow in the diagram represents this shared-core dependency;
`ABBA2Midpoint` is not executed during an `ABBA2Implicit` run.

### Inner nonlinear loop — projection residual

**File:**
[`src/simulation/methods/abba/_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py)

After the shared map returns, the reduced branch's `_evaluate_stages(...)` adds the multiplier term
required by Hairer's reduced symmetric projection:

\[
r_k = r(\mu_k) = d_k + 2\mu_k
    = u_f(\mu_k)-v_f(\mu_k)+2\mu_k.
\]

The returned projected `_ABBAStages` snapshots therefore contain the complete
`r_k` used by the convergence test, diagnostics, and optional tangent analysis.

### Inner nonlinear loop — convergence decision

After every residual evaluation, the solver checks

\[
\lVert r_k \rVert_\infty \leq \tau.
\]

The two outgoing arrows from `ResidualTest` are the two possible numerical
branches:

- **yes:** the multiplier has converged, so execution moves directly to
  `AcceptedStep`;
- **no:** execution enters `NonlinearCorrection`, computes a new multiplier,
  and follows the backward arrow to reevaluate all ABBA stages.

This is a convergence decision inside one time step. It must not be confused
with the outer scheduler's decision to advance to another physical time.

### Inner nonlinear loop — multiplier correction

**File:**
[`src/simulation/methods/abba/_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py),
using the exact stage derivatives from
[`_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py)

Both available nonlinear solvers target the same reduced root `r(mu) = 0`; they differ
only in how they compute the correction `Delta_mu_k`.

#### Newton path

Newton differentiates the four traversed ABBA stages with
`_differentiate_stages(...)`; `_evaluate_residual(...)` combines that
differentiation with the current stage evaluation. Newton evaluates exact
particle vector-field Jacobians at the relevant endpoint-time states and
assembles one independent
`2 x 2` reduced residual Jacobian `J_r` per particle. The packed system

\[
J_r(\mu_k)\,\Delta\mu_k = r_k
\]

is reshaped into particle blocks, solved in a batch, and repacked into
component-major order.

#### Broyden path

Broyden consumes the residual-only `_evaluate_stages(...)` path and does not
call `_differentiate_stages(...)` or the exact particle-Jacobian capability. It
starts with `4 I` as its residual-Jacobian approximation `B_0`, solves

\[
B_k\,\Delta\mu_k = r_k,
\]

and updates the approximation from successive multiplier and residual changes.

#### Return to the residual

Whichever solver is selected, the multiplier update is

\[
\mu_{k+1} = \mu_k - \Delta\mu_k.
\]

The backward arrow from `NonlinearCorrection` to `DuplicatedInput` means that
`u_0`, `v_0`, every A-B-B-A stage, `d_{k+1}`, and `r_{k+1}` are recomputed using
this new multiplier. This cycle continues until the convergence test succeeds.

If the iteration limit is reached, or if a Newton block is singular, the method
raises a contextual `RuntimeError` containing the time, step size, and residual
information.

### Simultaneous branch: `_solve_simultaneous_state_multiplier_step(...)`

**File:**
[`src/simulation/methods/abba/_projection_simultaneous.py`](../../../../src/simulation/methods/abba/_projection_simultaneous.py)

The alternative formulation starts from the uncorrected stage outputs and
solves for

\[
q=(u_f,v_f,\mu)\in\mathbb{R}^6
\]

per particle. Its residual concatenates two corrected step-equation defects
and the diagonal constraint:

\[
\begin{aligned}
d_u &= u_f-\mu-\widehat u_f(\mu),\\
d_v &= v_f+\mu-\widehat v_f(\mu),\\
g   &= u_f-v_f.
\end{aligned}
\]

Newton assembles one exact `6 x 6` block per particle. Broyden uses the same
six-component residual and a block initial approximation. At convergence the
diagonal constraint makes the two outputs equal up to tolerance; the solver
returns their mean as `z_(n+1)`. This temporary `q` is not stored as a state and
is unrelated to the shared-time splitting state described above.

### Accepted physical state

**File:**
The two branch modules construct the common `_ProjectedStep` record defined in
[`src/simulation/methods/abba/_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py).

In the reduced branch, once `mu_star` satisfies the stopping condition, the
final duplicated states are projected as

\[
u^+ = u_f + \mu_\star,
\qquad
v^+ = v_f - \mu_\star,
\]

and the accepted physical state is their neutral mean:

\[
z_{n+1} = \frac{u^+ + v^+}{2}.
\]

The simultaneous branch already includes the diagonal constraint in its root
and returns `(u_f+v_f)/2`. In exact arithmetic both routes produce the same
`z_(n+1)`; the mean suppresses only the finite-tolerance antisymmetric part.

The private `_ProjectedStep` returned to the scheduler contains:

| Field | Meaning |
|---|---|
| `state` | Accepted physical state `z_(n+1)` |
| `multiplier` | Converged projection multiplier `mu_star` |
| `stages` | Converged `_ABBAStages` snapshots |
| `iterations` | Number of nonlinear corrections |
| `residual_evaluations` | Number of residual evaluations |
| `residual_norm` | Final infinity norm of the selected formulation's residual |

For `ABBA2SharedTimeExtendedImplicit`, the coordinator then advances the
accepted conjugate value from the four converged ABBA stage snapshots. The
split variable is `k`, while the accepted variable is normalized as
`kappa=k/2`; the accumulated split increment is therefore divided by two.
The accepted internal value becomes `(z_(n+1),t_n+h,kappa_(n+1))`, but only
`z_(n+1)` is exposed in the physical state history.

### Outer time loop — continuation

Control now returns to `integrate_fixed_grid(...)`. The meaning of the accepted
state depends on the `observe` flag passed to the callback:

- With `observe=True`, it is a **main-grid advance**. The scheduler replaces
  the main state, and the coordinator records nonlinear diagnostics and may
  emit a step observation.
- With `observe=False`, it is a **shadow advance** used only to obtain a
  requested output sample. It does not replace the main state, affect later
  steps, contribute main-step diagnostics, or emit a step observation.

If an output time lies exactly on a main-grid endpoint, the scheduler reuses
that main state. If it lies inside a main interval, the scheduler computes a
shorter shadow advance from the preceding main-grid state. Consequently,
changing `output_times` does not change the underlying main integration
trajectory.

The scheduler repeats the outer loop until `t_f` and returns:

- a state history with shape `(state_size, number_of_output_times)`; and
- the number of accepted main-grid steps.

The coordinator then completes the `IntegrationData` object described at the
beginning of this section.

### Optional main-step observation

**File:** [`src/simulation/observation.py`](../../../../src/simulation/observation.py)

If `ABBA2Implicit.step_observer` is set, every accepted main-grid step emits an
`ABBA2ImplicitIntegrationStep`. Shadow advances never emit one.

The observation contains:

- method, dynamics, formulation, step index, start time, end time, and duration;
- state snapshots before and after the step;
- a `map_state` callable for reevaluating the same fixed-time numerical map;
- nonlinear solver, iteration count, residual evaluations, residual norm,
  tolerance, and multiplier norm;
- the exact generating dynamics instance; and
- copies of the converged multiplier and ABBA stage states.

The observation is data, not analysis. Diagnostic modules may consume it to
construct tangent maps, symplecticity measures, or reversibility checks without
importing private solver helpers or duplicating the integrator.

## Part 3: result boundary

This region separates private method output from the stable public result.

### `IntegrationData`

**File:** [`src/simulation/_result.py`](../../../../src/simulation/_result.py)

`IntegrationData` is a frozen internal transfer dataclass with three fields:

- `t`: requested saved times;
- `states`: sampled physical state history; and
- `diagnostics`: method-specific named numerical values.

It is intentionally small. `frozen=True` prevents field reassignment, but it
does not itself freeze the contained arrays or diagnostics dictionary. The
numerical method constructs the object, and `SimulationRunner` validates its
contents before any public `Solution` is created.

### Return to `SimulationRunner`

This box is an algorithmic continuation of the same
`SimulationRunner.simulate(...)` call shown in the assembly region; it is not a
second Python class or runner instance. `NumericalMethod.integrate(...)` has now
returned `IntegrationData`, so execution resumes in the runner. The runner:

1. checks requested times, state shape, finiteness, and the first state;
2. validates the complete packed history through the source layout; and
3. constructs `Solution` from the validated arrays, source configuration, and
   diagnostics.

### `Solution`

**File:** [`src/simulation/solution.py`](../../../../src/simulation/solution.py)

`Solution` is the read-only public simulation result.

#### Construction

Its constructor validates increasing finite times, the two-dimensional state
history shape, finite values, and layout compatibility with the source initial
configuration. It owns copies of `t`, `states`, and diagnostic arrays; marks
those arrays read-only; and exposes the diagnostic mapping through
`MappingProxyType`.

Canonical properties are:

| Property | Contents |
|---|---|
| `t` | Saved times with shape `(T,)` |
| `states` | Physical history with shape `(state_size, T)` |
| `source` | Referenced initial-state provider whose `layout` interprets the packed history |
| `diagnostics` | Read-only mapping of method diagnostics |

#### Interpretation helpers

- `components(layout=None)` splits the full history into physical
  component blocks.
- `positions()` delegates to `source.layout` and returns both position
  histories.

Deprecated compatibility views `y`, `trajectory`, `n_steps`, `k`, and `err`
remain available while older notebooks migrate to canonical names.

The source configuration is retained by reference rather than copied or frozen.
The sampled times, states, and diagnostic arrays remain protected, but callers
should not mutate the source configuration after constructing a solution if
they want its provenance to remain unchanged.

The arrow from the `Return to SimulationRunner` continuation to `Solution`
identifies `SimulationRunner` as the object that constructs the public result.
The `Solution` box states that the initial configuration and its composed layout
are retained for provenance and for interpretation of the computed history.
This information is written inside the box instead of using two long cross-phase
arrows, so the integration and result phases remain ordered from left to right.
The initial state remains distinct from the computed trajectory.

## Complete runtime walkthrough

The following list follows the boxes from left to right. Type and containment
boxes explain dependencies and therefore do not all represent later moments in
wall-clock execution.

1. `GuidingCenterDynamics` provides the concrete vector field and exact particle
   Jacobians.
2. `GuidingCenterJacobianSystem` and then `DynamicalSystem` expose progressively
   more general contracts implemented by that concrete dynamics object.
3. `InitialValueProblem` binds dynamics to a validated initial configuration.
4. **State contracts** defines `InitialConfiguration` and its composed
   `StateLayout`.
5. **Guiding-center initial state** supplies the concrete
   `GCInitialConfiguration` and `GCStateLayout` pair.
6. **Shared state implementation** provides their reusable
   `StateConfiguration` and `PackedStateLayout` base behavior.
7. `SimulationRequest` defines the time span, maximum main step, and requested
   output samples.
8. `SimulationRunner` receives the problem, request, and a `NumericalMethod`.
9. `NumericalMethod` defines the integration capability consumed by the runner.
10. `_ABBAImplicitConfig` validates the common nonlinear configuration but does not
    itself implement `integrate(...)`.
11. `ABBA2Implicit` implements `NumericalMethod`, validates
    `projection_formulation`, and enters `_integrate_projected_abba(...)`.
12. The coordinator checks guiding-centre capabilities, creates the fixed-grid
    `advance(...)` callback, and starts the outer scheduler.
13. `integrate_fixed_grid(...)` selects the output-independent main grid and
    requests a main or shadow advance.
14. `_step_solver_for(...)` selects the reduced or simultaneous helper.
15. The reduced helper sets `mu_0=0`; the simultaneous helper initializes
    `(u_f,v_f,mu)` from the uncorrected output. Both compute the same stopping
    tolerance from `z_n`.
16. The shared `_core.py` map executes the four A-B-B-A shears.
17. The reduced branch tests `d_k+2mu_k`; the simultaneous branch tests its two
    step defects plus `u_f-v_f`.
18. A failed test invokes Newton or Broyden and repeats the selected residual;
    a successful test constructs the common physical `_ProjectedStep`.
19. The scheduler either advances the main trajectory or stores a shadow
    sample, then repeats its outer loop until `t_f`. Main steps may emit an
    `ABBA2ImplicitIntegrationStep`.
20. The numerical method returns requested samples and diagnostics as
    `IntegrationData`.
21. Execution returns to the existing `SimulationRunner.simulate(...)` call,
    which validates the transfer object.
22. The runner constructs and returns the public `Solution`.

## Arrow and line conventions

| Diagram notation | Meaning |
|---|---|
| Solid arrow into a consumer | An argument or returned value is supplied to that consumer |
| Solid arrow between runtime steps | A call, construction, or forward hand-off |
| Backward solid arrow inside the nonlinear solve | Reevaluate the stages and residual with the updated multiplier |
| Solid line with hollow triangle | Explicit Python inheritance |
| Dashed line with hollow triangle | Structural implementation of a `Protocol` |
| Ordinary dashed dependency | Required capability, shared-core reuse, or optional side channel |
| Package boundary | Architectural responsibility, not necessarily a Python package |
| Yellow note | Important invariant or intentionally omitted detail |

The absence of text on most horizontal arrows is intentional. The action is
described inside the destination box, which keeps connection labels from
overlapping UML compartments in rendered diagrams.

Consequently, the three public inputs point into `SimulationRunner`, whereas
`IntegrationData` points into the `Return to SimulationRunner` continuation
because it is the value returned by `NumericalMethod.integrate(...)`. That
continuation then points to `Solution`, which the runner constructs.

The same input rule applies to `InitialValueProblem`: both `DynamicalSystem` and
`InitialConfiguration` use solid arrows directed into the problem. Every Python
`Protocol` shown in the diagram carries the `«protocol»` stereotype. A concrete
class that satisfies a protocol structurally uses a dashed line with a hollow
triangle, regardless of which side of the concrete class the protocol occupies.

## Minimal public usage

The following sketch shows how the run-assembly boxes meet at the public API:

```python
import numpy as np

from initial_conditions import GCInitialConfiguration
from simulation import ABBA2Implicit, InitialValueProblem, SimulationRequest, simulate

configuration = GCInitialConfiguration.from_components(
    x=np.asarray([initial_x]),
    y=np.asarray([initial_y]),
)
problem = InitialValueProblem(dynamics, configuration)
request = SimulationRequest.uniform(
    t_span=(0.0, final_time),
    max_step=max_step,
    sample_count=sample_count,
)
method = ABBA2Implicit(
    projection_formulation="reduced_multiplier",
    nonlinear_solver="newton",
    newton_absolute_tolerance=1e-13,
    newton_relative_tolerance=1e-12,
    newton_max_iterations=12,
)

solution = simulate(problem, method, request)
x, y = solution.positions()
```

Here `dynamics`, `initial_x`, `initial_y`, `final_time`, `max_step`, and
`sample_count` are supplied by the calling experiment or study.

## Scope and deliberate omissions

The main runtime row focuses on `ABBA2Implicit`; its formulation selector shows
both equivalent nonlinear branches. The extension boxes show class ownership,
state dimensions, and momentum normalization without expanding a second full
runtime row. `ABBA2Midpoint` appears only to make reuse of `_core.py` explicit;
it is not part of the depicted implicit runtime path. The diagram deliberately
omits:

- the internal potential and field model of the guiding-centre dynamics;
- higher-order ABBA compositions and other numerical methods;
- downstream diagnostic algorithms that consume step observations; and
- experiment- or notebook-specific construction of physical parameters.

Those systems interact with this path through the public dynamics,
`NumericalMethod`, observation, and `Solution` boundaries shown here.
