# Canonical ABBA simulation architecture

This document explains the companion
[`abba2-implicit-simulation-architecture.puml`](abba2-implicit-simulation-architecture.puml)
diagram. It follows one selected ABBA configuration from public input assembly
to the read-only `Solution` returned to the caller. The method family contains
five public classes, while orthogonal parameters select residual formulation,
nonlinear solver, and state extension. The diagram deliberately represents
those axes without drawing 51 nearly identical method boxes.

The diagram is a runtime-path view, not a complete package inventory. Some boxes
are concrete Python classes, while others name a distinct algorithmic role
implemented by private helpers in the same module. The distinction is stated
explicitly below.

## The three main parts

The horizontal diagram has three principal regions. The small inherited boundary
at the far left is an input to these regions rather than a fourth phase.

| Region | Main question | Starts with | Produces |
|---|---|---|---|
| **1. Run assembly** | What physical problem, method, and time request will be run? | Dynamics, initial configuration, one of five ABBA classes, and output schedule | A validated problem, method configuration, and request |
| **2. ABBA integration** | How are the state extension, composition, projection, and optional nonlinear solve applied? | The assembled run | Requested states plus method diagnostics |
| **3. Result boundary** | How is internal numerical output validated and exposed safely? | `IntegrationData` | A read-only public `Solution` |

The complete main path is:

```text
dynamics + initial configuration
            |
            v
InitialValueProblem + SimulationRequest + one canonical ABBA method
            |
            v
SimulationRunner -> method.integrate(...)
                 -> fixed grid -> configured ABBA steps
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

For physical and shared-time implicit configurations, Newton evaluates the
exact spatial particle Jacobians (`Phi_xx`, `Phi_xy`, and `Phi_yy`), whereas
Broyden evaluates residual values only. Fully extended Newton additionally
builds the analytic `4 x 4` extended-vector-field Jacobian from `Phi_xt`,
`Phi_yt`, and `Phi_tt`. Fully extended Broyden still needs the `Phi_t` momentum
flow but does not evaluate those analytic Jacobian derivatives. The HDF5
potential contract reconstructs `Phi_tt` frequency by frequency through
`evaluate(..., dt=2)`.

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
Callers import exactly five public method classes from `simulation` or
`simulation.methods`:

1. `ABBA2Midpoint`;
2. `ABBA2Implicit`;
3. `ABBA4Implicit`;
4. `ABBA4ImplicitSingleProjection`; and
5. `ABBA6Implicit`.

There are no public classes dedicated to state extensions. Every method selects
its state-space strategy through `state_extension`.

| Module | Responsibility |
|---|---|
| `__init__.py` | Reexports the five public ABBA classes and their canonical configuration values. |
| `_configuration.py` | Defines and validates `ProjectionFormulation`, `StateExtension`, and one-particle dimension diagnostics. |
| `_coefficients.py` | Owns the signed ABBA4 and ABBA6 composition coefficients. |
| `_core.py` | Implements the generic endpoint-time A-B-B-A map shared by midpoint and implicit projection. |
| `_projection_common.py` | Builds the shared displaced stages, exact stage tangents, and accepted-step record. |
| `_projection_reduced.py` | Implements formulation 1: the reduced projection-multiplier solve. |
| `_projection_simultaneous.py` | Implements formulation 2: the simultaneous output--multiplier solve. |
| `_implicit.py` | Owns shared implicit configuration, formulation dispatch, run coordination, fixed-grid hand-off, and diagnostics. |
| `order2_midpoint.py` | Implements `ABBA2Midpoint` and its three state extensions without a nonlinear solve. |
| `order2_implicit.py` | Implements `ABBA2Implicit` and dispatches its selected configuration. |
| `order4_implicit.py` | Implements three signed projected maps with one global configuration. |
| `order4_implicit_single_projection.py` | Applies one selected projection around the complete unprojected triple jump. |
| `order6_implicit.py` | Implements seven signed projected maps with one global configuration. |
| `extensions/` | Reserves private namespaces for extension-specific implementation details; it exports no numerical method class. |
| `methods/_fully_extended.py` | Implements the `R^8` full-state kernel, its `R^4` and `R^12` projection solves, and the fully extended midpoint path. |

Method-independent utilities remain outside the folder. In particular,
`methods/_nonlinear.py` is shared by ABBA and BM4 nonlinear solvers, while
`simulation/_fixed.py` is shared by numerical methods outside the ABBA family.

### `_ABBAImplicitConfig`

**File:**
[`src/simulation/methods/abba/_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py)

`_ABBAImplicitConfig` is a private frozen dataclass shared by the four implicit
classes. It is a configuration base, not a complete numerical method: it has no
`integrate(...)` implementation. Concrete descendants supply that operation and
thereby satisfy `NumericalMethod` structurally.

Its configurable fields are:

| Field | Meaning | Default |
|---|---|---:|
| `projection_formulation` | `"reduced_multiplier"` or `"simultaneous_state_multiplier"` | `"reduced_multiplier"` |
| `state_extension` | `"physical"`, `"shared_time"`, or `"fully_extended"` | `"physical"` |
| `newton_absolute_tolerance` | Absolute contribution to the nonlinear stopping threshold | `1e-13` |
| `newton_relative_tolerance` | State-scaled contribution to the threshold | `1e-12` |
| `newton_max_iterations` | Maximum nonlinear corrections per step | `12` |
| `nonlinear_solver` | Either `"newton"` or `"broyden"` | `"newton"` |
| `progress` | Enables the terminal progress indicator | `False` |
| `step_observer` | Optional callback for accepted main-grid steps | `None` |

#### `_ABBAImplicitConfig.__post_init__()`

Normalizes and validates the three canonical selectors, positive tolerances,
and the positive iteration limit.

### Canonical configuration axes

The four implicit classes expose a Cartesian product of three independent
axes:

| Axis | Choices | Numerical role |
|---|---:|---|
| `projection_formulation` | 2 | Selects the reduced or simultaneous residual. |
| `nonlinear_solver` | 2 | Selects Newton or Broyden for that residual. |
| `state_extension` | 3 | Selects physical, shared-time, or fully extended state duplication. |

Each implicit class therefore has `2 x 2 x 3 = 12` configurations.
`ABBA2Midpoint` has no residual and no nonlinear solver, so only its three
`state_extension` choices are valid. Across the five public classes this gives

```text
4 implicit classes x 12 + 1 midpoint class x 3 = 51 configurations.
```

`ABBA_PROJECTION_FORMULATIONS`, `NONLINEAR_SOLVERS`, and
`ABBA_STATE_EXTENSIONS` expose the canonical values. A configuration is a
runtime choice of one public method; it is not another class.

### The five public methods

#### `ABBA2Midpoint`

**File:**
[`src/simulation/methods/abba/order2_midpoint.py`](../../../../src/simulation/methods/abba/order2_midpoint.py)

`ABBA2Midpoint` starts both copies on the diagonal, applies one endpoint-time
A-B-B-A map, and accepts their arithmetic mean. It has no projection residual,
multiplier, Newton iteration, or Broyden iteration. Its only numerical-model
axis is `state_extension`; `progress` and `step_observer` are execution controls.

#### `ABBA2Implicit`

**File:**
[`src/simulation/methods/abba/order2_implicit.py`](../../../../src/simulation/methods/abba/order2_implicit.py)

`ABBA2Implicit` inherits `_ABBAImplicitConfig` and supplies `integrate(...)`.
One outer step contains one A-B-B-A map and one selected implicit projection.
Physical and shared-time configurations use `_integrate_projected_abba(...)`;
fully extended configurations use `_integrate_abba_fully_extended(...)`.

Because `ABBA2Implicit` supplies `integrate(problem, request)`, the dashed
hollow-triangle relation from this concrete class to `NumericalMethod` denotes
structural protocol implementation. The solid hollow-triangle relation from
`ABBA2Implicit` to `_ABBAImplicitConfig` denotes actual Python inheritance.

#### `ABBA4Implicit`

**File:**
[`src/simulation/methods/abba/order4_implicit.py`](../../../../src/simulation/methods/abba/order4_implicit.py)

One outer step composes signed durations `(gamma h, delta h, gamma h)`. Each of
the three maps is completed by its own implicit projection. The selected
formulation, solver, and extension are global: all three substeps use the same
configuration. Thus this class has 12 configurations, not independent binary
choices for each of its three substeps.

#### `ABBA4ImplicitSingleProjection`

**File:**
[`src/simulation/methods/abba/order4_implicit_single_projection.py`](../../../../src/simulation/methods/abba/order4_implicit_single_projection.py)

This method applies the same three signed unprojected maps continuously and
places one implicit projection around the complete triple jump. It performs one
nonlinear solve per outer step. This projection placement defines a different
numerical map from `ABBA4Implicit`; it is not a fourth configuration axis.

#### `ABBA6Implicit`

**File:**
[`src/simulation/methods/abba/order6_implicit.py`](../../../../src/simulation/methods/abba/order6_implicit.py)

One outer step contains seven signed projected maps using Yoshida's palindromic
sixth-order coefficients. As for `ABBA4Implicit`, its formulation, solver, and
extension choices are global for all seven substeps.

### Dimensional convention

For one guiding-centre particle, the accepted state, splitting state, and
nonlinear workspace have the following dimensions:

| `state_extension` | Accepted internal state | Base splitting state | Reduced unknown | Simultaneous unknown |
|---|---|---|---|---|
| `"physical"` | `z in R^2` | `(u,v) in R^4` | `mu in R^2` | `(u_f,v_f,mu) in R^6` |
| `"shared_time"` | `(z,t,kappa) in R^4` | `(u,v,t,k) in R^6` | `mu in R^2` | `(u_f,v_f,mu) in R^6` |
| `"fully_extended"` | `Z=(z,t,k) in R^4` | `(Z_1,Z_2) in R^8` | `mu in R^4` | `(Z_1f,Z_2f,mu) in R^12` |

Here `z=(x,y)`, `Z=(z,t,k)`, and the table's dimensions are literal for one
particle. For a physical run with `N` independent particles, its accepted,
splitting, reduced, and simultaneous dimensions scale to `2N`, `4N`, `2N`,
and `6N`. Shared-time and fully extended configurations currently require
exactly one `GuidingCenterDynamics` particle, so their tabulated dimensions do
not have a multi-particle generalization in the current API.

The shared-time strategy duplicates only `z` and shares `(t,k)`. Its accepted
momentum is `kappa=k/2`; the triangular momentum equation cannot feed back into
`z`, so it preserves the corresponding physical trajectory. The fully extended
strategy duplicates all of `Z`, advances direct `k`, and can define a different
physical map.

The accepted internal dimension and observer dimension deliberately differ for
shared time. Physical and shared-time observers receive the closed physical map
`z -> z_next` in `R^2`; `t` and `kappa` remain available through diagnostics.
Fully extended observers receive the accepted internal map `Z -> Z_next` in
`R^4`. The diagnostics fields `observer_state_dimension` and
`observer_state_kind` make this contract explicit for all five public methods.

The two appearances of `R^6` are unrelated. `(u,v,t,k)` is a genuine splitting
state for `shared_time`, whereas `(u_f,v_f,mu)` is a temporary nonlinear
workspace for a simultaneous physical or shared-time solve. For
`fully_extended`, the analogous simultaneous workspace is in `R^12`.

### Projection formulations: one exact root, two solve vectors

For a fixed method, extension, and step, both formulations impose the same
diagonal projection and define the same exact map at convergence. They differ
only in the nonlinear unknown retained by the solver.

| Formulation | Physical/shared-time unknown | Fully extended unknown |
|---|---|---|
| `reduced_multiplier` | `mu in R^2` | `mu in R^4` |
| `simultaneous_state_multiplier` | `(u_f,v_f,mu) in R^6` | `(Z_1f,Z_2f,mu) in R^12` |

Newton differentiates the selected residual exactly. Broyden evaluates that
same residual and updates a Jacobian approximation. Solver choice does not
change the exact root; finite tolerances and round-off can produce small
differences. Both values are recorded by diagnostics and step observations.

## Part 2: configured ABBA integration

This region makes the numerical execution order explicit. Read its principal
boxes from left to right:

```text
fixed grid -> state-extension strategy -> composition policy
           -> arithmetic mean, or implicit residual + Newton/Broyden
           -> accepted internal state -> physical sample -> IntegrationData
```

Implicit configurations have two nested loops:

1. The **outer time loop** advances the physical trajectory over the uniform
   main grid and, when necessary, computes independent shadow samples.
2. The **inner nonlinear loop** solves each projection selected by the method's
   composition policy. `ABBA4Implicit` performs three such solves,
   `ABBA4ImplicitSingleProjection` performs one, and `ABBA6Implicit` performs
   seven per outer step.

`ABBA2Midpoint` traverses the same outer loop and extension dispatch but bypasses
the formulation and nonlinear-solver boxes. It closes each duplicated map with
an arithmetic mean.

Most boxes are algorithmic roles rather than separate public Python classes.
Their files are deliberately separated by responsibility:

- [`_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py) coordinates
  the complete run;
- [`_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py)
  implements formulation 1;
- [`_projection_simultaneous.py`](../../../../src/simulation/methods/abba/_projection_simultaneous.py)
  implements formulation 2;
- [`_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py)
  retains only their shared stage and tangent kernels;
- [`_core.py`](../../../../src/simulation/methods/abba/_core.py) applies the
  physical/shared-time projection-independent A-B-B-A stage map; and
- [`methods/_fully_extended.py`](../../../../src/simulation/methods/_fully_extended.py)
  owns the duplicated `R^8` maps, `R^4` and `R^12` residuals, and fully extended
  midpoint mean.

The outer scheduler remains the shared utility
[`src/simulation/_fixed.py`](../../../../src/simulation/_fixed.py).

### Physical and shared-time coordinator: `_integrate_projected_abba(...)`

**File:**
[`src/simulation/methods/abba/_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py)

This private function coordinates an `ABBA2Implicit` physical or shared-time
run. `order4_implicit.py` generalizes the same pattern to the three- and
seven-projection compositions, while
`order4_implicit_single_projection.py` coordinates the one-projection triple
jump.

Before stepping, it verifies that:

- the dynamics implements `GuidingCenterJacobianSystem`;
- the dynamics is planar with `state_dimension == 2`; and
- when Newton is selected, an exact vector-field Jacobian can be evaluated at
  the initial state.

When `state_extension="shared_time"`, it additionally requires one concrete
`GuidingCenterDynamics` particle, initializes the accepted internal state as
`(z,t,kappa)`, and checks that its carried time stays aligned with the fixed
grid. A `physical` run carries only `z`.

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
`z` history enters `IntegrationData.states`. Their observers likewise receive
the closed `R^2` physical map, while the carried `t` and `kappa` remain in those
diagnostics. `observer_state_dimension=2` and
`observer_state_kind="physical_map"` record that choice.

The `newton_*` diagnostic names remain alongside the general `nonlinear_*`
names for compatibility. The coordinator does not itself perform the temporal
or nonlinear iteration: it creates the callback and hands control to the outer
time loop.

### Fully extended coordinator: `_integrate_abba_fully_extended(...)`

**File:**
[`src/simulation/methods/_fully_extended.py`](../../../../src/simulation/methods/_fully_extended.py)

When an implicit class selects `state_extension="fully_extended"`, its public
`integrate(...)` method delegates to this coordinator. It initializes
`Z=(z,t,k)` with `k=0`, duplicates the complete state into `(Z_1,Z_2) in R^8`,
and dispatches the selected residual:

- the reduced formulation solves `mu in R^4`;
- the simultaneous formulation solves `(Z_1f,Z_2f,mu) in R^12`.

The method variant determines the composition coefficients and projection
placement. `ABBA4ImplicitSingleProjection` builds one complete unprojected
triple-jump base map before its single solve. The other implicit variants solve
after each constituent A-B-B-A map. The same configuration is used for every
constituent map.

`ABBA2Midpoint(state_extension="fully_extended")` enters the neighboring
`_integrate_abba_fully_extended_midpoint(...)` path. It duplicates the same
`R^4` accepted state into `R^8`, but closes the copies by their arithmetic mean
and performs no nonlinear solve.

For both implicit and midpoint fully extended paths, the observer receives the
accepted internal `R^4` map rather than its `R^2` physical projection. The
coordinator records `observer_state_dimension=4` and
`observer_state_kind="accepted_internal_map"`; sampled solution states remain
physical and therefore have dimension two.

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

The following residual and stage subsections expand the representative
`ABBA2Implicit(state_extension="physical")` path. Shared time reuses this
physical solve and advances `(t,kappa)` alongside it. Fully extended execution
uses dimensionally analogous residuals in `_fully_extended.py`, with the
`R^4/R^8/R^12` dimensions stated above.

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
The same public class can carry the shared `(t,kappa)` variables or use the
fully duplicated `R^8` kernel, according to `state_extension`. The dashed arrow
in the diagram represents shared-core dependency; midpoint is a separate public
method, not a branch executed inside an implicit run.

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

With `state_extension="shared_time"`, the coordinator then advances the accepted
conjugate value from the four converged ABBA stage snapshots. The split variable
is `k`, while the accepted variable is normalized as `kappa=k/2`; the
accumulated split increment is therefore divided by two. The accepted internal
value becomes `(z_(n+1),t_n+h,kappa_(n+1))`, but only `z_(n+1)` is exposed in
the physical state history.

With `state_extension="fully_extended"`, the corresponding accepted record is
`Z_(n+1)=(z_(n+1),t_(n+1),k_(n+1)) in R^4`. The runner still receives only the
physical `z` history; time, momentum, and generalized-energy values are exposed
as diagnostics.

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

If `ABBA2Implicit.step_observer` is set for a physical or shared-time run, every
accepted main-grid step emits an `ABBA2ImplicitIntegrationStep`. Composed and
single-projection methods emit their corresponding observation records. Fully
extended implicit runs emit `FullyExtendedImplicitIntegrationStep`; midpoint
runs emit the general `IntegrationStep`. Shadow advances never emit an event.

For `physical` and `shared_time`, `state_before`, `state_after`, and the domain
and codomain of `map_state` are the closed physical state in `R^2`. In
particular, a shared-time observer does not receive the accepted internal
`(z,t,kappa) in R^4`; that carried pair is exposed as `extended_time` and
`extended_kappa` in integration diagnostics. For `fully_extended`, those three
observer objects use the accepted internal state `Z=(z,t,k) in R^4`. This is
also why the latter can expose a full `4 x 4` accepted-map Jacobian.

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
10. The public-method box selects one of the five canonical classes.
11. `ABBA2Midpoint` validates only `state_extension`; each implicit class
    inherits `_ABBAImplicitConfig` and validates all three canonical axes.
12. `state_extension` chooses the physical `R^4`, shared-time `R^6`, or fully
    duplicated `R^8` base splitting state.
13. The method class chooses one, three, or seven signed A-B-B-A maps and whether
    projection occurs after each map or once around the complete composition.
14. Midpoint closes the copies by arithmetic mean and bypasses the nonlinear
    branch.
15. An implicit method selects the reduced or simultaneous residual. Its
    workspace is `R^2/R^6` for physical and shared-time execution, or
    `R^4/R^12` for fully extended execution.
16. Newton or Broyden solves that residual. The same choice is reused globally
    by all substeps of a composed method.
17. The accepted internal state is reduced back to the physical `z` sample;
    extension variables remain in diagnostics. Physical and shared-time
    observers receive the closed `R^2` map, while fully extended observers
    receive the accepted internal `R^4` map.
18. `integrate_fixed_grid(...)` either advances the main trajectory or stores a
    shadow sample and repeats until `t_f`.
19. Accepted main steps may emit the observation type associated with the
    selected method and extension.
20. The numerical method returns requested physical samples and diagnostics as
    `IntegrationData`.
21. Execution returns to the existing `SimulationRunner.simulate(...)` call,
    which validates the transfer object.
22. The runner constructs and returns the public `Solution`.

## Arrow and line conventions

| Diagram notation | Meaning |
|---|---|
| Solid arrow into a consumer | An argument or returned value is supplied to that consumer |
| Solid arrow between runtime steps | A call, construction, or forward hand-off |
| Solid line with hollow triangle | Explicit Python inheritance |
| Dashed line with hollow triangle | Structural implementation of a `Protocol` |
| Ordinary dashed arrow | Required capability, structural reuse, optional side channel, or midpoint bypass of the nonlinear solver |
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
    state_extension="physical",
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

The diagram shows all five public method classes and all three configuration
axes, but it does not duplicate the runtime row for every one of the 51 valid
configurations. Each option list is vertical inside its box, and the execution
phases remain ordered horizontally. The diagram deliberately omits:

- the internal potential and field model of the guiding-centre dynamics;
- the individual algebraic stages inside each higher-order composition;
- numerical methods outside the ABBA family;
- downstream diagnostic algorithms that consume step observations; and
- experiment- or notebook-specific construction of physical parameters.

Those systems interact with this path through the public dynamics,
`NumericalMethod`, observation, and `Solution` boundaries shown here.
