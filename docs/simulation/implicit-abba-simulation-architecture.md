# Implicit ABBA simulation architecture

This document explains the companion
[`implicit-abba-simulation-architecture.puml`](implicit-abba-simulation-architecture.puml)
diagram. It follows one `ImplicitABBA1` simulation from public input assembly,
through the reduced implicit ABBA solve, to the immutable `Solution` returned to
the caller.

The diagram is a runtime-path view, not a complete package inventory. Some boxes
are concrete Python classes, while others name a distinct algorithmic role
implemented by private helpers in the same module. The distinction is stated
explicitly below.

## The three main parts

The horizontal diagram has three principal regions. The small inherited boundary
at the far left is an input to these regions rather than a fourth phase.

| Region | Main question | Starts with | Produces |
|---|---|---|---|
| **1. Run assembly** | What physical problem, method, and time request will be run? | Dynamics, initial configuration, method parameters, and output schedule | A validated `InitialValueProblem`, `ImplicitABBA1`, and `SimulationRequest` |
| **2. Implicit ABBA integration** | How is each numerical step advanced and projected back to a physical state? | The assembled run | Requested states plus nonlinear-solver diagnostics |
| **3. Result boundary** | How is internal numerical output validated and exposed safely? | `IntegrationData` | An immutable public `Solution` |

The complete main path is:

```text
dynamics + initial configuration
            |
            v
InitialValueProblem + SimulationRequest + ImplicitABBA1
            |
            v
SimulationRunner -> ImplicitABBA1.integrate(...)
                 -> fixed grid -> projected ABBA steps
            |
            v
IntegrationData -> Return to SimulationRunner -> Solution
```

## Inherited boundary: guiding-centre dynamics

### `DynamicalSystem`

**File:** [`src/dynamics/protocols.py`](../../src/dynamics/protocols.py)

`DynamicalSystem` is the general runtime-checkable dynamics protocol. It requires
`state_dimension` and `vector_field(t, state)`. `InitialValueProblem` is typed
against this broad contract so general numerical methods need not know which
physical system they receive.

### `GuidingCenterJacobianSystem`

**File:** [`src/dynamics/protocols.py`](../../src/dynamics/protocols.py)

`GuidingCenterJacobianSystem` inherits the `DynamicalSystem` protocol and adds
`particle_vector_field_jacobians(t, state)`. The method returns one exact `2 x 2`
spatial Jacobian for each independent guiding-centre particle. The implicit ABBA
coordinator narrows the general problem dynamics to this capability.

### `GuidingCenterDynamics`

**File:** [`src/dynamics/gc.py`](../../src/dynamics/gc.py)

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
`GuidingCenterJacobianSystem` contract. Only Newton evaluates the exact particle
Jacobians; Broyden evaluates the vector field and residual only.

## Part 1: run assembly

This region creates a coherent simulation request before any numerical work is
performed.

### `InitialConfiguration`

**File:** [`src/simulation/configuration.py`](../../src/simulation/configuration.py)

`InitialConfiguration` is a runtime-checkable protocol, not a concrete initial
condition. A concrete implementation such as
[`GCInitialConfiguration`](../../src/initial_conditions/gc.py) owns the initial
physical state and exposes a separate object that interprets packed arrays.

Its members are:

| Member | Responsibility |
|---|---|
| `initial_state` | Returns an independent initial-state copy, or `None` if unset. |
| `layout` | Returns the independent `StateLayout` used to interpret physical arrays. |

The configuration owns the initial state, not the packed-memory rules. Physical
parameters such as the potential, gyroaverage radius, or magnetic normalization
remain in the dynamics object.

### `StateLayout`

**File:** [`src/simulation/configuration.py`](../../src/simulation/configuration.py)

`StateLayout` is the independent runtime-checkable contract consumed by the
simulation core:

| Member | Responsibility |
|---|---|
| `state_dimension` | Declares the number of physical components per particle. |
| `validate_packed_state_layout(state)` | Checks one packed state or a complete state history. |
| `split(state)` | Exposes component-major storage as physical component blocks. |
| `particle_count(state)` | Computes the number of represented particles. |
| `positions(state)` | Returns the planar `x` and `y` position blocks. |

Every physical state supported by GC2D represents particles in a plane, so
position extraction is part of the base contract rather than an optional
capability. External layouts remain supported through structural typing, but
they must expose their particle positions.

`pack_components(...)`, `as_blocks(...)`, and `from_blocks(...)` are useful
construction and implementation operations, but the generic simulation
contract does not require them.

### `PackedStateLayout`

**File:** [`src/initial_conditions/base.py`](../../src/initial_conditions/base.py)

`PackedStateLayout` contains the reusable component-major reshape, validation,
packing, and particle-count implementation. It stores no initial state and no
physical parameters. Concrete layouts inherit it and provide
`state_dimension` plus the physical meaning of each block.

### `StateConfiguration`

**File:** [`src/initial_conditions/base.py`](../../src/initial_conditions/base.py)

`StateConfiguration` is an abstract initial-state storage base. Its abstract
`layout` property prevents construction until a concrete subclass supplies a
state interpretation.

It owns an optional initial-state copy and provides:

- `set_initial_state(...)`, which validates and stores one finite flat state;
- `initial_state`, which returns an independent copy; and
- temporary forwarding methods for notebook compatibility, while supported
  simulation code accesses `configuration.layout` directly.

The class does not inherit the protocol. Concrete configurations complete
the `InitialConfiguration` contract by returning a concrete layout.

### `GCStateLayout`

**File:** [`src/initial_conditions/gc.py`](../../src/initial_conditions/gc.py)

`GCStateLayout` inherits the common packed-layout implementation and defines:

- `state_dimension = 2`;
- component order `[x_1, ..., x_N, y_1, ..., y_N]`;
- `split(...) -> GCState(x, y)`; and
- `positions(...)` as required by `StateLayout`.

It structurally conforms to `StateLayout` without inheriting that protocol.

### `GCInitialConfiguration`

**File:** [`src/initial_conditions/gc.py`](../../src/initial_conditions/gc.py)

`GCInitialConfiguration` is a real Python subclass of `StateConfiguration`,
shown with a solid inheritance triangle. It:

- owns the optional initial-state copy;
- returns the shared stateless `GCStateLayout` through `layout`; and
- provides `from_components(x=..., y=...)` as its named constructor.

This is the concrete object normally supplied to `InitialValueProblem` for a
guiding-centre run. The problem still refers to it through the broader
`InitialConfiguration` protocol. The ordinary dashed dependency from
`GCInitialConfiguration` to `InitialConfiguration` denotes structural
conformance without inheritance, as stated in the diagram legend. Keeping that
meaning in the legend avoids placing a long label across either class box.

### `InitialValueProblem`

**File:** [`src/simulation/problem.py`](../../src/simulation/problem.py)

`InitialValueProblem` is a frozen value object that binds one dynamics instance
to one initial configuration.

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

### `SimulationRequest`

**File:** [`src/simulation/request.py`](../../src/simulation/request.py)

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

**File:** [`src/simulation/runner.py`](../../src/simulation/runner.py)

`SimulationRunner` is the public orchestration boundary. The module-level
`simulate(problem, method, request)` function is a convenience facade that
creates a runner and calls `SimulationRunner.simulate(...)`.

In the run-assembly region, `InitialValueProblem`, `NumericalMethod`, and
`SimulationRequest` all point into `SimulationRunner` because all three are
arguments of that public method. Their later responsibilities differ, but their
role at the API boundary is the same: caller-supplied input.

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

**File:** [`src/simulation/methods/base.py`](../../src/simulation/methods/base.py)

`NumericalMethod` is the runtime-checkable contract consumed by
`SimulationRunner`. It requires one operation:

```python
integrate(problem: InitialValueProblem, request: SimulationRequest) -> IntegrationData
```

The runner therefore depends on a method capability rather than on
`ImplicitABBA1` specifically. Other numerical methods can enter through the same
interface.

### `_ImplicitABBA`

**File:**
[`src/simulation/methods/_implicit_abba.py`](../../src/simulation/methods/_implicit_abba.py)

`_ImplicitABBA` is a private frozen dataclass that structurally implements
`NumericalMethod`. It owns the shared nonlinear configuration and implements the
common `integrate(...)` delegation used by reduced implicit ABBA variants.

Its configurable fields are:

| Field | Meaning | Default |
|---|---|---:|
| `newton_absolute_tolerance` | Absolute contribution to the nonlinear stopping threshold | `1e-13` |
| `newton_relative_tolerance` | State-scaled contribution to the threshold | `1e-12` |
| `newton_max_iterations` | Maximum nonlinear corrections per step | `12` |
| `nonlinear_solver` | Either `"newton"` or `"broyden"` | `"newton"` |
| `progress` | Enables the terminal progress indicator | `False` |
| `step_observer` | Optional callback for accepted main-grid steps | `None` |

#### `_ImplicitABBA.__post_init__()`

Normalizes and validates positive tolerances, the positive iteration limit, and
the nonlinear-solver name.

#### `_ImplicitABBA.integrate(...)`

Delegates the run to `_integrate_projected_abba(...)`, passing the selected step
solver, formulation name, nonlinear settings, progress option, and observer.

The dashed hollow-triangle arrow to `NumericalMethod` again means structural
implementation rather than explicit inheritance.

### `ImplicitABBA1`

**File:**
[`src/simulation/methods/abba_implicit_1.py`](../../src/simulation/methods/abba_implicit_1.py)

`ImplicitABBA1` is a real Python subclass of `_ImplicitABBA`, shown by the solid
inheritance triangle. The class adds no new instance method; it specializes the
base by selecting:

- `_solve_projected_step` as `_step_solver`; and
- `implicit_1_reduced_equation_11` as `_solver_formulation`.

Consequently, an `ImplicitABBA1` instance inherits the validated tolerances,
solver selection, observer configuration, and `integrate(...)` implementation
from `_ImplicitABBA` while identifying the concrete reduced projection equation.

## Part 2: implicit ABBA integration

Most boxes in this region correspond to roles inside
[`src/simulation/methods/_projected_abba.py`](../../src/simulation/methods/_projected_abba.py).
The scheduler is the separate shared utility
[`src/simulation/_fixed.py`](../../src/simulation/_fixed.py).

### Integration coordinator: `_integrate_projected_abba(...)`

This private function coordinates the complete method run.

Before stepping, it verifies that:

- the dynamics implements `GuidingCenterJacobianSystem`;
- the dynamics is planar with `state_dimension == 2`; and
- when Newton is selected, an exact vector-field Jacobian can be evaluated at
  the initial state.

It then defines the `advance(t, state, step, step_index, observe)` callback used
by the fixed-grid scheduler. Each callback invocation solves one projected ABBA
step. Accepted main-grid calls record nonlinear diagnostics and may emit an
observation; shadow calls return a state without altering the main diagnostic
history.

After the scheduler finishes, the coordinator packages saved states and these
diagnostics into `IntegrationData`:

- main `step_count`;
- solver and formulation names;
- nonlinear iteration and residual-evaluation counts;
- residual norms and stopping tolerances;
- projection-multiplier norms; and
- configured absolute tolerance, relative tolerance, and iteration limit.

The `newton_*` diagnostic names remain alongside the general `nonlinear_*`
names for compatibility.

### Fixed-grid scheduler: `integrate_fixed_grid(...)`

**File:** [`src/simulation/_fixed.py`](../../src/simulation/_fixed.py)

The scheduler separates the numerical trajectory from the requested sampling
schedule.

1. It chooses the smallest uniform main-step count whose internal step does not
   exceed `request.max_step`.
2. It advances the main trajectory from `t_0` to `t_f` with equal steps.
3. If an output time lies exactly on a main-grid endpoint, it reuses that state.
4. If an output time lies inside a main interval, it performs a shorter **shadow
   advance** from the preceding main-grid state.

A shadow advance is used only to obtain a requested sample. It does not replace
the main state, affect later steps, contribute main-step diagnostics, or emit a
step observation. Consequently, changing `output_times` does not change the
underlying main integration trajectory.

The function returns:

- a state history with shape `(state_size, number_of_output_times)`; and
- the number of accepted main-grid steps.

### Projected step: `_solve_projected_step(...)`

This is the step solver selected by `ImplicitABBA1`. Its inputs are the dynamics,
current time `t`, physical state `z`, step duration `h`, nonlinear tolerances,
iteration limit, and solver choice.

The function:

1. validates the current physical state;
2. initializes the projection multiplier with `mu = 0`;
3. computes the stopping threshold

   \[
   \tau = \text{atol} + \text{rtol}\,
   \max\left(1, \lVert z \rVert_\infty\right);
   \]

4. repeatedly evaluates the duplicated ABBA stage map;
5. updates `mu` with Newton or good Broyden steps until the residual norm is at
   most `tau`; and
6. returns a `_ProjectedStep` record.

If the iteration limit is reached, or if a Newton block is singular, the method
raises a contextual `RuntimeError` containing the time, step size, and residual
information.

### Duplicated ABBA stage map

This is a conceptual diagram box implemented by `_evaluate_stages(...)`,
`_evaluate_unprojected_stages(...)`, and the private `_ABBAStages` record. It is
not a public class named `ABBAStages`.

For current physical state `z` and multiplier `mu`, two displaced copies are
created:

\[
u_0 = z + \mu,
\qquad
v_0 = z - \mu.
\]

With vector field `f`, step `h`, initial time `t`, and final time `t+h`, the
endpoint-time A-B-B-A shears are:

\[
\begin{aligned}
u_1 &= u_0 + \frac{h}{2}f(t,v_0), \\
v_1 &= v_0 + \frac{h}{2}f(t,u_1), \\
v_f &= v_1 + \frac{h}{2}f(t+h,u_1), \\
u_f &= u_1 + \frac{h}{2}f(t+h,v_f).
\end{aligned}
\]

The `_ABBAStages` record retains `u_initial`, `v_initial`, `u_first`, `v_final`,
`u_final`, and the residual. These snapshots support the nonlinear solve,
diagnostics, and optional exact tangent analysis.

`_checked_vector_field(...)` guards every stage evaluation against a changed
shape or non-finite value.

### Nonlinear projection root

This is another conceptual box. The reduced nonlinear equation is solved inside
`_solve_projected_step(...)`; Broyden's reusable iteration machinery lives in
`_solve_broyden(...)`.

The reduced residual is:

\[
r(\mu) = u_f(\mu) - v_f(\mu) + 2\mu.
\]

The bidirectional arrow between the stage map and the root means:

1. the root solver proposes a multiplier `mu`;
2. the ABBA map evaluates `u_f`, `v_f`, and `r(mu)`; and
3. the root solver uses that residual to produce the next multiplier.

#### Newton path

Newton differentiates the four traversed ABBA stages with
`_differentiate_stages(...)`. It evaluates exact particle vector-field
Jacobians at the relevant endpoint-time states and assembles one independent
`2 x 2` reduced residual Jacobian per particle. The packed residual is reshaped
into particle blocks, solved in a batch, and repacked into component-major
order.

#### Broyden path

Broyden evaluates the same reduced residual but does not call the exact
particle-Jacobian capability. It starts with `4 I` as the residual-Jacobian
approximation and updates that approximation from successive residuals and
multiplier changes.

Both paths target the same reduced implicit root. The choice changes how the
root is found, not the accepted mathematical equation.

### Accepted physical step

This conceptual box corresponds to the private `_ProjectedStep` dataclass.
After convergence, the projected copies are

\[
u^+ = u_f + \mu,
\qquad
v^+ = v_f - \mu,
\]

and the accepted physical state is their neutral mean:

\[
z_{n+1} = \frac{u^+ + v^+}{2}.
\]

The returned `_ProjectedStep` also contains:

| Field | Meaning |
|---|---|
| `state` | Accepted physical state `z_(n+1)` |
| `multiplier` | Converged projection multiplier `mu` |
| `stages` | Converged `_ABBAStages` snapshots |
| `iterations` | Number of nonlinear corrections |
| `residual_evaluations` | Number of residual evaluations |
| `residual_norm` | Final infinity norm of the reduced residual |

### Optional step observation

**File:** [`src/simulation/observation.py`](../../src/simulation/observation.py)

If `ImplicitABBA1.step_observer` is set, every accepted main-grid step emits an
`ImplicitABBAIntegrationStep`. Shadow advances never emit one.

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

**File:** [`src/simulation/_result.py`](../../src/simulation/_result.py)

`IntegrationData` is a frozen internal transfer object with three fields:

- `t`: requested saved times;
- `states`: sampled physical state history; and
- `diagnostics`: method-specific named numerical values.

It is intentionally small. The numerical method constructs it, and
`SimulationRunner` validates it before any public `Solution` is created.

### `Solution`

**File:** [`src/simulation/solution.py`](../../src/simulation/solution.py)

`Solution` is the immutable public simulation result.

#### Construction

Its constructor validates increasing finite times, the two-dimensional state
history shape, finite values, and layout compatibility with the source initial
configuration. It owns copies of arrays, marks them read-only, and exposes the
diagnostic mapping through `MappingProxyType`.

Canonical properties are:

| Property | Contents |
|---|---|
| `t` | Saved times with shape `(T,)` |
| `states` | Physical history with shape `(state_size, T)` |
| `source` | Initial-state provider whose `layout` interprets the packed history |
| `diagnostics` | Read-only mapping of method diagnostics |

#### Interpretation helpers

- `components(layout=None)` splits the full history into physical
  component blocks.
- `positions()` delegates to `source.layout` and returns both position
  histories.

Deprecated compatibility views `y`, `trajectory`, `n_steps`, `k`, and `err`
remain available while older notebooks migrate to canonical names.

The arrow from the `Return to SimulationRunner` continuation to `Solution`
identifies `SimulationRunner` as the object that constructs the public result.
The source arrow from `InitialConfiguration` means that the initial-state
provider is retained for provenance, while the separate dotted arrow from
`StateLayout` records which object validates and interprets the computed
history. None of these relations means that the initial state becomes the
computed trajectory.

## Complete runtime walkthrough

The diagram can be read from left to right as the following sequence:

1. A guiding-centre dynamics object supplies `vector_field(...)` and exact
   particle Jacobians.
2. An `InitialConfiguration` supplies a finite packed initial state and its
   component layout.
3. `InitialValueProblem` validates that dynamics and layout are compatible.
4. `SimulationRequest` defines the time span, maximum main step, and saved
   output times.
5. `ImplicitABBA1` defines nonlinear tolerances, solver choice, and optional
   observation.
6. `simulate(...)` validates the public objects and calls
   `ImplicitABBA1.integrate(...)`.
7. `_integrate_projected_abba(...)` checks the guiding-centre capabilities and
   configures the fixed-grid `advance(...)` callback.
8. `integrate_fixed_grid(...)` creates the output-independent uniform main grid.
9. Every main or shadow advance invokes `_solve_projected_step(...)`.
10. The step solver alternates between evaluating the duplicated ABBA map and
    updating `mu` with Newton or Broyden.
11. A converged root produces one accepted physical `_ProjectedStep`.
12. Accepted main-grid steps contribute diagnostics and may emit an
    `ImplicitABBAIntegrationStep` observation.
13. The numerical method returns requested samples and diagnostics as
    `IntegrationData` to `SimulationRunner`.
14. `SimulationRunner` validates that internal data and constructs the
    immutable public `Solution`.

## Arrow and line conventions

| Diagram notation | Meaning |
|---|---|
| Solid arrow into a consumer | An argument or returned value is supplied to that consumer |
| Solid arrow between runtime steps | A call, construction, or forward hand-off |
| Solid two-way arrow | Iteration between multiplier selection and stage-map residual evaluation |
| Solid line with hollow triangle | Explicit Python inheritance |
| Dashed line with hollow triangle | Structural implementation of a `Protocol` |
| Dashed dependency arrow to a protocol | Structural conformance without Python inheritance |
| Dashed dependency arrow to another class | Consumed capability or optional side channel |
| Package boundary | Architectural responsibility, not necessarily a Python package |
| Yellow note | Important invariant or intentionally omitted detail |

The absence of text on most horizontal arrows is intentional. The action is
described inside the destination box, which keeps connection labels from
overlapping UML compartments in rendered diagrams.

Consequently, the three public inputs point into `SimulationRunner`, whereas
`IntegrationData` points into the `Return to SimulationRunner` continuation
because it is the value returned by `NumericalMethod.integrate(...)`. That
continuation then points to `Solution`, which the runner constructs.

## Minimal public usage

The following sketch shows how the run-assembly boxes meet at the public API:

```python
import numpy as np

from initial_conditions import GCInitialConfiguration
from simulation import ImplicitABBA1, InitialValueProblem, SimulationRequest, simulate

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
method = ImplicitABBA1(
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

The diagram focuses on `ImplicitABBA1`, the reduced equation-(11) formulation.
It deliberately omits:

- the internal potential and field model of the guiding-centre dynamics;
- the simultaneous `ImplicitABBA2` formulation;
- higher-order ABBA compositions and other numerical methods;
- downstream diagnostic algorithms that consume step observations; and
- experiment- or notebook-specific construction of physical parameters.

Those systems interact with this path through the public dynamics,
`NumericalMethod`, observation, and `Solution` boundaries shown here.
