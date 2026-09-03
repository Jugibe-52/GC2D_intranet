# ABBA2 implicit default physical simulation architecture

This document explains the companion
[`abba2-implicit-simulation-architecture.puml`](abba2-implicit-simulation-architecture.puml)
source. It renders three complementary views so static relationships are not
mixed with execution order:

1. the class, protocol, record, and result relationships;
2. the reduced-multiplier Newton loop for one implicit step; and
3. the end-to-end execution sequence, including main steps, shadow samples,
   optional observation, and result construction.

The expanded runtime branch is exactly the default
`ABBA2Implicit` configuration:

```python
ABBA2Implicit(
    projection_formulation="reduced_multiplier",
    state_extension="physical",
    nonlinear_solver="newton",
)
```

The method object exposes other selector values, but the three views stay
scoped to the default and do not expand the simultaneous, Broyden, shared-time,
or fully extended kernels. Those branches, the higher-order ABBA classes, and
all 51 valid family configurations are documented once in the authoritative
[`Canonical ABBA numerical architecture`](../../abba/simulation/abba-numerical-architecture.md).

## Runtime scope and call flow

The solid default path is:

```text
simulate(problem, method, request)
        |
        v
SimulationRunner.simulate(...)
        |
        v
ABBA2Implicit.integrate(...)
        |
        +-> _step_solver_for("reduced_multiplier")
        |       -> _solve_reduced_multiplier_step(...)
        |
        v
_integrate_projected_abba(...)
        |
        v
integrate_fixed_grid(...) -> advance(...)
        |
        v
displace copies -> endpoint-time A-B-B-A stages
        -> exact reduced Newton correction -> accepted physical state
        |
        v
IntegrationData -> SimulationRunner -> Solution
```

The optional dashed branch constructs `ABBA2ImplicitIntegrationStep` only for
accepted main-grid steps. Type-inheritance and protocol arrows in the diagram
describe structural relationships; they are not additional numerical calls.

## Principal files

| File | Responsibility in this scoped path |
|---|---|
| [`src/simulation/methods/abba/order2_implicit.py`](../../../../src/simulation/methods/abba/order2_implicit.py) | Public `ABBA2Implicit` method and extension dispatch |
| [`src/simulation/methods/abba/_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py) | Shared implicit configuration, solver selection, physical-run coordination, and diagnostics |
| [`src/simulation/methods/abba/_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py) | Reduced multiplier residual and Newton loop |
| [`src/simulation/methods/abba/_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py) | Displaced copies, exact stage differentiation, and accepted-step records |
| [`src/simulation/methods/abba/_core.py`](../../../../src/simulation/methods/abba/_core.py) | Projection-independent endpoint-time A--B--B--A stages |
| [`src/simulation/_fixed.py`](../../../../src/simulation/_fixed.py) | Output-independent main grid and shadow sampling |
| [`src/simulation/observation.py`](../../../../src/simulation/observation.py) | Optional accepted-step event records |
| [`src/simulation/runner.py`](../../../../src/simulation/runner.py) | Public validation and `Solution` construction |

## Public objects and configuration

[`InitialValueProblem`](../../../../src/simulation/problem.py) binds a
`DynamicalSystem` to an initial configuration and exposes a fresh validated
physical state. [`SimulationRequest`](../../../../src/simulation/request.py)
defines `(t_0,t_f)`, `max_step`, and the requested saved times.
[`SimulationRunner`](../../../../src/simulation/runner.py) consumes those two
objects and a structural
[`NumericalMethod`](../../../../src/simulation/methods/base.py), then validates
the method's transfer object before returning a public result.

The diagram's
[`InitialConfiguration`](../../../../src/simulation/configuration.py) and
`StateLayout` boxes are the structural state boundary consumed by
`InitialValueProblem` and the runner. The configuration owns the optional
initial-state copy; its composed layout validates the packed representation,
counts particles, and later interprets the returned history. These protocol
relationships prepare and validate the run but do not add numerical stages to
the ABBA step.

`ABBA2Implicit` inherits the frozen private `_ABBAImplicitConfig`. The static
view groups its solver settings compactly; the complete runtime fields are:

| Field | Default | Scoped role |
|---|---:|---|
| `projection_formulation` | `"reduced_multiplier"` | Selects the reduced multiplier solver shown |
| `state_extension` | `"physical"` | Keeps the accepted state in the physical packed space |
| `nonlinear_solver` | `"newton"` | Selects exact Newton differentiation shown |
| `newton_absolute_tolerance` | `1e-13` | Absolute stopping contribution |
| `newton_relative_tolerance` | `1e-12` | State-scaled stopping contribution |
| `newton_max_iterations` | `12` | Maximum Newton corrections in one attempted step |
| `progress` | `False` | Enables the shared terminal progress display |
| `step_observer` | `None` | Receives accepted main-step events |

Construction validates the three selector strings, requires positive finite
tolerances, and requires a positive integer iteration limit. For the physical
Newton path, `_integrate_projected_abba(...)` additionally requires:

- dynamics satisfying
  [`GuidingCenterJacobianSystem`](../../../../src/dynamics/protocols.py);
- `state_dimension == 2`; and
- one finite batched particle-vector-field Jacobian at the initial state.

Construction calls `_validate_projection_formulation(...)`,
`_validate_state_extension(...)`, `_validate_nonlinear_solver(...)`,
`_positive_finite(...)`, and `_positive_integer(...)`; each normalizes one
configuration field before the frozen method can enter the runner.

The physical packed convention for `N` independent particles is

```text
[x_1, ..., x_N, y_1, ..., y_N].
```

The accepted state and multiplier have dimension `2N`; the duplicated ABBA
state has dimension `4N`. Every Newton correction is solved as `N` independent
`2 x 2` systems and then restored to component-major order.

## Fixed-grid coordinator

`ABBA2Implicit.integrate(...)` calls `_step_solver_for(...)`. The scoped
selector value returns `_solve_reduced_multiplier_step(...)`, which is passed
to `_integrate_projected_abba(...)` as a callback. The coordinator defines its
own `advance(t, state, step, step_index, observe)` closure and delegates time
scheduling to `integrate_fixed_grid(...)`.

The scheduler selects the smallest uniform main-step count whose internal step
does not exceed `request.max_step` and uses

\[
h_{\mathrm{main}}=\frac{t_f-t_0}{\text{step_count}}.
\]

The private `_step_count(...)` helper performs that integer selection. After
the run, `_state_dimension_diagnostics(...)` derives the accepted, duplicated,
nonlinear, and observer dimensions recorded with the solver statistics.

Each main interval invokes `advance(...)` with `observe=True`. Requested times
at a main endpoint reuse the corresponding main state. A requested time inside
an interval invokes a shorter shadow step from a copy of the preceding main
state with `observe=False`.

Shadow steps solve the same nonlinear projection with their shorter duration,
so they can incur Newton work or raise the same solver errors. Their returned
states are used only as output samples: they do not replace the main state,
change later main steps, update progress, enter per-main-step diagnostic
arrays, or emit observations. The main trajectory is therefore independent of
the output sampling schedule.

## Hairer reduced symmetric projection

Let `z_n` be one accepted physical state, `h` the current main or shadow step,
and `mu` the packed projection multiplier. The reduced solver starts from

\[
\mu_0=0
\]

and `_evaluate_displaced_stages(...)` constructs two copies in opposite
directions:

\[
u_0=z_n+\mu,\qquad v_0=z_n-\mu.
\]

With `s=h/2`, `_evaluate_unprojected_stages(...)` applies the explicit
endpoint-time map

\[
\begin{aligned}
u_1 &= u_0+s f(t_n,v_0),\\
v_1 &= v_0+s f(t_n,u_1),\\
v_f &= v_1+s f(t_n+h,u_1),\\
u_f &= u_1+s f(t_n+h,v_f).
\end{aligned}
\]

`_checked_vector_field(...)` enforces finite, shape-preserving field values.
The endpoint-time convention is essential for the symmetry of a
nonautonomous step.

`_evaluate_stages(...)` completes Hairer's reduced residual:

\[
r(\mu)=u_f(\mu)-v_f(\mu)+2\mu.
\]

The stopping threshold is computed once per attempted advance:

\[
\tau=\mathrm{atol}+\mathrm{rtol}\,
\max\left(1,\lVert z_n\rVert_\infty\right).
\]

Convergence means

\[
\lVert r(\mu_k)\rVert_\infty\leq\tau.
\]

The private `_ABBAStages` record retains the traversed states. In this reduced
layer its `residual` field contains the complete `r(mu)`, rather than only the
unprojected separation `u_f-v_f` produced by the neutral core.

## Exact Newton correction

When the initial multiplier does not satisfy the threshold,
`_differentiate_stages(...)` evaluates the exact particle vector-field
Jacobians

\[
\begin{aligned}
W_1&=D_zf(t_n,v_0), & W_2&=D_zf(t_n,u_1),\\
W_3&=D_zf(t_n+h,u_1), & W_4&=D_zf(t_n+h,v_f).
\end{aligned}
\]

Each evaluation passes through `_checked_vector_field_jacobian(...)`, which
requires the finite shape `(N, 2, 2)` before the ordered products are formed.

Writing `S=W_2+W_3`, the exact reduced residual Jacobian assembled by the code
is

\[
J_r=4I-s(W_1+W_2+W_3+W_4)
    +s^2(W_4S+SW_1)-s^3W_4SW_1.
\]

The matrix order is significant because stage Jacobians at different points
need not commute. For every particle, `numpy.linalg.solve(...)` solves

\[
J_r(\mu_k)\,\Delta\mu_k=r(\mu_k),
\qquad
\mu_{k+1}=\mu_k-\Delta\mu_k.
\]

The next iteration reconstructs both displaced copies and reevaluates all four
stages. A zero-correction convergence at `mu_0` reports zero Newton iterations
and one residual evaluation. A singular particle block or failure to meet the
threshold after the configured corrections raises a contextual `RuntimeError`;
there is no adaptive retry or step-size reduction in this fixed-grid method.

The internal `_ResidualEvaluation` record, omitted from the compact static
view, carries `u_final`, `v_final`, the residual, the exact reduced Jacobian,
and the complete duplicated ABBA Jacobian. `_ProjectedStep` carries the
accepted state, multiplier, stage snapshots, correction count,
residual-evaluation count, and final residual norm.

## Accepted physical state

At a converged multiplier `mu_star`, the symmetric output correction gives

\[
u^+=u_f+\mu_\star,\qquad
v^+=v_f-\mu_\star.
\]

The exact root makes `u^+=v^+`. The implementation returns their neutral mean

\[
z_{n+1}=\frac{u^++v^+}{2},
\]

which suppresses the finite-tolerance antisymmetric part without changing the
exact projected map.

## Main-step diagnostics

After the fixed grid finishes, `_integrate_projected_abba(...)` returns
[`IntegrationData`](../../../../src/simulation/_result.py) with the requested
times, physical history, and the following scoped diagnostics:

| Key | Meaning or default-branch value |
|---|---|
| `step_count` | Accepted uniform main-grid steps |
| `nonlinear_solves_per_step` | `1` |
| `nonlinear_solver` | `"newton"` |
| `nonlinear_iterations` | Newton corrections per main step |
| `residual_evaluations` | Residual evaluations per main step |
| `nonlinear_residual_norms` | Final reduced residual infinity norm per main step |
| `nonlinear_tolerances` | State-scaled threshold per main step |
| `nonlinear_absolute_tolerance` | Configured absolute tolerance |
| `nonlinear_relative_tolerance` | Configured relative tolerance |
| `nonlinear_max_iterations` | Configured correction limit |
| `projection_multiplier_norms` | Final multiplier infinity norm per main step |
| `projection_formulation` | `"reduced_multiplier"` |
| `state_extension` | `"physical"` |
| `accepted_internal_state_dimension` | `2N` |
| `base_splitting_state_dimension` | `4N` |
| `nonlinear_unknown_dimension` | `2N` |
| `observer_state_dimension` | `2N` |
| `observer_state_kind` | `"physical_map"` |

The compatibility keys `newton_iterations`, `newton_residual_norms`,
`newton_absolute_tolerance`, `newton_relative_tolerance`, and
`newton_max_iterations` mirror their general `nonlinear_*` counterparts.
Every per-step array contains main steps only; shadow solves are deliberately
absent.

## Optional accepted-step observation

When `step_observer` is set, `advance(...)` creates one
[`ABBA2ImplicitIntegrationStep`](../../../../src/simulation/observation.py) for
each accepted main step. It inherits the general and implicit observation
fields from `IntegrationStep` and `ImplicitIntegrationStep`, then adds copies
of the converged multiplier and ABBA stage states.
The nested `apply_step(...)` closure is installed as the event's `map_state`
callable.
The event contains:

- method, dynamics, formulation, solver, step index, times, and duration;
- physical `state_before` and `state_after` snapshots;
- corrections, residual evaluations, final residual, stopping threshold, and
  multiplier norm;
- the exact dynamics instance; and
- `map_state`, which resolves the same fixed-time reduced Newton map for a new
  packed physical candidate.

The event is solver data, not a precomputed analysis. Diagnostic code can use
the retained stages and dynamics to construct the ideal-root tangent without
importing private solver functions. Shadow advances never emit an event.

## Public result boundary

When `ABBA2Implicit.integrate(...)` returns, execution resumes inside the same
runner call. The runner checks that saved times match the request, validates
state shape, finiteness, the unchanged initial sample, and the complete source
layout, then constructs
[`Solution`](../../../../src/simulation/solution.py). The public result owns
read-only copies of times, states, and diagnostic arrays and retains the source
initial configuration for component and position interpretation.

## Mathematical derivation and guarantees

The authoritative reduced physical derivation is
[`ABBA2_implicit.tex`](../ABBA2_implicit.tex), with its
[`compiled PDF`](../ABBA2_implicit.pdf). It derives:

- the duplicated guiding-centre phase space and diagonal constraint;
- the explicit endpoint-time A--B--B--A map and its symmetry;
- Hairer's symmetric input and output correction;
- the reduced residual used here;
- the ordered exact ABBA and residual Jacobians implemented by
  `_differentiate_stages(...)`; and
- the ideal-root physical tangent and symplecticity proof.

For a concrete HDF5 guiding-centre problem, the potential and exact derivative
capabilities entering this simulation boundary are documented separately in
[`gc2d-h5-import.md`](../dynamics/gc2d-h5-import.md).

The exact projected map is symplectic on the physical guiding-centre phase
space under the derivation's local hypotheses: the duplicated ABBA map is
symplectic, the reduced root is locally unique, its multiplier Jacobian is
invertible, and the projection equation is solved exactly. A finite stopping
tolerance introduces a corresponding finite symplectic defect.

## Limitations of this companion

- The diagram expands only physical reduced Newton. The simultaneous residual,
  Broyden update, shared-time normalization, and fully extended `R^8` kernel
  are outside its numerical region.
- It describes `ABBA2Implicit`, not the three- or seven-map higher-order
  compositions and not the single-projection ABBA4 placement.
- The fixed grid is not adaptive. Nonconvergence aborts the run rather than
  retrying with a smaller step.
- Exact Newton requires finite batched particle Jacobians with the protocol's
  expected shape. The concrete potential's derivative construction remains
  behind the dynamics boundary.
- The physical branch carries no time-conjugate momentum and publishes no
  generalized-energy diagnostic.
- Potential loading, downstream tangent or symplecticity analysis, and
  notebook-specific experiment assembly are deliberately outside the diagram.

## Minimal public usage

```python
from simulation import ABBA2Implicit, SimulationRequest, simulate

solution = simulate(
    problem,
    ABBA2Implicit(
        projection_formulation="reduced_multiplier",
        state_extension="physical",
        nonlinear_solver="newton",
        newton_absolute_tolerance=1e-13,
        newton_relative_tolerance=1e-12,
        newton_max_iterations=12,
    ),
    SimulationRequest.uniform(
        t_span=(0.0, final_time),
        max_step=max_step,
        sample_count=sample_count,
    ),
)
```

The caller supplies a validated guiding-centre-compatible `problem` and the
physical time and sampling parameters.
