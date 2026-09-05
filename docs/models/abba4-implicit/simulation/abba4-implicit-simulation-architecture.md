# ABBA4 implicit default physical simulation architecture

This document explains the companion
[`abba4-implicit-simulation-architecture.puml`](abba4-implicit-simulation-architecture.puml)
diagram. The expanded runtime branch is exactly the default
`ABBA4Implicit` configuration:

```python
ABBA4Implicit(
    projection_formulation="reduced_multiplier",
    state_extension="physical",
    nonlinear_solver="newton",
)
```

The complete ABBA family, its five public classes, orthogonal configuration
axes, and all 51 valid configurations are documented in the authoritative
[`Canonical ABBA numerical architecture`](../../abba/simulation/abba-numerical-architecture.md).
This companion does not duplicate the simultaneous, Broyden, shared-time, or
fully extended branches. The default ABBA2 reduced-Newton kernel reused by each
signed substep is described in detail by the
[`ABBA2 implicit companion`](../../abba2-implicit/simulation/abba2-implicit-simulation-architecture.md).

## Scoped runtime path

The default physical call flow is:

```text
simulate(problem, ABBA4Implicit(), request)
        |
        v
SimulationRunner.simulate(...)
        |
        v
ABBA4Implicit.integrate(...)
        |
        v
_integrate_abba4_implicit(...)
        |
        v
_integrate_composed_implicit_abba(...)
        |
        v
integrate_fixed_grid(...) -> advance(...) -> solve_step(...)
        |
        v
_solve_composed_abba_step(...)
        |
        +-> projected ABBA(gamma h)
        +-> projected ABBA(delta h)
        +-> projected ABBA(gamma h)
        |
        v
IntegrationData -> SimulationRunner -> Solution
```

Each arrow to a signed ABBA map represents a complete, independently solved
Hairer projection. The middle duration is negative. The optional observation
branch emits one `ABBA4ImplicitIntegrationStep` for the complete outer step;
its ordered `substeps` tuple retains all three signed solves.

## Principal files

| File | Responsibility in this scoped path |
|---|---|
| [`src/simulation/methods/abba/order4_implicit.py`](../../../../src/simulation/methods/abba/order4_implicit.py) | Public method, triple-jump coordinator, signed-substep records, aggregation, and observation assembly |
| [`src/simulation/methods/_abba_coefficients.py`](../../../../src/simulation/methods/_abba_coefficients.py) | Canonical ABBA4 and ABBA6 composition coefficients |
| [`src/simulation/methods/abba/_implicit.py`](../../../../src/simulation/methods/abba/_implicit.py) | Shared implicit configuration, solver selector, and shared-time helper outside this scope |
| [`src/simulation/methods/abba/_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py) | Reduced multiplier residual and Newton loop reused three times |
| [`src/simulation/methods/abba/_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py) | Displaced copies, exact stage differentiation, and `_ProjectedStep` |
| [`src/simulation/methods/abba/_core.py`](../../../../src/simulation/methods/abba/_core.py) | Endpoint-time A--B--B--A stage map |
| [`src/simulation/_fixed.py`](../../../../src/simulation/_fixed.py) | Output-independent main grid and shadow sampling |
| [`src/simulation/observation.py`](../../../../src/simulation/observation.py) | Per-substep snapshots and complete-step observation records |
| [`src/simulation/runner.py`](../../../../src/simulation/runner.py) | Public input validation and `Solution` construction |

## Public boundary and default configuration

[`InitialValueProblem`](../../../../src/simulation/problem.py) binds a planar
guiding-centre-Jacobian dynamics object to a validated initial configuration.
[`SimulationRequest`](../../../../src/simulation/request.py) supplies the time
span, maximum main step, and requested output times. The convenience
`simulate(...)` function creates a
[`SimulationRunner`](../../../../src/simulation/runner.py), which validates the
problem, method, and request before calling `ABBA4Implicit.integrate(...)`.

`ABBA4Implicit` inherits the frozen `_ABBAImplicitConfig`. Its defaults are:

| Field | Default | Role in the scoped path |
|---|---:|---|
| `projection_formulation` | `"reduced_multiplier"` | Selects `_solve_reduced_multiplier_step(...)` for all three substeps |
| `state_extension` | `"physical"` | Carries only the accepted physical packed state |
| `nonlinear_solver` | `"newton"` | Uses exact stage differentiation for every substep solve |
| `newton_absolute_tolerance` | `1e-13` | Absolute contribution to each substep threshold |
| `newton_relative_tolerance` | `1e-12` | Substep-state-scaled threshold contribution |
| `newton_max_iterations` | `12` | Maximum corrections for each independent substep solve |
| `progress` | `False` | Enables the shared main-grid progress display |
| `step_observer` | `None` | Receives complete accepted outer-step events |

The physical Newton path requires dynamics satisfying
[`GuidingCenterJacobianSystem`](../../../../src/dynamics/protocols.py),
`state_dimension == 2`, and finite particle Jacobians with shape `(N,2,2)`.
The packed convention is component-major:

```text
[x_1, ..., x_N, y_1, ..., y_N].
```

For `N` particles the accepted state, each multiplier, and the reduced
nonlinear unknown have dimension `2N`; each duplicated ABBA state has dimension
`4N`. Configuration selection is global: all three substeps use the same
formulation, solver, extension, and tolerances.

## Triple-jump coefficients

[`_ABBA4_COEFFICIENTS`](../../../../src/simulation/methods/_abba_coefficients.py)
contains

\[
\gamma=\frac{1}{2-\sqrt[3]{2}},
\qquad
\delta=-\frac{\sqrt[3]{2}}{2-\sqrt[3]{2}},
\]

so that

\[
(c_1,c_2,c_3)=(\gamma,\delta,\gamma)
\approx(1.35120719196,-1.70241438392,1.35120719196).
\]

The coefficients are palindromic and satisfy the composition conditions

\[
2\gamma+\delta=1,
\qquad
2\gamma^3+\delta^3=0.
\]

The first identity advances the signed composition by one complete outer step.
The second cancels the leading third-order defect of a symmetric second-order
base map, producing the designed fourth-order triple jump when the projected
submaps are solved to their ideal roots.

Let `P_(q,T)` denote one complete projected ABBA map with signed duration `q`
starting at time `T`. Chronologically, one outer step is

\[
\begin{aligned}
z_1 &= P_{\gamma h,t_n}(z_n),\\
z_2 &= P_{\delta h,t_n+\gamma h}(z_1),\\
z_{n+1} &=
P_{\gamma h,t_n+(\gamma+\delta)h}(z_2).
\end{aligned}
\]

Equivalently,

\[
\Psi_h^{[4]}=
P_{\gamma h,t_n+(\gamma+\delta)h}
\circ P_{\delta h,t_n+\gamma h}
\circ P_{\gamma h,t_n}.
\]

The signed clock is not monotone inside the outer step:

| Substep | Start | Duration | End |
|---|---|---|---|
| 1 | `t_n` | `gamma h > 0` | `t_n + gamma h` |
| 2 | `t_n + gamma h` | `delta h < 0` | `t_n + (gamma + delta)h` |
| 3 | `t_n + (gamma + delta)h` | `gamma h > 0` | `t_n + h` |

Because `gamma + delta` is negative, the middle map crosses back past `t_n`
before the last positive map reaches `t_n+h`. This signed start-time sequence
is required for a nonautonomous fourth-order composition; replacing it with
three evaluations at the outer start time would define a different method.

`_solve_composed_abba_step(...)` advances `current_time` by each signed
duration, feeds every accepted physical state into the next solve, stores an
`_AcceptedSubstep`, and verifies within floating-point tolerance that the final
signed time equals `t_n+h`. The returned `_ComposedABBAStep` contains the final
state and the ordered tuple of three accepted records.

## Projection after every signed substep

The composition policy is the literal diagnostic value
`"project_each_abba_substep"`. For a generic substep starting at `T` with
signed duration `q`, the reduced solver initializes a fresh multiplier

\[
\mu_0=0,
\qquad
u_0=z+\mu,
\qquad
v_0=z-\mu.
\]

With `s=q/2`, the endpoint-time A--B--B--A stages are

\[
\begin{aligned}
u_1 &=u_0+s f(T,v_0),\\
v_1 &=v_0+s f(T,u_1),\\
v_f &=v_1+s f(T+q,u_1),\\
u_f &=u_1+s f(T+q,v_f).
\end{aligned}
\]

These equations use the signed `q`; for the middle substep both `s` and the
endpoint displacement are negative. The reduced Hairer residual is

\[
r(\mu)=u_f(\mu)-v_f(\mu)+2\mu,
\]

and its independent stopping threshold is

\[
\tau_j=\mathrm{atol}+\mathrm{rtol}\,
\max\left(1,\lVert z_{j-1}\rVert_\infty\right).
\]

The scale therefore belongs to that substep's own accepted input, not always
to the outer `z_n`.

For exact particle Jacobians `W_1,...,W_4` evaluated at the four signed stage
points and `S=W_2+W_3`, `_differentiate_stages(...)` assembles

\[
J_r=4I-s(W_1+W_2+W_3+W_4)
    +s^2(W_4S+SW_1)-s^3W_4SW_1.
\]

The ordered products must not be commuted. Each Newton correction solves

\[
J_r(\mu_k)\Delta\mu_k=r(\mu_k),
\qquad
\mu_{k+1}=\mu_k-\Delta\mu_k,
\]

as independent `2 x 2` particle blocks. Once
`||r(mu_star)||_infinity <= tau_j`, the corrected copies are

\[
u^+=u_f+\mu_\star,
\qquad
v^+=v_f-\mu_\star,
\]

and the substep returns

\[
z^+=\frac{u^++v^+}{2}.
\]

This accepted `z^+` becomes the input to the next signed substep. The
multiplier and duplicated stages do not remain live across substep boundaries;
the next solve starts again from `mu_0=0` around its new physical state.

This placement is numerically distinct from
`ABBA4ImplicitSingleProjection`, which keeps the duplicated copies through the
entire unprojected triple jump and solves one projection around that complete
composition. Projection placement is a method distinction, not a fourth
configuration axis.

## Fixed main grid and shadow compositions

[`integrate_fixed_grid(...)`](../../../../src/simulation/_fixed.py) chooses the
smallest uniform main-step count whose internal outer step does not exceed
`request.max_step`:

\[
h_{\mathrm{main}}=\frac{t_f-t_0}{\text{step_count}}.
\]

For every main interval it calls the nested `advance(...)` callback with
`observe=True`. That callback runs the complete three-solve triple jump,
replaces the main state, records diagnostics, advances progress, and may emit
one complete-step observation.

An output time inside a main interval triggers an independent shadow
composition from a copy of the preceding main node. The shadow outer duration
is the distance to that output time, but it still expands into
`(gamma q, delta q, gamma q)` and performs three projected Newton solves.
The sample is stored with `observe=False`: it does not replace the main state,
affect later steps, contribute diagnostic rows, advance progress, or emit an
observation. Output times at main endpoints reuse the corresponding main state.

Consequently, changing the saved-time schedule changes shadow work but not the
underlying main trajectory. A failed shadow solve can still abort the run;
shadow status suppresses recording, not numerical validation.

## Aggregated diagnostics

The coordinator first forms one row per accepted main step and one column per
signed substep. It then returns
[`IntegrationData`](../../../../src/simulation/_result.py) with the physical
history and these diagnostics:

| Key | Default physical meaning |
|---|---|
| `step_count` | Number of accepted uniform main steps |
| `implicit_substeps_per_step` | `3` |
| `nonlinear_solves_per_step` | `3` |
| `composition_coefficients` | `(gamma, delta, gamma)` |
| `composition_policy` | `"project_each_abba_substep"` |
| `nonlinear_solver` | `"newton"` |
| `projection_formulation` | `"reduced_multiplier"` |
| `substep_projection_formulation` | `"reduced_multiplier"` for all three columns |
| `state_extension` | `"physical"` |
| `nonlinear_iterations` | Row-wise sum of the three correction counts |
| `residual_evaluations` | Row-wise sum of the three residual-evaluation counts |
| `nonlinear_residual_norms` | Residual of the substep with the largest `residual/tolerance` ratio |
| `nonlinear_tolerances` | Tolerance from that same selected substep |
| `projection_multiplier_norms` | Row-wise maximum multiplier infinity norm |
| `substep_nonlinear_iterations` | Array with shape `(step_count,3)` |
| `substep_residual_evaluations` | Array with shape `(step_count,3)` |
| `substep_nonlinear_residual_norms` | Array with shape `(step_count,3)` |
| `substep_nonlinear_tolerances` | Array with shape `(step_count,3)` |
| `substep_projection_multiplier_norms` | Array with shape `(step_count,3)` |
| `nonlinear_absolute_tolerance` | Configured absolute tolerance |
| `nonlinear_relative_tolerance` | Configured relative tolerance |
| `nonlinear_max_iterations` | Configured limit applied separately to each solve |
| `accepted_internal_state_dimension` | `2N` |
| `base_splitting_state_dimension` | `4N` |
| `nonlinear_unknown_dimension` | `2N` per solve |
| `observer_state_dimension` | `2N` |
| `observer_state_kind` | `"physical_map"` |

The selected residual and tolerance deliberately come from one consistent
substep. Selecting their independent absolute maxima could report a ratio that
never occurred. Compatibility keys `newton_iterations`,
`newton_residual_norms`, `newton_absolute_tolerance`,
`newton_relative_tolerance`, and `newton_max_iterations` mirror the general
nonlinear values.

Shadow compositions contribute no rows. The counts represent nonlinear work;
they do not include the vector-field and Jacobian evaluations hidden inside
each residual or Newton assembly.

## Complete-step and substep observations

For every accepted main step, `_substep_observation(...)` converts the three
`_AcceptedSubstep` records into three ordered
[`ABBA2ImplicitIntegrationStep`](../../../../src/simulation/observation.py)
snapshots. Each snapshot contains:

- its signed start time, end time, and duration;
- continuous `state_before` and `state_after` snapshots;
- its own fixed-time projected `map_state` callable;
- multiplier and converged ABBA stage copies; and
- its own correction count, residual evaluations, residual, tolerance, and
  multiplier norm.

These three records are always used to aggregate main-step diagnostics. They
are not sent separately to the configured callback.

When `step_observer` is set, `advance(...)` additionally emits one
[`ABBA4ImplicitIntegrationStep`](../../../../src/simulation/observation.py).
It inherits `ABBAImplicitCompositionIntegrationStep`,
`ImplicitIntegrationStep`, and `IntegrationStep`, and contains:

- the complete outer `state_before`, `state_after`, start time, end time, and
  duration;
- `map_state`, which reevaluates all three signed projected solves;
- a copy of the three composition coefficients;
- the ordered tuple of three substep snapshots;
- summed correction and residual-evaluation counts;
- the residual and tolerance from the substep with the worst normalized
  residual; and
- the maximum substep multiplier norm.

The substep states are continuous: the first begins at the outer input, each
accepted output equals the next input, and the third output equals the outer
result. Shadow compositions construct no observation snapshots and emit no
event.

Downstream diagnostics can form the exact ideal-root tangent of each substep
from its multiplier and stage snapshots, then multiply the three physical
Jacobians in chronological order. The observation stores solver data; it does
not itself perform tangent or symplecticity analysis.

## Public result boundary

After `_integrate_composed_implicit_abba(...)` returns `IntegrationData`, the
existing runner call verifies requested times, history shape, finiteness, the
unchanged initial sample, and source-layout compatibility. It then constructs
[`Solution`](../../../../src/simulation/solution.py), which owns read-only
copies of times, states, and diagnostic arrays and retains the source initial
configuration for component and position interpretation.

## Derivation, properties, and limitations

The full fourth-order construction is derived in
[`ABBA4 implicit theory`](../tex/theory.tex) and its
[`compiled PDF`](../tex/theory.pdf). Focused companion notes cover the
[`simultaneous state--multiplier formulation`](../tex/simultaneous-formulation.tex),
the [`Jacobian formula summary`](../tex/jacobian-formula-summary.tex), and the
[`Jacobian diagnostic workflow`](../tex/jacobian-diagnostics.tex); each source
has a same-named compiled PDF beside it. These documents specialize the ABBA2
projected kernel to the three independent roots and signed durations of the
fourth-order composition.

The focused contracts in
[`tests/test_abba4_implicit.py`](../../../../tests/test_abba4_implicit.py)
verify the coefficient identities, negative central duration, continuous
signed start times, three-column diagnostics, fourth-order refinement,
reversibility, nonautonomous signed-time behavior, and the composed ideal-root
Jacobian. The family tests in
[`tests/test_abba_configuration_cube.py`](../../../../tests/test_abba_configuration_cube.py)
verify all configuration axes and that the aggregated residual and tolerance
come from the same normalized-worst substep.

The following limitations apply to the scoped diagram:

- It expands only physical reduced Newton. The simultaneous formulation,
  Broyden, shared-time normalization, and fully extended kernel are linked
  through the canonical family document rather than repeated here.
- The fixed grid is not adaptive. A singular residual Jacobian or a substep
  that exceeds its iteration limit aborts the outer composition; there is no
  smaller-step retry.
- The negative central duration is intrinsic to the real fourth-order
  triple-jump coefficients. Dynamics and potentials must support evaluation at
  every resulting signed intermediate time.
- Fourth-order, reversibility, and physical symplecticity are ideal-map
  properties. Finite nonlinear tolerances introduce corresponding defects and
  can eventually limit observed refinement.
- This method performs three projections per outer step and is not equivalent
  to `ABBA4ImplicitSingleProjection`.
- The physical branch carries no conjugate momentum and publishes no
  generalized-energy diagnostic.
- Potential construction, downstream diagnostic algorithms, and
  experiment-specific parameter assembly remain outside this simulation
  diagram.

## Minimal public usage

```python
from simulation import ABBA4Implicit, SimulationRequest, simulate

solution = simulate(
    problem,
    ABBA4Implicit(
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
