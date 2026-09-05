# ABBA6 implicit default physical simulation architecture

This document explains the companion
[`abba6-implicit-simulation-architecture.puml`](abba6-implicit-simulation-architecture.puml)
diagram. Its expanded runtime path is exactly the default numerical
configuration:

```python
ABBA6Implicit(
    projection_formulation="reduced_multiplier",
    state_extension="physical",
    nonlinear_solver="newton",
)
```

`ABBA6Implicit` also supports a simultaneous residual, Broyden, shared time,
and a fully extended state. Those axes are intentionally not expanded here.
Their contracts and the complete 51-configuration family are documented in
the authoritative
[`Canonical ABBA numerical architecture`](../../abba/simulation/abba-numerical-architecture.md).

## Scoped runtime path

The solid path in the diagram is:

```text
simulate(problem, ABBA6Implicit(), request)
        |
        v
SimulationRunner -> ABBA6Implicit.integrate(...)
        |
        v
_integrate_composed_implicit_abba(...)
        |
        v
integrate_fixed_grid(...) -> advance(...)
        |
        v
seven signed projected ABBA2 substeps
        |
        +-> one reduced Newton solve after each A-B-B-A map
        |
        v
aggregate main-step diagnostics and optional observation
        |
        v
IntegrationData -> SimulationRunner -> Solution
```

The public runtime calls the generic `_solve_composed_abba_step(...)` with the
ABBA6 coefficients. The private `_solve_abba6_step(...)` is an equivalent
focused entry used by numerical tests; it is not an extra stage in the public
call path.

## Principal files

| File | Responsibility in this scoped path |
|---|---|
| [`order6_implicit.py`](../../../../src/simulation/methods/abba/order6_implicit.py) | Public method, coefficient selection, and physical/extended dispatch |
| [`_abba_coefficients.py`](../../../../src/simulation/methods/_abba_coefficients.py) | Yoshida's seven real composition coefficients |
| [`order4_implicit.py`](../../../../src/simulation/methods/abba/order4_implicit.py) | Shared signed-composition coordinator, aggregation, and observation construction |
| [`_projection_reduced.py`](../../../../src/simulation/methods/abba/_projection_reduced.py) | Reduced multiplier residual and Newton loop |
| [`_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py) | Displaced copies and exact residual Jacobian |
| [`_core.py`](../../../../src/simulation/methods/abba/_core.py) | Endpoint-time A--B--B--A stage map |
| [`_fixed.py`](../../../../src/simulation/_fixed.py) | Output-independent main grid and shadow samples |
| [`observation.py`](../../../../src/simulation/observation.py) | Outer and constituent accepted-step records |
| [`runner.py`](../../../../src/simulation/runner.py) | Public validation and `Solution` construction |

## Public configuration and dynamics boundary

`ABBA6Implicit` is a frozen subclass of the shared private
`_ABBAImplicitConfig`. The default physical Newton branch uses these fields:

| Field | Default | Scoped role |
|---|---:|---|
| `projection_formulation` | `"reduced_multiplier"` | Solves one multiplier per constituent ABBA map |
| `state_extension` | `"physical"` | Carries only the accepted packed physical state |
| `nonlinear_solver` | `"newton"` | Uses exact residual differentiation |
| `newton_absolute_tolerance` | `1e-13` | Absolute stopping contribution for each substep |
| `newton_relative_tolerance` | `1e-12` | Substep-state-scaled stopping contribution |
| `newton_max_iterations` | `12` | Maximum corrections in each of seven solves |
| `progress` | `False` | Enables the shared main-grid progress display |
| `step_observer` | `None` | Receives one outer event per accepted main step |

The physical Newton coordinator requires a planar
[`GuidingCenterJacobianSystem`](../../../../src/dynamics/protocols.py):

- `state_dimension == 2`;
- `vector_field(t, state)` returns a finite shape-preserving packed field; and
- `particle_vector_field_jacobians(t, state)` returns finite `(N, 2, 2)`
  blocks.

For `N` independent particles, the physical convention is

```text
z = [x_1, ..., x_N, y_1, ..., y_N] in R^(2N).
```

One reduced multiplier also has dimension `2N`. Each Newton correction is
solved as `N` independent `2 x 2` systems, not as one dense `2N x 2N` solve.

## Yoshida seven-stage composition

Let `P_h(t, z)` denote one complete projected implicit ABBA2 map: it starts at
time `t`, advances by signed duration `h`, and returns one physical state. The
implementation stores Yoshida's coefficients as

\[
\begin{aligned}
a &= 0.78451361047755726382,\\
b &= 0.23557321335935813368,\\
c &= -1.17767998417887100695,\\
d &= 1.31518632068391121889,
\end{aligned}
\]

\[
(w_1,\ldots,w_7)=(a,b,c,d,c,b,a).
\]

The stored binary64 values are palindromic and satisfy, to floating-point
round-off,

\[
\sum_{j=1}^{7}w_j=1,
\qquad
\sum_{j=1}^{7}w_j^3=0,
\qquad
\sum_{j=1}^{7}w_j^5=0.
\]

For an outer state `z_n`, start time `t_n`, and duration `h`, the coordinator
sets

\[
z^{(0)}=z_n,
\qquad
t_j=t_n+h\sum_{\ell<j}w_\ell,
\qquad
z^{(j)}=P_{w_jh}(t_j,z^{(j-1)}),
\]

and returns `z_(n+1) = z^(7)`. The coefficient sum is checked indirectly by
requiring the accumulated substep time to agree with `t_n+h` within a scaled
floating-point tolerance.

The signed time path is non-monotone. In fractions of the outer step, its
constituent intervals are:

| Substep | `w_j` | Start fraction | End fraction | Direction |
|---:|---:|---:|---:|---|
| 1 | `0.7845136104775573` | `0` | `0.7845136104775573` | Forward |
| 2 | `0.23557321335935813` | `0.7845136104775573` | `1.0200868238369154` | Forward |
| 3 | `-1.177679984178871` | `1.0200868238369154` | `-0.15759316034195558` | Backward |
| 4 | `1.3151863206839112` | `-0.15759316034195558` | `1.1575931603419556` | Forward |
| 5 | `-1.177679984178871` | `1.1575931603419556` | `-0.020086823836915402` | Backward |
| 6 | `0.23557321335935813` | `-0.020086823836915402` | `0.21548638952244273` | Forward |
| 7 | `0.7845136104775573` | `0.21548638952244273` | `1` | Forward |

Substeps 3 and 5 therefore run backward. Intermediate field evaluations also
occur slightly before `t_n` and after `t_n+h`; a non-autonomous dynamics must
be valid at those times. The endpoint-time stage kernel receives the signed
duration unchanged, so it naturally exchanges its chronological endpoints on
a backward substep.

## One reduced projected substep

Every one of the seven substeps starts a new independent projection problem.
The converged multiplier from one substep is not used as the initial guess for
the next: each solve begins with

\[
\mu_0=0.
\]

For current physical state `z`, signed duration `q=w_j h`, start time `t`, and
`s=q/2`, nonlinear iterate `mu_k` displaces the two copies as

\[
u_0=z+\mu_k,
\qquad
v_0=z-\mu_k.
\]

The common endpoint-time A--B--B--A kernel applies

\[
\begin{aligned}
u_1 &= u_0+s f(t,v_0),\\
v_1 &= v_0+s f(t,u_1),\\
v_f &= v_1+s f(t+q,u_1),\\
u_f &= u_1+s f(t+q,v_f).
\end{aligned}
\]

Hairer's reduced residual is

\[
r(\mu_k)=u_f(\mu_k)-v_f(\mu_k)+2\mu_k,
\]

with one threshold per constituent state,

\[
\tau_j=\operatorname{atol}+\operatorname{rtol}
\max\left(1,\lVert z^{(j-1)}\rVert_\infty\right).
\]

The substep converges when `||r(mu_k)||_infinity <= tau_j`. Negative duration
does not change this norm or tolerance rule.

## Exact Newton correction

For a non-converged residual, the default path evaluates four exact particle
field Jacobians at the traversed stages:

\[
\begin{aligned}
W_1&=D_zf(t,v_0), & W_2&=D_zf(t,u_1),\\
W_3&=D_zf(t+q,u_1), & W_4&=D_zf(t+q,v_f).
\end{aligned}
\]

With `S=W_2+W_3`, the implemented reduced residual Jacobian is

\[
J_r=4I-s(W_1+W_2+W_3+W_4)
    +s^2(W_4S+SW_1)-s^3W_4SW_1.
\]

Matrix order is retained because the stage Jacobians need not commute. For
each particle, Newton solves

\[
J_r(\mu_k)\,\Delta\mu_k=r(\mu_k),
\qquad
\mu_{k+1}=\mu_k-\Delta\mu_k,
\]

then reconstructs both displaced copies and repeats all four stages. A root at
`mu_0` records zero corrections and one residual evaluation. A singular block
or failure after the configured correction limit raises `RuntimeError`; the
outer fixed grid does not retry with a smaller step.

At convergence the output correction is

\[
u^+=u_f+\mu_\star,
\qquad
v^+=v_f-\mu_\star,
\qquad
z^{(j)}=\frac{u^++v^+}{2}.
\]

That accepted physical state immediately becomes the input to substep `j+1`.
Thus `composition_policy="project_each_abba_substep"` means seven projections
and seven nonlinear solves per outer step, rather than one projection around
the complete seven-map composition.

The ABBA2 derivation and ideal-root geometric argument are not duplicated
here. See
[`ABBA2 theory`](../../abba2-implicit/tex/theory.tex) and its
[`compiled PDF`](../../abba2-implicit/tex/theory.pdf).

## Fixed grid and shadow advances

[`integrate_fixed_grid(...)`](../../../../src/simulation/_fixed.py) chooses the
fewest uniform main steps whose duration does not exceed `request.max_step`:

\[
h_{\mathrm{main}}=\frac{t_f-t_0}{\text{step_count}}.
\]

Every main interval calls the composed `advance(...)` callback with
`observe=True`. A requested saved time strictly inside a main interval instead
triggers a shorter shadow outer step from a copy of the preceding main state
with `observe=False`. That shadow outer step still performs the same seven
signed projected maps and can incur seven Newton solves.

The shadow result is saved but never replaces the main state. It does not
change later main steps, progress, per-main-step diagnostic arrays, or
observations. Output density can therefore change computational work without
changing the underlying main-grid trajectory.

## Diagnostic aggregation

For `M=step_count` main steps, the default path returns the following central
diagnostics:

| Key | Shape or value | Aggregation |
|---|---|---|
| `step_count` | scalar `M` | Uniform main-grid steps |
| `implicit_substeps_per_step` | `7` | Signed ABBA maps per outer step |
| `nonlinear_solves_per_step` | `7` | Independent reduced solves per outer step |
| `composition_coefficients` | `(7,)` | Stored Yoshida coefficient vector |
| `composition_policy` | `"project_each_abba_substep"` | Projection placement |
| `nonlinear_solver` | `"newton"` | Scoped solver |
| `projection_formulation` | `"reduced_multiplier"` | Outer formulation |
| `substep_projection_formulation` | `"reduced_multiplier"` | Formulation shared by all seven solves |
| `state_extension` | `"physical"` | Scoped accepted state |

The aggregate main-step arrays have shape `(M,)`:

- `nonlinear_iterations` and compatibility alias `newton_iterations` sum the
  seven correction counts;
- `residual_evaluations` sums the seven residual-evaluation counts;
- `nonlinear_residual_norms` and `nonlinear_tolerances` select the norm and
  tolerance from the same substep with the largest `residual/tolerance` ratio;
- `newton_residual_norms` aliases that selected residual norm; and
- `projection_multiplier_norms` stores the largest multiplier infinity norm
  among the seven substeps.

The corresponding per-substep arrays have shape `(M, 7)`:

- `substep_nonlinear_iterations`;
- `substep_residual_evaluations`;
- `substep_nonlinear_residual_norms`;
- `substep_nonlinear_tolerances`; and
- `substep_projection_multiplier_norms`.

Scalar configuration keys retain
`nonlinear_absolute_tolerance`, `nonlinear_relative_tolerance`,
`nonlinear_max_iterations`, and their `newton_*` compatibility aliases. The
physical reduced path additionally reports:

| Key | Value for `N` particles |
|---|---|
| `accepted_internal_state_dimension` | `2N` |
| `base_splitting_state_dimension` | `4N` |
| `nonlinear_unknown_dimension` | `2N` per sequential solve |
| `observer_state_dimension` | `2N` |
| `observer_state_kind` | `"physical_map"` |

Shadow solves are absent from all `(M,)` and `(M, 7)` arrays.

## Outer observation and seven retained substeps

On every main step the coordinator constructs seven immutable
`ABBA2ImplicitIntegrationStep` records so it can aggregate accepted solver
data. If `step_observer` is configured, it emits one outer
[`ABBA6ImplicitIntegrationStep`](../../../../src/simulation/observation.py)
containing those records as `substeps`.

The outer event exposes:

- physical `state_before`, `state_after`, `start_time`, `time`, and outer
  `duration`;
- a copy of the seven `composition_coefficients`;
- summed correction and residual-evaluation counts;
- the residual and tolerance from the worst relative-residual substep;
- the maximum substep multiplier norm;
- the exact dynamics instance; and
- `map_state`, which resolves the same complete seven-solve map for another
  physical candidate.

Each retained constituent record has its own continuous signed start time,
duration, final time, state snapshots, multiplier, converged A--B--B--A stage
states, solver counters, and fixed substep `map_state`. Adjacent records satisfy
`substeps[j].state_after == substeps[j+1].state_before`. They are nested data,
not seven separate calls to the external observer. Shadow advances create no
outer event and no retained diagnostic row.

## Public result boundary

The coordinator returns `IntegrationData` with requested times, physical
history, and diagnostics. `SimulationRunner` verifies exact requested times,
finite state shape, the unchanged initial sample, and source-layout
compatibility before constructing the read-only public `Solution`. This
boundary is identical to the canonical ABBA result path.

## Designed property and limitations

- The palindromic composition of the symmetric second-order projected base map
  is designed for global order six. The focused rotation and non-autonomous
  polynomial tests measure refinement gains near `2^6=64`; this is empirical
  protection of the implementation, not an adaptive error guarantee.
- Sixth order assumes sufficiently smooth dynamics and projection roots solved
  accurately enough that nonlinear error does not dominate truncation error.
- The two negative substeps and central overshoot require evaluations outside
  the outer interval. Problems that are undefined, irreversible, or unstable
  under backward evolution are not suitable without additional analysis.
- Seven independent nonlinear solves make an outer or shadow step materially
  more expensive than `ABBA2Implicit`. No solve is reused across substeps.
- The main grid is fixed. A failed constituent solve aborts the step; there is
  no adaptive step rejection, retry, or automatic tolerance adjustment.
- Ideal-root reversibility and canonical geometric properties can be degraded
  by finite Newton tolerances and round-off. The method does not make a
  symplecticity claim for an arbitrary `GuidingCenterJacobianSystem` merely
  because it satisfies the structural protocol.
- This companion covers only physical + reduced multiplier + Newton. Use the
  canonical ABBA document for simultaneous, Broyden, shared-time, and fully
  extended behavior.

## Regression evidence

[`tests/test_abba6.py`](../../../../tests/test_abba6.py) verifies coefficient
palindromy and moment sums, seven continuous signed substeps, sixth-order
refinement, reversibility, non-autonomous signed stage times, and the reference
accuracy study. The composed diagnostic selection rule and canonical dimension
keys are checked in
[`tests/test_abba_configuration_cube.py`](../../../../tests/test_abba_configuration_cube.py).
Main-grid-only observation and diagnostic behavior of the shared projected
kernel is checked in [`tests/test_abba.py`](../../../../tests/test_abba.py).

The reusable accuracy experiment lives in
[`src/studies/abba6_accuracy.py`](../../../../src/studies/abba6_accuracy.py).
