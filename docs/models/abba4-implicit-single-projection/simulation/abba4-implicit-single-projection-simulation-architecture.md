# ABBA4ImplicitSingleProjection physical reduced-Newton architecture

## Scope

This companion documents one concrete public configuration:

```python
from simulation import ABBA4ImplicitSingleProjection

method = ABBA4ImplicitSingleProjection(
    projection_formulation="reduced_multiplier",
    state_extension="physical",
    track_energy=False,
    nonlinear_solver="newton",
    newton_absolute_tolerance=1e-13,
    newton_relative_tolerance=1e-12,
    newton_max_iterations=12,
)
```

These are the class defaults. The defining feature is projection placement:
one symmetric implicit projection surrounds the complete unprojected
`(gamma, delta, gamma)` triple jump. The two physical copies pass continuously
from one signed A-B-B-A factor to the next; they are not projected or averaged
between factors.

The authoritative description of the complete ABBA family, including the
simultaneous formulation, Broyden, optional physical energy tracking, and fully
extended state, is
[`docs/models/abba/simulation/abba-numerical-architecture.md`](../../abba/simulation/abba-numerical-architecture.md).
Those axes are intentionally not duplicated here.

## Physical contract and dimensions

The default path requires a planar `GuidingCenterJacobianSystem` with
`state_dimension == 2`. For `N` independent particles, the component-major
physical state is

\[
z=[x_1,\ldots,x_N,y_1,\ldots,y_N]^T\in\mathbb R^{2N}.
\]

The dynamics supplies

- `vector_field(t, state)` for every A-B-B-A shear; and
- `particle_vector_field_jacobians(t, state)` with shape `(N, 2, 2)` for the
  exact Newton blocks.

The physical accepted state and reduced multiplier both have dimension `2N`.
The unprojected base map acts on two copies `(u, v)` with total dimension `4N`.
No time or conjugate-momentum coordinate is appended in this scope.

## One signed unprojected A-B-B-A factor

Write `A_(s,t)` for the endpoint-time map of signed duration `s`, beginning at
time `t`. From two input copies `(u_0,v_0)`, the implementation evaluates

\[
\begin{aligned}
u_1 &= u_0 + \frac{s}{2}f(t,v_0), \\
v_1 &= v_0 + \frac{s}{2}f(t,u_1), \\
v_f &= v_1 + \frac{s}{2}f(t+s,u_1), \\
u_f &= u_1 + \frac{s}{2}f(t+s,v_f).
\end{aligned}
\]

Every field value must be finite and retain the input shape. The private
`_ABBAStages` snapshot stores `u_initial`, `v_initial`, `u_first`, `v_final`,
and `u_final`; `v_1` is an implementation-local intermediate.

## Continuous unprojected triple jump

The fourth-order coefficients are

\[
\gamma=\frac{1}{2-\sqrt[3]{2}},\qquad
\delta=-\frac{\sqrt[3]{2}}{2-\sqrt[3]{2}},\qquad
2\gamma+\delta=1.
\]

For an outer step of duration `h` beginning at `t_n`, the base composition is

\[
\mathcal B_h(t_n)=
\mathcal A_{\gamma h,\,t_n+(\gamma+\delta)h}
\circ
\mathcal A_{\delta h,\,t_n+\gamma h}
\circ
\mathcal A_{\gamma h,\,t_n}.
\]

Evaluation follows the right-to-left order in this expression. The output of
each factor is the exact input of the next factor. In particular:

1. the first factor advances by `gamma h`;
2. the negative middle factor advances by `delta h`; and
3. the final factor advances by `gamma h` and ends at `t_n + h`.

There is no intermediate projection, diagonal re-embedding, or arithmetic
mean. The implementation labels this base composition
`"unprojected_abba4_triple_jump"`.

## One reduced projection around the complete base

For one particle, define the diagonal embedding, normal embedding, and copy
difference operators

\[
E=\begin{pmatrix}I\\I\end{pmatrix},\qquad
N=\begin{pmatrix}I\\-I\end{pmatrix},\qquad
G=\begin{pmatrix}I&-I\end{pmatrix}=N^T.
\]

The same block notation applies independently to every particle. At Newton
iterate `mu_k`, the two copies are displaced once, before all three factors:

\[
X_0(\mu_k)=Ez_n+N\mu_k
=\begin{pmatrix}z_n+\mu_k\\z_n-\mu_k\end{pmatrix}.
\]

Let

\[
X_b(\mu_k)=\mathcal B_h(t_n)X_0(\mu_k)
=\begin{pmatrix}u_b(\mu_k)\\v_b(\mu_k)\end{pmatrix}.
\]

The final normal correction is again `(+mu_k,-mu_k)`. Enforcing equality of
the corrected output copies gives the reduced residual

\[
\begin{aligned}
r(\mu_k)
&=G X_b(\mu_k)+2\mu_k \\
&=u_b(\mu_k)-v_b(\mu_k)+2\mu_k.
\end{aligned}
\]

At convergence, the corrected copies are

\[
u^+=u_b+\mu,\qquad v^+=v_b-\mu,
\]

and the accepted physical state is their neutral mean,

\[
z_{n+1}=\frac{u^++v^+}{2}=\frac{u_b+v_b}{2}.
\]

The multiplier cancels algebraically in the final mean, but it changes the
displaced input and therefore the complete base trajectory and accepted state.

## Ordered analytic Newton system

The initial guess and stopping threshold are

\[
\mu_0=0,\qquad
\tau=\operatorname{atol}+\operatorname{rtol}
\max(1,\lVert z_n\rVert_\infty).
\]

Let `M_j` be the exact doubled `4 x 4` tangent block for factor `j`, evaluated
at that factor's stored stage states and signed time interval. The derivative
of the complete base map is accumulated in execution order as

\[
M_b=M_3M_2M_1.
\]

This order is mandatory: the stage and factor Jacobians generally do not
commute. If

\[
M_b=
\begin{pmatrix}
M_{uu}&M_{uv}\\
M_{vu}&M_{vv}
\end{pmatrix},
\]

then the reduced residual Jacobian is

\[
\begin{aligned}
K(\mu_k)
&=G M_b N+2I \\
&=M_{uu}-M_{uv}-M_{vu}+M_{vv}+2I.
\end{aligned}
\]

The implementation forms one `K` block with shape `(2, 2)` per particle and
solves them as one NumPy batch:

\[
K(\mu_k)\,\Delta\mu_k=r(\mu_k),\qquad
\mu_{k+1}=\mu_k-\Delta\mu_k.
\]

Every new multiplier rebuilds the displaced copies and reevaluates all three
unprojected factors. For an accepted Newton step,
`residual_evaluations = nonlinear_iterations + 1`: the initial residual is
counted, while `nonlinear_iterations` counts corrections.

One outer step therefore owns exactly one nonlinear root solve, but one
residual evaluation traverses three A-B-B-A maps and Newton may perform several
such evaluations.

## Difference from ABBA4Implicit

Both classes use the same signed coefficient sequence, but they implement
different numerical maps:

| Method | Projection placement | Nonlinear solves per outer step | State passed between signed factors |
|---|---|---:|---|
| `ABBA4ImplicitSingleProjection` | One projection around the complete triple jump | 1 | The two unprojected copies `(u,v)` |
| `ABBA4Implicit` | One projection after each signed factor | 3 | A projected physical state re-embedded for the next factor |

Thus the single-projection method is not merely a cheaper execution of
`ABBA4Implicit`. The focused tests explicitly verify that their finite-step
states differ on a nonlinear problem.

## Fixed grid and shadow steps

`_integrate_abba4_implicit_single_projection` delegates scheduling to the
shared `integrate_fixed_grid` runner:

1. It advances an output-independent uniform main grid with step size no larger
   than `SimulationRequest.max_step`.
2. Every main callback performs the complete outer Newton solve, updates the
   diagnostic arrays, and may emit one observation.
3. An off-grid saved time is evaluated by a shorter shadow step from the
   preceding main node.
4. A shadow callback still performs the complete triple-jump solve, but
   `observe=False`: it does not modify the main trajectory, append work
   diagnostics, update progress, or emit an observation.

Consequently, `step_count` and every per-step diagnostic array describe main
steps only. Requested sample density can add shadow work without changing the
accepted main-grid path.

## Diagnostic keys

For this default physical reduced-Newton configuration, the returned
`IntegrationData.diagnostics` contains the following exact keys.

### Composition and selectors

| Key | Value or shape |
|---|---|
| `step_count` | Number of main-grid steps |
| `implicit_substeps_per_step` | `1`, meaning one implicit outer projection |
| `nonlinear_solves_per_step` | `1` |
| `unprojected_abba_maps_per_step` | `3`, the factors in the accepted base composition |
| `unprojected_abba_maps_per_residual_evaluation` | `3` |
| `composition_coefficients` | NumPy copy of `(gamma, delta, gamma)` |
| `base_composition` | `"unprojected_abba4_triple_jump"` |
| `projection_placement` | `"around_complete_base_composition"` |
| `nonlinear_solver` | `"newton"` |
| `projection_formulation` | `"reduced_multiplier"` |
| `state_extension` | `"physical"` |
| `track_energy` | `False` |

`unprojected_abba_maps_per_step` describes the three-factor topology; actual
base-map work also depends on `residual_evaluations`.

### Main-step work arrays

Each of these arrays has shape `(step_count,)`:

- `nonlinear_iterations`;
- `residual_evaluations`;
- `nonlinear_residual_norms`;
- `nonlinear_tolerances`; and
- `projection_multiplier_norms`.

The comparison-compatible one-solve forms have shape `(step_count, 1)`:

- `substep_nonlinear_iterations`;
- `substep_residual_evaluations`;
- `substep_nonlinear_residual_norms`;
- `substep_nonlinear_tolerances`; and
- `substep_projection_multiplier_norms`.

### Solver controls and compatibility names

The normalized controls are

- `nonlinear_absolute_tolerance`;
- `nonlinear_relative_tolerance`; and
- `nonlinear_max_iterations`.

Compatibility keys retain the established Newton names:

- `newton_iterations` aliases the main `nonlinear_iterations` values;
- `newton_residual_norms` aliases `nonlinear_residual_norms`; and
- `newton_absolute_tolerance`, `newton_relative_tolerance`, and
  `newton_max_iterations` repeat the configured controls.

### State-space metadata

For `N` physical particles:

| Key | Value |
|---|---:|
| `accepted_internal_state_dimension` | `2N` |
| `base_splitting_state_dimension` | `4N` |
| `observer_state_dimension` | `2N` |
| `observer_state_kind` | `"physical_map"` |
| `nonlinear_unknown_dimension` | `2N` |

## Main-step observation

When `step_observer` is configured, each main step emits one
`ABBA4ImplicitSingleProjectionIntegrationStep`. It contains

- the method and dynamics names, step index, start time, end time, and outer
  duration;
- independent physical `state_before` and `state_after` copies;
- `map_state`, which reruns the same fixed-time, fixed-duration outer solve on
  another physical candidate;
- `formulation_name`, `nonlinear_solver`, iteration count, residual-evaluation
  count, residual norm, tolerance, and multiplier norm;
- the exact dynamics instance and converged outer `multiplier`;
- a copy of `(gamma, delta, gamma)`; and
- three ordered `UnprojectedABBAIntegrationStep` records.

Each unprojected record stores its signed start time, end time, duration, and
the five exposed A-B-B-A stage snapshots. The middle duration is negative.
These substeps have no individual multiplier or nonlinear solve; their outputs
remain off the diagonal and feed the next record directly.

The observer stores data rather than an accepted-map Jacobian. The diagnostic
helper `abba4_implicit_single_projection_step_particle_jacobians` reconstructs
the ordered base tangent from these snapshots and differentiates the ideal
outer root. Shadow steps emit no record.

## Verified behavior

`tests/test_abba4_single_projection.py` checks the focused contracts:

- fourth-order refinement on an exactly solvable non-autonomous rotation;
- one nonlinear solve, one observation per main step, three ordered base
  substeps, and a negative middle duration;
- the exact reduced residual Jacobian against centered differences;
- the reconstructed accepted-map Jacobian against centered differences;
- fifth-order scaling of the outer multiplier under step halving;
- tight forward/backward reversibility and a finite-step result distinct from
  `ABBA4Implicit`; and
- work normalization by three unprojected maps per residual or tangent
  evaluation.

These tests use tight nonlinear tolerances. They do not turn a finite-tolerance
solve into an exact algebraic root.

## Limitations and failure modes

- Newton has no finite-difference fallback. The dynamics must provide finite
  exact independent-particle Jacobian blocks at every traversed stage state.
- The signed triple jump is non-monotone in time. Because `gamma > 1` and
  `delta < 0`, internal evaluations can lie outside the outer interval
  `[t_n,t_n+h]`; the dynamics must be defined there.
- The method uses a uniform fixed main grid and has no local-error estimator,
  adaptive controller, or rejected-step recovery.
- The corrected copies satisfy the diagonal constraint only to `tau`. Observed
  reversibility, symplecticity, and other ideal-root properties can therefore
  depend on nonlinear tolerances and round-off.
- A singular reduced block or failure to meet `tau` within
  `newton_max_iterations` raises `RuntimeError`; there is no automatic solver
  switch.
- One nonlinear solve does not imply one residual evaluation: each Newton
  correction repeats the entire unprojected triple jump and its ordered
  tangent construction.
- Physical energy tracking, fully extended, simultaneous, and Broyden behavior
  belongs to the canonical family document linked in the scope section.

## Files

- [`src/simulation/methods/abba/order4_implicit_single_projection.py`](../../../../src/simulation/methods/abba/order4_implicit_single_projection.py)
- [`src/simulation/methods/abba/_core.py`](../../../../src/simulation/methods/abba/_core.py)
- [`src/simulation/methods/abba/_projection_common.py`](../../../../src/simulation/methods/abba/_projection_common.py)
- [`src/simulation/methods/_abba_coefficients.py`](../../../../src/simulation/methods/_abba_coefficients.py)
- [`src/simulation/_fixed.py`](../../../../src/simulation/_fixed.py)
- [`src/simulation/observation.py`](../../../../src/simulation/observation.py)
- [`src/diagnostics/trajectory_symplecticity/jacobians.py`](../../../../src/diagnostics/trajectory_symplecticity/jacobians.py)
- [`tests/test_abba4_single_projection.py`](../../../../tests/test_abba4_single_projection.py)
- [`Companion PlantUML source`](abba4-implicit-single-projection-simulation-architecture.puml)
