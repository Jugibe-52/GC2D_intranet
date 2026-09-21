> Current state contract: the integration workspace is spatially duplicated,
> R4 without energy or R6 with time and passive momentum per planar
> particle. The detailed projection equations below still act only on space.
> See [the current guide](bm4-simulation-architecture.md) for the complete API.

# BM4Implicit implementation walkthrough

This guide explains the implementation of `BM4Implicit` in source order. It is
intended to make every declaration and non-trivial operation in
[`implicit.py`](../../../../src/simulation/methods/bm4/implicit.py) and
[`_core.py`](../../../../src/simulation/methods/bm4/_core.py) traceable to its
mathematical or architectural purpose without relying on fragile source-line
numbers.

The shorter
[`bm4-simulation-architecture.md`](bm4-simulation-architecture.md) describes the
component boundaries. This document is the detailed code-reading companion.

## Notation, storage, and shapes

Let `p` be the number of independent guiding-centre particles and let

\[
m=2p
\]

be the size of one physical state. The symbol `p` is used consistently for the
particle count so that `N` remains unambiguously available for the normal
embedding `N = G.T` below. The implementation uses component-major packing:

\[
z=(x_1,\ldots,x_p,y_1,\ldots,y_p)\in\mathbb R^m.
\]

The nonlinear kernel accepts this physical vector. The integration state stores
both spatial copies, and optionally p times and p passive momenta. BM4 works with
two physical copies,

\[
Y=\begin{pmatrix}u\\v\end{pmatrix}
 = (u_x,u_y,v_x,v_y)_{\text{component-major}}
 \in\mathbb R^{2m}=\mathbb R^{4p}.
\]

For particle-local Jacobian calculations, `_particle_blocks` views the same
packed vector as a matrix whose row for particle `i` is

\[
(u_{x,i},u_{y,i},v_{x,i},v_{y,i}).
\]

No time coordinate or time-conjugate momentum is appended. The word
"extended" in `GCExtendedFormulation` refers here to the two physical copies,
not to a completely extended phase space.

The principal shapes are:

| Quantity | Symbol | NumPy shape |
|---|---:|---:|
| Physical state | `z` | `(m,) = (2*p,)` |
| Projection multiplier | `mu` | `(m,) = (2*p,)` |
| Doubled state | `Y = [u; v]` | `(2*m,) = (4*p,)` |
| Per-particle doubled blocks | `q_i` | `(p, 4)` |
| Batch of particle Jacobians | `J_i` | `(p, 4, 4)` |
| Packed doubled-state Jacobian | `J` | `(2*m, 2*m) = (4*p, 4*p)` |
| Constraint matrix | `G` | `(m, 2*m)` |
| Normal embedding | `N = G.T` | `(2*m, m)` |
| Returned history for `S` saved times | `states` | `(m, S)` |

## The complete step in one formula

Define the diagonal embedding, averaging projection, diagonal constraint, and
normal embedding by

\[
E=\begin{pmatrix}I_m\\I_m\end{pmatrix},\qquad
P=\frac12\begin{pmatrix}I_m&I_m\end{pmatrix},\qquad
G=\begin{pmatrix}I_m&-I_m\end{pmatrix},\qquad
N=G^T.
\]

Let `Psi_(h,t)` be the complete twelve-stage BM4 map implemented by
`_advance_composition`. For an accepted physical state `z_n`, the method finds
`mu` from

\[
r(\mu)
=G\,\Psi_{h,t_n}(Ez_n+N\mu)+2\mu=0.
\]

It then returns

\[
z_{n+1}
=P\left(\Psi_{h,t_n}(Ez_n+N\mu)+N\mu\right).
\]

The same multiplier therefore displaces the input and corrects the output.
This is Hairer's symmetric projection around the entire BM4 cycle. There is no
projection between its internal stages.

The word **implicit** names this reduced multiplier root solve. It does not mean
that the twelve base stages are implicit: every direct or adjoint stage is an
explicit sequential update supplied by the prepared GC formulation.

## `_core.py`, in source order

### Module header and imports

The copyright and SPDX lines preserve the origin and license of the BM4
composition. The module docstring states its single responsibility: apply a
fourth-order palindromic composition made from a prepared direct map and its
adjoint.

`from __future__ import annotations` postpones evaluation of annotations. The
remaining imports have narrow roles:

| Import | Use in `_core.py` |
|---|---|
| `Callable` | Type the observer-facing fixed-stage closure |
| `Literal` | Restrict an observation's flow name to `"flow"` or `"adjoint_flow"` |
| `numpy` | Store coefficients and normalize map results |
| `PreparedDirectAdjointFormulation` | Describe the prepared direct/adjoint map contract |
| `IntegrationStage` | Immutable record produced for one observed stage |
| `StageObserver` | Optional callback type receiving those records |

### `_BM4_HALF_STAGES`, `_BM4_STAGES`, and `_BM4_ORDERS`

`_BM4_HALF_STAGES` stores six signed coefficients:

\[
\begin{aligned}
a_1&= 0.0792036964311957,&
a_2&= 0.1303114101821663,\\
a_3&= 0.2228614958676077,&
a_4&=-0.3667132690474257,\\
a_5&= 0.3246481886897062,&
a_6&= 0.1096884778767498.
\end{aligned}
\]

They are stored as floating-point values and satisfy

\[
\sum_{j=1}^{6}a_j=\frac12
\]

up to floating-point rounding. The negative fourth coefficient is deliberate:
that subflow advances backwards before later subflows recover the net forward
step.

`np.flip(_BM4_HALF_STAGES)` reverses the half sequence, and `np.concatenate`
places it after the original half. Thus `_BM4_STAGES` is the twelve-entry
palindrome

\[
(b_1,\ldots,b_{12})
=(a_1,a_2,a_3,a_4,a_5,a_6,a_6,a_5,a_4,a_3,a_2,a_1),
\]

whose sum is one. A stage duration is `b_j * h`, so all twelve signed durations
sum to the requested complete-step duration `h`.

`np.tile([1, 0], 6)` creates `_BM4_ORDERS`:

```text
1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
```

An order value of `1` selects the adjoint map; `0` selects the direct map. In
one-based stage notation the exact schedule is:

| Stage | Coefficient | Map | Evaluation time before the stage clock update |
|---:|---:|---|---|
| 1 | `a1` | adjoint | current clock |
| 2 | `a2` | direct | current clock `+ a2*h` |
| 3 | `a3` | adjoint | current clock |
| 4 | `a4` | direct | current clock `+ a4*h` |
| 5 | `a5` | adjoint | current clock |
| 6 | `a6` | direct | current clock `+ a6*h` |
| 7 | `a6` | adjoint | current clock |
| 8 | `a5` | direct | current clock `+ a5*h` |
| 9 | `a4` | adjoint | current clock |
| 10 | `a3` | direct | current clock `+ a3*h` |
| 11 | `a2` | adjoint | current clock |
| 12 | `a1` | direct | current clock `+ a1*h` |

The private leading underscores are intentional. These arrays are
implementation details shared with the analytic Jacobian traversal, not public
method configuration.

### `_checked_map`

`_checked_map(mapper, duration, t, state)` is the guarded entry point for one
prepared direct or adjoint stage.

1. `callable(mapper)` rejects a malformed prepared formulation before trying to
   invoke it.
2. The call order is exactly `mapper(duration, t, state)`, which is the prepared
   formulation protocol.
3. `np.asarray(...)` accepts array-like results without prescribing a new
   numeric dtype.
4. The result must have exactly the same shape as the input. A direct or
   adjoint stage is not allowed to change the doubled-state dimension.
5. The checked array is returned for the next stage.

Finiteness is checked after the complete base map by `_bm4_map` in
`implicit.py`. `_checked_map` is specifically the per-stage shape barrier.

### `_advance_composition`

`_advance_composition` applies exactly one complete BM4 cycle. Its positional
arguments are the prepared formulation, start time, doubled input state, and
complete-step duration. Its keyword-only arguments attach diagnostic identity
and observation behavior without changing the numerical map:

| Argument | Meaning |
|---|---|
| `step_index` | Accepted main-grid index recorded on stage observations |
| `stage_observer` | Callback for stage snapshots, or `None` for the normal fast path |
| `formulation_name` | Observer metadata |
| `method_name` | Observer metadata |

The `zip(..., strict=True)` traversal pairs every coefficient with exactly one
order flag. `strict=True` turns a future accidental length mismatch into an
error instead of silently dropping stages. `enumerate` supplies the zero-based
`stage_index` stored in diagnostics.

For each pair:

1. `duration = coefficient * step` forms the signed substep.
2. An order of `0` selects `prepared.direct_map`, labels it as `"flow"`, and
   evaluates it at `current_time + duration`.
3. An order of `1` selects `prepared.adjoint_map`, labels it as
   `"adjoint_flow"`, and evaluates it at `current_time`.
4. The local `apply_stage` closure freezes the selected map, duration, and
   evaluation time in default arguments. Freezing them matters because an
   observer can retain this callable after the loop has moved to later stages.
5. `state_before` keeps the current stage input and `apply_stage` produces the
   next doubled state through `_checked_map`.
6. If observation is enabled, a second small closure named `map_state` exposes
   that exact fixed stage to diagnostics. It also freezes `apply_stage`, avoiding
   Python's late-binding behavior.
7. `IntegrationStage` stores copied before/after arrays, the fixed-stage
   callable, the precise dynamics object, names, indices, time, and signed
   duration. Copies prevent later state rebinding or mutation from changing the
   recorded snapshot.
8. `t += duration` moves the composition clock by the signed duration.

The observer is called after the stage has been evaluated but before the local
clock moves to the next stage. After all twelve updates, the accumulated clock
has advanced by `step`, and the function returns the final doubled state.

There is no call to a projection routine in this loop. Projection belongs to
the enclosing nonlinear equation in `implicit.py`.

### Empty `__all__`

`__all__: list[str] = []` declares that `_core.py` exports no public API through
wildcard import. `implicit.py` deliberately imports its private constants and
composition function by name because it implements the one supported public
BM4 method.

## `implicit.py`, in source order

### Module purpose and imports

The module docstring states the defining placement rule: one symmetric Hairer
projection surrounds one complete BM4 composition step.

`from __future__ import annotations` postpones evaluation of type annotations.
The numerical class now contains its initialization and execution operations directly.

Every import supports one part of the implementation:

| Import | Purpose |
|---|---|
| `Callable` | Complete-map callable accepted by the numerical Jacobian helper |
| `dataclass` | Immutable step result and numerical class with per-run fields |
| `Literal`, `TypeAlias` | Closed string choices for Jacobian selection |
| `numpy as np` | Arrays, norms, finite checks, linear algebra, and defaults |
| `GuidingCenterDynamics` | Runtime requirement and exact derivatives for the analytic Jacobian |
| `IntegrationMethod`, `StepInfo`, `StepResult` | Shared lifecycle and typed one-step records |
| `DiagnosticValue` | Typed metadata and exported auxiliary diagnostics |
| `GCDoubledMaps` | Directly constructed doubled physical direct/adjoint maps |
| `gc_coupling_matrix` | Exact particle-local harmonic coupling and its tangent map |
| `PreparedDirectAdjointFormulation` | Type contract for the prepared maps and metadata |
| `ImplicitBM4IntegrationStep` | Complete accepted-step observer record |
| `IntegrationStage` | Type of each reconstructed base-stage record |
| `StepObserver` | Optional public callback type |
| `InitialValueProblem` | Physical problem passed to `integrate` |
| `SimulationRequest` | Validated span, maximum step, and saved times |
| `NonlinearSolver` | Closed type containing `"newton"` and `"broyden"` |
| `_solve_broyden` | Shared good-Broyden iteration |
| `_validate_nonlinear_solver` | Runtime solver-name validation |
| `_BM4_STAGES`, `_BM4_ORDERS` | Reuse the exact core schedule in the analytic tangent traversal |
| `_advance_composition` | Evaluate or observe the complete base cycle |

The Jacobian implementation imports the core coefficient arrays rather than
copying them. This is an important consistency invariant: the state map and its
analytic derivative cannot acquire different stage schedules independently.

### `NewtonJacobianMethod` and `NEWTON_JACOBIAN_METHODS`

`NewtonJacobianMethod` is the static type alias

```python
Literal["analytic", "finite_difference"]
```

and `NEWTON_JACOBIAN_METHODS` is the corresponding runtime tuple. The alias
helps type checking; the tuple lets `BM4Implicit.__post_init__` reject invalid
strings at runtime. The choice affects Newton only. Broyden builds and updates
its own reduced residual-Jacobian approximation, although the configured
Jacobian-method value remains validated and reported as metadata.

### `_ProjectedBM4Step`

This private frozen, slotted dataclass transports everything produced by one
converged projection solve:

| Field | Shape/type | Meaning |
|---|---|---|
| `state` | `(m,)` | Accepted physical `z_(n+1)` |
| `multiplier` | `(m,)` | Converged Hairer multiplier `mu` |
| `internal_input` | `(2*m,)` | Displaced doubled input `[z_n + mu; z_n - mu]` |
| `mapped` | `(2*m,)` | Raw complete-cycle result `Psi(internal_input)` before output correction |
| `iterations` | integer | Number of accepted nonlinear corrections; zero if `mu=0` already satisfies the equation |
| `residual_evaluations` | integer | Evaluations of the reduced residual along the nonlinear iterates |
| `residual_norm` | float | Infinity norm of the converged reduced residual |

`frozen=True` prevents rebinding fields and `slots=True` removes an arbitrary
instance dictionary. Array fields are explicitly copied at result boundaries
where later diagnostic ownership matters.

### Scalar validators

The three small validators centralize constructor normalization:

- `_positive_finite` explicitly rejects Python and NumPy booleans, converts the
  value to `float`, and requires a finite value strictly greater than zero. It
  is used for both tolerances and the relative finite-difference increment.
- `_nonnegative_finite` has the same Boolean and finiteness checks but permits
  zero. It is used for the coupling frequency; zero disables the harmonic
  mixing matrix but leaves the explicit cross-copy shears active.
- `_positive_integer` rejects booleans, non-integral objects, and values below
  one. It accepts both Python integers and NumPy integer scalars, then returns a
  plain `int`.

Booleans need explicit rejection because Python's `bool` is a subclass of
`int`, and silently accepting `True` as a tolerance or iteration count would
hide configuration errors.

### `_bm4_map`

`_bm4_map(prepared, t, internal_state, step)` wraps the core composition as the
base-map function used by both the residual and the numerical Jacobian.

1. `np.asarray(..., dtype=float)` establishes floating-point arithmetic for the
   doubled input.
2. `_advance_composition` executes all twelve stages with observation disabled.
   The fixed placeholder `step_index=0` is irrelevant because no stage record
   is emitted. The names are likewise descriptive metadata for a path that has
   no active observer.
3. The completed result must keep the input shape and contain only finite
   values. This catches a malformed prepared map and any overflow or invalid
   arithmetic that remains visible in the final base-cycle result. Intermediate
   stage finiteness is not checked separately.
4. The result is normalized to a floating-point array and returned.

The function accepts and returns doubled vectors of shape `(2*m,)`, not
physical vectors.

### `_central_difference_jacobian`

This generic helper differentiates a same-shape map `F: R^q -> R^q`. In this
module `q=2*m=4*p` and `F` is the complete doubled BM4 map.

After converting the base state to floating point, `np.empty((q, q))`
allocates the result because every column will be filled. For column `j`, the
perturbation is

\[
\delta_j
=\epsilon_{\mathrm{rel}}\max(1,|Y_j|),
\]

and the stored centered difference is

\[
J_{:,j}
=\frac{F(Y+\delta_j e_j)-F(Y-\delta_j e_j)}{2\delta_j}.
\]

The `max(1, abs(component))` scaling avoids an unusably tiny absolute
perturbation near zero while scaling proportionally for large components. Each
forward and backward map must preserve shape; after all columns, the full
matrix must be finite.

One numerical Jacobian costs `2*q = 4*m = 8*p` complete twelve-stage BM4 map
evaluations. These perturbation-map calls are Jacobian work; they are not
included in the recorded `residual_evaluations`, which counts nonlinear
iterate residuals.

### `_projection_matrices`

For a physical size `m`, the function builds

\[
G=\begin{pmatrix}I_m&-I_m\end{pmatrix}
\]

with `np.concatenate((identity, -identity), axis=1)`, and

\[
N=\begin{pmatrix}I_m\\-I_m\end{pmatrix}=G^T
\]

with concatenation along axis zero. In code the returned names are
`constraint` and `normal`. Their identities imply

\[
GE=0,\qquad GN=2I_m.
\]

`G` measures disagreement between the two copies, while `N` embeds a physical
multiplier as opposite displacements of those copies.

### `_bm4_evaluation`

This helper makes the input half of the symmetric projection explicit:

```text
internal_input = [state + multiplier; state - multiplier]
mapped = complete_BM4_map(internal_input)
```

It returns both arrays. Keeping `internal_input` is necessary for Newton's map
Jacobian and later lets the observer reconstruct the exact converged base
stages without solving the nonlinear equation again.

### `_bm4_map_jacobian`

This dispatcher differentiates the complete base map at `internal_input`.

- `method == "analytic"` calls `_analytic_bm4_map_jacobian`.
- The only other valid public choice is `"finite_difference"`, so the fallback
  applies `_central_difference_jacobian` to a closure with fixed `prepared`,
  `t`, and `step`.

Runtime validation happens when `BM4Implicit` is constructed, so this private
dispatcher does not repeat it.

### `_particle_blocks` and `_packed_physical`

The analytic derivative is most naturally written as one `4 x 4` tangent map
per independent particle, while the simulation stores component-major arrays.
These helpers bridge the two layouts.

`_particle_blocks` first requires an exact `(4*p,)` input. It then forms an
`(p, 4)` matrix with columns

```text
u_x, u_y, v_x, v_y
```

by slicing the four contiguous component blocks and using `np.column_stack`.
An incorrect length raises an error that makes the analytic Jacobian's
guiding-centre assumption explicit.

`_packed_physical(blocks, offset)` performs the reverse selection for one
physical copy. Offset `0` concatenates `u_x` and `u_y`; offset `2`
concatenates `v_x` and `v_y`. Each result has shape `(m,)` and is suitable for
the component-major dynamics API.

### `_batched_identity`

`np.broadcast_to(np.eye(4), (p, 4, 4))` creates a read-only broadcasted view of
the same identity for all particles. `.copy()` makes the batch writable because
the following stage-Jacobian helpers replace its off-diagonal `2 x 2` blocks.

### `_direct_stage_particle_jacobians`

For one particle, write the current doubled state as `(u,v)`, let `d` be the
signed stage duration, let `f_tau` be the guiding-centre vector field at the
fixed evaluation time, and let `C(d,omega)` be the exact harmonic coupling.
The prepared direct map performs

\[
v^*=v+d f_\tau(u),\qquad
u^*=u+d f_\tau(v^*),\qquad
Y^+=C(d,\omega)(u^*,v^*).
\]

The code mirrors this order:

1. `_packed_physical(state, 0)` and offset `2` obtain all `u` and `v` values.
2. `particle_vector_field_jacobians(time, first)` computes
   `A_u = Df_tau(u)` with batch shape `(p, 2, 2)`.
3. `updated_second` reproduces `v*`, because the second derivative must be
   evaluated at the state actually used by the next shear.
4. A second derivative call computes `A_v* = Df_tau(v*)`.
5. `second_shear` inserts `d*A_u` in the lower-left block.
6. `first_shear` inserts `d*A_v*` in the upper-right block.
7. `gc_coupling_matrix(d, frequency)` supplies the common exact `4 x 4`
   coupling matrix.

Thus each returned particle factor is

\[
F_{\mathrm{direct}}
=C(d,\omega)
\begin{pmatrix}I&dA_{v^*}\\0&I\end{pmatrix}
\begin{pmatrix}I&0\\dA_u&I\end{pmatrix}.
\]

NumPy broadcasts the `4 x 4` coupling over the `(p, 4, 4)` shear batches.

### `_adjoint_stage_particle_jacobians`

The adjoint reverses the direct operation order. It first applies the coupling
and then the two shears:

\[
(u_c,v_c)=C(d,\omega)(u,v),\qquad
u^*=u_c+d f_\tau(v_c),\qquad
v^*=v_c+d f_\tau(u^*).
\]

`state @ coupling.T` is the row-vector batch form of applying `C` to every
particle. The helper then computes `Df_tau(v_c)`, reconstructs `u*`, and
computes `Df_tau(u*)`. The two writable identities receive the corresponding
upper-right and lower-left blocks. The returned factor is

\[
F_{\mathrm{adjoint}}
=\begin{pmatrix}I&0\\dA_{u^*}&I\end{pmatrix}
 \begin{pmatrix}I&dA_{v_c}\\0&I\end{pmatrix}
 C(d,\omega).
\]

This ordering is essential: moving the coupling to the other side would
differentiate a different map.

### `_packed_particle_jacobians`

The stage traversal holds independent particle tangent matrices in shape
`(p, 4, 4)`, but Newton needs a derivative in the global component-major
coordinates.

The helper allocates a zero `(4*p, 4*p)` matrix. For each particle `i`, the
indices

```text
i, p+i, 2*p+i, 3*p+i
```

locate `(u_xi, u_yi, v_xi, v_yi)` in the packed vector. `np.ix_` selects their
four-by-four Cartesian block and inserts that particle's Jacobian. All
cross-particle entries remain zero because the guiding-centre particles are
independent in this formulation.

### `_analytic_bm4_map_jacobian`

This function differentiates the same twelve-stage map as `_bm4_map`, but
accumulates exact particle tangent factors while advancing the state.

Its guards, together with the exact derivative call, document the analytic
path's domain:

- `prepared.dynamics` must be `GuidingCenterDynamics`, which supplies
  `particle_vector_field_jacobians`;
- when Newton needs a correction, its effective potential must have
  `interpolation_order >= 3`, because the exact Jacobian then requests second
  spatial derivatives inside
  `GuidingCenterDynamics.particle_vector_field_jacobians`;
- `prepared.particle_count` must be a positive integer; and
- `prepared.coupling_frequency` must be present.

The validated values are converted to ordinary `int` and `float`. `current` is
a private floating-point copy of the doubled input. `particle_jacobian` starts
as an identity batch, and `current_time` starts at the complete step's input
time.

The strict coefficient/order traversal duplicates the core schedule exactly.
For every stage it:

1. forms the signed duration;
2. converts the current packed state to per-particle blocks;
3. uses `current_time + duration` for a direct stage or `current_time` for an
   adjoint stage, matching `_advance_composition`;
4. computes the corresponding exact particle factor;
5. advances `current` with the actual prepared direct or adjoint map; and
6. left-multiplies the accumulated derivative,

\[
J_{j}=F_jJ_{j-1},\qquad J_0=I.
\]

The local clock then advances by the same signed duration. Left multiplication
is the chain rule in execution order, so after twelve stages the batch contains
`F_12 ... F_2 F_1`.

The final batch must be finite. `_packed_particle_jacobians` converts it to the
global `(2*m, 2*m)` component-major derivative returned to Newton.

### `_solve_reduced_projected_bm4_step`: common setup

This is the numerical heart of the public method. It accepts a physical state,
one fixed time and step, solver tolerances, a Jacobian choice, and a nonlinear
solver choice. It returns `_ProjectedBM4Step` only after convergence.

The common setup performs these operations:

1. Convert `state` to a floating-point array named `value`.
2. Require a finite, non-empty one-dimensional physical vector. A matrix,
   scalar, empty vector, `NaN`, or infinity is rejected.
3. Set `physical_size = m` and construct `G` and `N`.
4. Start from `mu_0 = 0`, meaning the first residual evaluates BM4 on two equal
   copies of the accepted physical state.
5. Compute the state-scaled stopping threshold

   \[
   \tau=\text{atol}+\text{rtol}\max(1,\lVert z_n\rVert_\infty).
   \]

   The scale is fixed from the step's input state; it is not changed as the
   multiplier iteration moves.

### The Broyden branch

For `nonlinear_solver == "broyden"`, the local `residual_function` maps a trial
multiplier to

\[
r(\mu)=G\,M(\mu)+2\mu,
\qquad
M(\mu)=\Psi_{h,t}([z+\mu;z-\mu]).
\]

It returns `(mapped, candidate)` as a payload alongside the residual. The
shared Broyden solver carries this payload with every residual evaluation, so
the converged raw map and multiplier are already synchronized; no extra BM4
evaluation is needed after convergence.

Writing `mapped = [M_1; M_2]`, the difference between the corrected copies is

\[
(M_1+\mu)-(M_2-\mu)=M_1-M_2+2\mu=r(\mu).
\]

The residual infinity norm is therefore exactly the diagonal defect of the two
corrected copies, not merely an indirect solver indicator.

The initial reduced Jacobian approximation is `4*I`. Its reason follows from
the small-step identity limit. If the complete base map is locally
approximately the identity, then

\[
r(\mu)
\approx G(Ez+N\mu)+2\mu
=GN\mu+2\mu
=4\mu,
\]

because `GE=0` and `GN=2I`. Therefore `Dr` tends to `4I` as the complete step
tends to zero. This is a cheap structure-aware initial approximation, not an
arbitrary scale factor.

The shared implementation uses good Broyden updates. Given a correction `s`
and residual change `y`, its Jacobian approximation changes by

\[
B_{k+1}=B_k+
\frac{(y-B_ks)s^T}{s^Ts}.
\]

It reports zero iterations and one residual evaluation if `mu=0` already
passes the threshold. A singular approximate Jacobian, non-finite correction,
correction too small before convergence, or exhausted iteration limit raises a
`RuntimeError` with the BM4 time and step context.

After convergence, the implementation applies the output half of Hairer's
projection:

```text
corrected = mapped + N @ multiplier
projected_state = mean(corrected first copy, corrected second copy)
```

Algebraically, the opposite `+mu` and `-mu` corrections cancel in the mean, but
forming `corrected` keeps the implemented operation identical to the
projection definition. The returned record owns the physical state,
multiplier, reconstructed displaced input, raw mapped state, Broyden work
counters, and final residual infinity norm.

`jacobian_method` and `jacobian_relative_step` are intentionally unused in this
branch. They configure only Newton, even though the method configuration
keeps and reports them for a uniform configuration schema.

In public configuration terms, Broyden therefore ignores both
`newton_jacobian_method` and `newton_jacobian_relative_step`. It evaluates only
the reduced residual and updates its `4I` approximation from secant data.

### The Newton branch

The explicit `nonlinear_solver != "newton"` guard is defense in depth for
private callers. Public construction has already restricted the field to
Newton or Broyden.

The loop uses `range(max_iterations + 1)` because the initial residual is
checked before any correction and at most `max_iterations` corrections may be
accepted. At iterate `mu_k` it:

1. obtains both the displaced input and its complete BM4 image;
2. forms `residual = G @ mapped + 2*mu_k`;
3. measures its infinity norm; and
4. accepts immediately when that norm does not exceed the fixed threshold.

On acceptance it performs the same normal output correction and two-copy
average as Broyden. `iterations=k` counts corrections, while
`residual_evaluations=k+1` counts the initial residual plus one after each
correction. Numerical-Jacobian perturbation maps are not counted as residual
evaluations.

If the residual is still too large and another correction is allowed, the
complete-map derivative

\[
J_{\mathrm{BM4}}
=D\Psi_{h,t}(Ez_n+N\mu_k)
\]

comes from the configured analytic or centered-difference path. The chain rule
for the reduced residual gives

\[
D_\mu r
=GJ_{\mathrm{BM4}}N+2I_m.
\]

`np.linalg.solve` computes `correction` from

\[
(D_\mu r)\,\text{correction}=r,
\]

and the update is `mu <- mu - correction`, the usual Newton step. The code
solves the linear system directly instead of explicitly forming an inverse.

A singular reduced Jacobian becomes a contextual `RuntimeError`. If the final
permitted iterate still exceeds the threshold, the method raises rather than
returning an unconverged physical state. The error includes time, step,
residual norm, threshold, and correction limit.

Every one-step call starts again from `mu=0`. There is no multiplier warm start,
Newton damping, line search, adaptive retry, or automatic step reduction. A
failure therefore propagates to the fixed-grid integration instead of silently
changing the requested numerical procedure.

### `BM4Implicit.initialize`: resources on the numerical class

`new_run` creates another `BM4Implicit` from constructor options and invokes its
`initialize`. It creates `state_formulation = DoubledFormulation(...)` with
the selected tracking flag and obtains its initial diagonal state. It binds
`self.formulation = GCDoubledMaps(...)` without energy for the spatial solve.
When tracking is enabled, residual evaluations retain their physical shear
inputs, signed durations and evaluation times. Only the converged residual's
points survive the solve; the same map instance serves both tracking settings.
Metadata and diagnostic aliases belong to this fresh run.

The ordinary class method `_solve(t, state, h)` calls the reduced Hairer kernel
with this instance's formulation and solver controls. Global scheduling, history
collection and event dispatch remain in `simulation/integration.py`.

### `advance(t, state, h)` and `StepResult`

The numerical advance extracts physical coordinates and calls `self._solve`
exactly once. With energy tracking, it evaluates `-partial_t H` at the 24
retained shear points in their original order and divides the summed momentum
increment by two. No spatial stage or coupling is repeated. `state_formulation.finish`
embeds the accepted physical state twice and updates each particle time and momentum.
The result is `StepResult(after, statistics, result)`, where `after` has the
full internal layout and `result` retains spatial solve details and optional
energy quadrature points.
It has no `step_index` or `observe` parameter. The same operation is used for main
and shadow steps; their recording policy belongs to the common coordinator.
The five metric values are iteration count, residual evaluations, residual norm,
effective tolerance and projection-multiplier norm. They are not stored in the
method. `IntegrationCollector` copies rows only for accepted main steps.

### `build_observation(info, step)`

Only a requested accepted-step observation reaches this adapter. It replays the
complete twelve-stage cycle from `step.details.internal_input`, checks exact
equality with `step.details.mapped`, and builds `ImplicitBM4IntegrationStep`.
The event contains independent physical before/after states, copied multiplier,
solve statistics and twelve doubled-state stage snapshots.

The retained `map_state(candidate)` calls `self._solve` with the captured
time and duration. It never enters the coordinator or recursively emits events.
The adapter returns the event; `integrate_method` invokes the observer. The
replay is diagnostic work and never replaces the accepted physical result.

### `export_history` and the common controller

The inherited exporter delegates to `DoubledFormulation`, extracting one spatial
copy and, when enabled, the common energy histories. The integration workspace
has 4N or 6N entries. `track_energy=True` evaluates passive quadrature using
the converged residual's retained shear states; this does not modify the
multiplier or add physical field evaluations. The normalized increment is half
the accumulated sum. The common coordinator chooses `FixedStepController` by default.
Its uniform grid uses the same `_step_count` and arithmetic as before. It delivers
accepted `StepInfo`/`StepResult` pairs. Interior requested samples use a shortened
map from an independent copy of the preceding main state, without collection or
observation. The main result is unchanged by output density.

`IntegrationCollector` stores internal samples and small metric arrays; physical
extraction happens once at export.
`step_times`, `step_start_times` and `step_sizes` identify each accepted interval;
these arrays are independent of the requested output schedule. The added
`output_interpolation_count` counts interior samples, not nonlinear evaluations.

### Diagnostics dictionary

At finalization, the common collector converts its rows into NumPy arrays;
`integrate_method` combines them with prepared metadata and output extraction. Per-step arrays have length `step_count` and
refer only to accepted main steps.

| Diagnostic key | Value |
|---|---|
| `step_count` | Number of accepted uniform main steps |
| `nonlinear_solver` | `"newton"` or `"broyden"` |
| `nonlinear_iterations` | Per-step correction-count array |
| `residual_evaluations` | Per-step residual-evaluation array |
| `nonlinear_residual_norms` | Per-step converged infinity norms |
| `nonlinear_tolerances` | Per-step state-scaled thresholds |
| `nonlinear_absolute_tolerance` | Configured absolute tolerance |
| `nonlinear_relative_tolerance` | Configured relative tolerance |
| `nonlinear_max_iterations` | Configured correction limit |
| `newton_iterations` | Compatibility alias of `nonlinear_iterations` |
| `newton_residual_norms` | Compatibility alias of `nonlinear_residual_norms` |
| `projection_multiplier_norms` | Per-step multiplier infinity-norm array |
| `newton_absolute_tolerance` | Compatibility/configuration alias of the nonlinear absolute tolerance |
| `newton_relative_tolerance` | Compatibility/configuration alias of the nonlinear relative tolerance |
| `newton_max_iterations` | Compatibility/configuration alias of the nonlinear limit |
| `newton_jacobian_relative_step` | Configured centered-difference scale |
| `newton_jacobian_method` | Configured Newton derivative strategy |
| `coupling_frequency` | Harmonic coupling rate |
| `projection_solver_formulation` | Fixed identity marker `bm4_implicit_reduced` |

The keys retaining `newton_` in their names are compatibility aliases. When
Broyden is selected, the count and residual arrays still describe Broyden
work, while `nonlinear_solver` disambiguates them. Likewise, the reported
Jacobian configuration is inactive metadata for Broyden. Here "alias" means a
separately constructed diagnostic array with the same values, not shared NumPy
object identity.

The same naming convention applies to an `ImplicitBM4IntegrationStep` event:
its inherited `newton_iterations`, `newton_residual_norm`, and
`newton_tolerance` fields contain the selected nonlinear solver's values. Read
the event's `nonlinear_solver` field before interpreting those historical
names.

Work accounting is deliberately narrower than total runtime:

- `residual_evaluations` counts complete-map calls made at nonlinear iterates;
- one finite-difference Newton Jacobian adds `8*p` complete twelve-stage cycles;
- one analytic Newton Jacobian adds another twelve-stage state traversal plus
  its exact tangent-factor evaluations;
- an enabled observer adds one twelve-stage replay per accepted main step; and
- an interior saved time adds a complete shadow nonlinear solve whose work is
  not included in accepted-step arrays.

Only the first category is represented by `residual_evaluations`.

`IntegrationData` receives `request.output_times`, the physical history, and
this diagnostics dictionary. The common simulation runner later builds an
immutable public `Solution`; BM4 does not put its doubled workspace into that
solution.

### `BM4Implicit` public dataclass

`BM4Implicit` is a numerical dataclass. Constructor fields describe its controls;
`formulation` is excluded from construction and belongs to an initialized run.
`simulate` creates a fresh instance for each execution, so the supplied
configuration is reusable. The caller still manages any stateful observer.

| Field | Default | Validation and effect |
|---|---:|---|
| `track_energy` | `False` | Store p times and p normalized passive momenta; no change to the spatial solve |
| `coupling_frequency` | `0.0` | Finite and non-negative; zero disables doubled-copy harmonic coupling while retaining the reduced Hairer projection |
| `newton_absolute_tolerance` | `1e-13` | Finite and strictly positive; absolute part of every nonlinear threshold |
| `newton_relative_tolerance` | `1e-12` | Finite and strictly positive; multiplies the input-state infinity scale |
| `newton_max_iterations` | `12` | Positive integer; maximum Newton or Broyden corrections |
| `newton_jacobian_relative_step` | `cbrt(machine epsilon)` | Finite and positive; component scaling for centered differences |
| `newton_jacobian_method` | `"analytic"` | Exactly `"analytic"` or `"finite_difference"`; used by Newton |
| `nonlinear_solver` | `"newton"` | Exactly `"newton"` or `"broyden"` |
| `progress` | `False` | Enables the shared main-step progress display when truthy |
| `step_observer` | `None` | Optional callback receiving one record per accepted main step |

The cube root of machine epsilon is a conventional balance for centered
finite differences in finite-precision calculations. The field is validated
even when the analytic Jacobian or Broyden makes it inactive, keeping every
instance internally well formed.

`__post_init__` normalizes the constructor options.
It replaces numeric inputs with their normalized Python `float` or `int`
values, checks the Jacobian string against `NEWTON_JACOBIAN_METHODS`, and uses
the shared validator for the nonlinear solver. `progress` and `step_observer`
rely on their public type contracts rather than additional custom runtime
validation.

`integrate(problem, request)` is inherited from `IntegrationMethod`. It calls
`new_run` to create and initialize a fresh `BM4Implicit`, then calls
`integrate_method(run)`. The driver invokes ordinary methods on that instance. All solver and Jacobian choices solve the
same reduced projected BM4 map.

### Public `__all__`

`__all__ = ["BM4Implicit"]` makes this module's single-class architecture
explicit. Private records, helpers, coefficient arrays, aliases, and
runtime-choice tuples remain implementation details. The package-level `bm4`
initializer re-exports this class; the wider `simulation` initializer also
exports the public `ImplicitBM4IntegrationStep` observation record.

## End-to-end pseudocode

The following pseudocode condenses the implementation without hiding any
numerical boundary:

```text
construct BM4Implicit and validate configuration
construct DoubledFormulation with the selected tracking flag
bind physical GC direct/adjoint maps for the spatial solve
if tracking is enabled, retain shear inputs during residual evaluations

choose uniform accepted main grid from t_span and max_step
for each main interval (t, h):
    z = state_formulation.physical(current_internal_state)     # shape (m,)
    mu = zeros_like(z)                                          # shape (m,)
    threshold = atol + rtol * max(1, norm_inf(z))

    solve r(mu) = G * BM4_12_stages([z+mu; z-mu]) + 2*mu = 0
        with Newton:
            use G * J_BM4 * N + 2*I
            obtain J_BM4 analytically or by centered differences
        or with good Broyden:
            start reduced Jacobian approximation at 4*I
            update it from successive residual evaluations

    mapped, energy_points = values retained by the converged residual
    corrected = mapped + [mu; -mu]                              # shape (2m,)
    z_next = mean(corrected copy 1, corrected copy 2)           # shape (m,)

    record accepted-step nonlinear diagnostics
    if a step observer exists:
        replay the same 12 base stages from the converged internal input
        require replayed output == mapped
        emit one physical step record containing the 12 internal records

    if tracking is enabled:
        increment = sum(duration * (-partial_t H)(time, point)
                        for time, duration, point in energy_points) / 2
    current_internal_state = state_formulation.finish(
        current_internal_state, z_next, t+h, increment)
    obtain any interior saved times through non-observed shadow solves

return saved physical history with shape (m, saved_times) and diagnostics
```

## Invariants and failure modes

| Invariant | Enforcement | Failure |
|---|---|---|
| Public and accepted states are finite non-empty vectors | One-step solver and fixed-grid validation | `ValueError` |
| Every direct/adjoint stage preserves doubled-state shape | `_checked_map` | `ValueError` |
| A complete BM4 map preserves shape and remains finite | `_bm4_map` | `ValueError` |
| Numerical Jacobian map evaluations preserve shape and remain finite | `_central_difference_jacobian` and `_bm4_map` | `ValueError` |
| Analytic Jacobian is used only with compatible GC dynamics and prepared metadata | `_analytic_bm4_map_jacobian` guards | `TypeError` |
| Analytic stage product remains finite | `_analytic_bm4_map_jacobian` | `ValueError` |
| Nonlinear controls are finite and in range | validators and `BM4Implicit.__post_init__` | `ValueError` |
| Only Newton or Broyden is selected | shared and private solver guards | `ValueError` |
| Reduced Newton matrix is solvable | `np.linalg.solve` exception handling | `RuntimeError` |
| Broyden approximation and corrections remain usable | shared Broyden checks | `RuntimeError` |
| Residual meets its threshold before acceptance | both nonlinear branches | `RuntimeError` on exhaustion |
| Observer replay is exactly the solved base map | `np.array_equal(observed_mapped, result.mapped)` | `RuntimeError` |
| Main and shadow steps preserve the selected internal shape | `integrate_method` | `ValueError` |
| Every requested output time is covered | final fixed-grid check | `RuntimeError` |

These checks deliberately fail at the boundary where an assumption is broken.
In particular, there is no fallback that accepts an unconverged multiplier,
silently changes dimensions, or substitutes an arithmetic projection.

## Common interpretation pitfalls

- **Accepted internal states contain two identical spatial copies.** Stage
  workspaces may separate them. The exported trajectory contains one copy, and
  optional clock/momentum histories are returned as diagnostics.
- **There is one projection per complete cycle, not one per internal stage.**
  `_advance_composition` contains no projection call.
- **The projection is symmetric, not only a final average.** The multiplier
  changes the base-map input and applies the matching normal output correction.
- **Newton and Broyden are solver strategies, not distinct BM4 methods.** They
  target the same residual and produce the same state contract.
- **"Implicit" refers to the projection root, not the base stages.** All twelve
  direct/adjoint stages are explicit sequential prepared-map evaluations.
- **The analytic and finite-difference choices differentiate the complete base
  map.** They do not change the twelve-stage map itself.
- **Negative BM4 coefficients are expected.** They create signed backward
  subflows inside a net forward step.
- **Observer stage replay is diagnostic.** It adds work only when requested and
  never replaces the converged accepted result.
- **Saved output density is independent of accepted integration density.**
  Interior output samples use shadow steps that do not perturb the main grid.
- **The solver is neither warm-started nor adaptive.** Each multiplier begins at
  zero, and a failed root solve is not retried with a shorter step.
- **`newton_*` diagnostic aliases may contain Broyden statistics.** Always read
  `nonlinear_solver` when interpreting them; prefer the `nonlinear_*` keys in
  new analysis code.

## Related documents

- [BM4Implicit simulation architecture](bm4-simulation-architecture.md)
- [Canonical mathematical theory](../tex/theory.tex)
- [Reduced Hairer derivation](../tex/implicit-reduced.tex)
- [Symbolic analytic-Jacobian audit](../bm4_jacobian_sympy.py)
