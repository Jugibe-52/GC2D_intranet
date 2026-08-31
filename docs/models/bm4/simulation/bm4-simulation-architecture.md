# BM4 fourth-order composition and projection architecture

## Public family

The BM4 implementation exposes six public classes. They share the same
twelve-stage coefficient schedule, but they do not place projection at the
same point and are not interchangeable aliases.

| Public class | Compatible problem/formulation | Projection policy | Nonlinear solve |
|---|---|---|---|
| `BM4Composition` | Any `DirectAdjointFormulation`; built-in GC and FC bindings | Formulation-owned physical output extraction; no stage or cycle projection | None |
| `ProjectedBM4Composition` | A `StageProjectedFormulation`; currently `GCStageProjectedFormulation` | Arithmetic re-embedding after every BM4 stage | None |
| `MidpointBM4` | GC | One arithmetic re-embedding after the complete uncoupled BM4 cycle | None |
| `BM4Implicit1` | GC, one or more particles | One Hairer symmetric projection around the complete coupled BM4 cycle | Reduced multiplier in `R^(2N)` |
| `BM4Implicit2` | GC, one or more particles | Same exact projected map as `BM4Implicit1` | Simultaneous doubled output and multiplier in `R^(6N)` |
| `BM4_implicit2` | One GC particle | One full-diagonal projection of duplicated `Z=(x,y,t,k)` | Full-state multiplier in `R^4` |

The underscore in `BM4_implicit2` is significant. That class is the fully
extended-state method; it is not an alias for the simultaneous physical-state
method `BM4Implicit2`.

## Fourth-order composition

The six independent half-cycle coefficients are

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

They satisfy `sum(a_1, ..., a_6) = 1/2`. One step uses

\[
(b_1,\ldots,b_{12})=
(a_1,a_2,a_3,a_4,a_5,a_6,a_6,a_5,a_4,a_3,a_2,a_1)
\]

and alternates the adjoint and direct maps in this chronological order:

```text
adjoint(b1 h), direct(b2 h), ..., adjoint(b11 h), direct(b12 h)
```

The negative coefficient `a_4` is an intentional backward subflow. After each
stage, the composition clock advances by `b_j h`. An adjoint stage evaluates
its map at the current clock; a direct stage evaluates at the clock plus its
signed duration. The coefficient sum makes the final clock exactly one full
step later, up to floating-point roundoff.

For a first-order map and its exact adjoint, this palindromic composition is
symmetric and has designed global order four. `MidpointBM4` has a tested
fourth-order physical trajectory. Other projection placements define different
maps and are evaluated independently rather than being treated as aliases.

## Explicit projection placements

Let the physical state have size `m`, and define

\[
E=\begin{pmatrix}I\\I\end{pmatrix},\qquad
P=\frac12\begin{pmatrix}I&I\end{pmatrix}.
\]

`BM4Composition(GCExtendedFormulation(...))` begins with `Ez_0`, retains both
copies across complete steps, and applies `P` only when producing the saved
physical history.

`ProjectedBM4Composition(GCStageProjectedFormulation())` applies `EP` after
every direct or adjoint stage. The projection is part of each observed stage
map.

`MidpointBM4` instead computes one complete uncoupled cycle

\[
(u^+,v^+)=\Psi_h(Ez_n),\qquad
z_{n+1}=P(u^+,v^+),
\]

and starts the next complete cycle from `Ez_(n+1)`. Its diagnostics record one
`copy_separation_norm` per main step, `projection_scope =
"complete_bm4_cycle"`, one projection, and 24 GC vector-field evaluations per
complete step.

## Hairer symmetric projection on physical GC state

`BM4Implicit1` and `BM4Implicit2` use the coupled
`GCExtendedFormulation`. With

\[
G=\begin{pmatrix}I&-I\end{pmatrix},\qquad N=G^T,
\]

they displace the diagonal input by a multiplier,

\[
\widehat Y_n=Ez_n+N\mu,
\qquad M=\Psi_{h,t_n}(\widehat Y_n),
\]

and apply the matching output correction. The reduced formulation solves

\[
r(\mu)=GM+2\mu=0,
\qquad
z_{n+1}=P(M+N\mu).
\]

If `J_BM4 = D Psi_(h,t_n)(widehat Y_n)`, its Newton matrix is

\[
D_\mu r=GJ_{\mathrm{BM4}}N+2I.
\]

`BM4Implicit2` retains the doubled corrected output `Y` as a nonlinear unknown:

\[
F(Y,\mu)=
\begin{pmatrix}
Y-N\mu-\Psi_{h,t_n}(Ez_n+N\mu)\\
GY
\end{pmatrix}=0.
\]

Its Newton matrix is

\[
\begin{pmatrix}
I&-(I+J_{\mathrm{BM4}})N\\
G&0
\end{pmatrix}.
\]

Eliminating the doubled-output correction gives the reduced Newton equation,
so both classes compute the same exact physical root when they converge to the
same local solution. The two formulations differ in nonlinear workspace and
work counts, not in their defining projected map.

Both accept `nonlinear_solver="newton"` or `"broyden"`. Newton uses either the
exact product of all twelve GC stage Jacobians or a centered-difference
Jacobian of the complete BM4 map. Good Broyden updates the selected residual
from a deterministic identity-map approximation and does not differentiate
each iteration.

## Fully extended projection

`BM4_implicit2` promotes one particle to

\[
Z=(x,y,t,k),\qquad K(Z)=H(t,x,y)+k,
\]

duplicates the complete state to `(Z_1,Z_2) in R^8`, applies the same BM4
coefficient schedule, and solves a four-component full-diagonal projection.
The physical solution contains `(x,y)`; diagnostics retain `extended_time`,
the direct `extended_momentum`, the generalized energy, and its drift.

This variant accepts Newton or Broyden, requires `GuidingCenterDynamics`, and
currently requires exactly one particle. Its observer-facing complete map is
in `R^4`; each `FullyExtendedBaseMap` snapshot exposes the duplicated `R^8`
base cycle and its Jacobian.

## Fixed-grid lifecycle

Every variant delegates step scheduling to `integrate_fixed_grid`:

1. Prepare or construct the method-specific internal initial state.
2. Advance an output-independent main grid with steps no larger than
   `SimulationRequest.max_step`.
3. Evaluate off-grid saved times by shadow advances from the preceding main
   node.
4. Project the internal history to physical states and return immutable
   `IntegrationData`, later wrapped as `Solution`.

Shadow advances affect requested output values but emit no observer events and
do not enter per-main-step diagnostics. Consequently, changing
`sample_count` cannot change the main-grid trajectory.

## Observations and diagnostics

- `BM4Composition`, `ProjectedBM4Composition`, and `MidpointBM4` can emit one
  `IntegrationStage` for each of the twelve stages of every main step.
- A stage record fixes the signed duration, evaluation time, flow kind,
  snapshots, physical dynamics instance, and exact stage `map_state`.
- `BM4Implicit1` and `BM4Implicit2` emit one
  `ImplicitBM4IntegrationStep` per main step. It contains nonlinear work, the
  converged multiplier, and the twelve reconstructed base-stage snapshots.
- `BM4_implicit2` emits `FullyExtendedImplicitIntegrationStep` records with the
  accepted extended map and full base-map data.

The implicit physical-state solutions publish the selected solver and Jacobian
strategy, iterations, residual evaluations, final residuals and tolerances,
projection-multiplier norms, coupling frequency, and projection formulation.
`MidpointBM4` publishes copy-separation and projection-placement metadata.
`BM4Composition(track_energy=True)` publishes extended momentum and maximum
generalized-energy drift.

## Public usage

```python
from simulation import BM4Implicit1, SimulationRequest, simulate

solution = simulate(
    problem,
    BM4Implicit1(
        coupling_frequency=0.2,
        nonlinear_solver="newton",
        newton_jacobian_method="analytic",
        newton_absolute_tolerance=1e-14,
        newton_relative_tolerance=1e-13,
        newton_max_iterations=40,
    ),
    SimulationRequest.uniform(
        t_span=(0.0, 2.0),
        max_step=0.05,
        sample_count=41,
    ),
)
```

For generic GC or FC composition, inject the formulation explicitly:

```python
from simulation import BM4Composition, FCSplitFormulation

method = BM4Composition(FCSplitFormulation(), track_energy=True)
```

## Studies and mathematical derivations

Reusable BM4 studies cover:

- midpoint, stage-projected, and implicit projected symplecticity;
- implicit Newton/Broyden iteration histories;
- generalized-energy and fully extended diagnostics;
- trajectory accuracy and multi-method comparisons; and
- equal-grid comparisons with Gauss--Legendre 4 and HBVM(4,2).

The principal modules are `src/studies/bm4_midpoint_symplecticity.py`,
`src/studies/projected_bm4_symplecticity.py`,
`src/studies/bm4_implicit_symplecticity.py`, and
`src/studies/bm4_implicit_iterations.py`. Presentation remains in
`src/visualization` and opt-in numerical analysis remains in `src/diagnostics`.

The complete projection derivations are available as
[`BM4_implicit_1.pdf`](../BM4_implicit_1.pdf) and
[`BM4_implicit_2.pdf`](../BM4_implicit_2.pdf), with their companion LaTeX
sources [`BM4_implicit_1.tex`](../BM4_implicit_1.tex) and
[`BM4_implicit_2.tex`](../BM4_implicit_2.tex) at the BM4 model root.
[`bm4_jacobian_sympy.py`](../bm4_jacobian_sympy.py) verifies the ordered
symbolic stage-product factors.

The companion component diagram is
[`bm4-simulation-architecture.puml`](bm4-simulation-architecture.puml).
