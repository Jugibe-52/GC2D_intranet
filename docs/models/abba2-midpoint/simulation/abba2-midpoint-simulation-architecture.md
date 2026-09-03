# ABBA2 midpoint physical simulation architecture

This document explains the companion
[`abba2-midpoint-simulation-architecture.puml`](abba2-midpoint-simulation-architecture.puml)
diagram. Its scope is deliberately narrow: it follows
`ABBA2Midpoint(state_extension="physical")` from public simulation input to the
returned `Solution`.

The complete ABBA family, its five public classes, three state extensions, and
all 51 valid configurations are documented in the authoritative
[`Canonical ABBA numerical architecture`](../../abba/simulation/abba-numerical-architecture.md).
This companion does not repeat that configuration matrix. In particular, the
shared-time and fully extended midpoint branches are real runtime paths, but
the diagram intentionally omits their additional `(t,kappa)` or `(t,k)`
variables.

## Scope shown by the diagram

The solid runtime path is:

```text
DynamicalSystem + InitialValueProblem + SimulationRequest
                         +
        ABBA2Midpoint(state_extension="physical")
                         |
                         v
               SimulationRunner.simulate(...)
                         |
                         v
              ABBA2Midpoint.integrate(...)
                         |
                         v
                integrate_fixed_grid(...)
                         |
                         v
                  _midpoint_abba_step(...)
                         |
                         v
          IntegrationData -> SimulationRunner -> Solution
```

The optional dashed branch emits an `IntegrationStep` only for accepted
main-grid steps. The diagram does not show an implicit projection, a residual,
a multiplier, Newton, or Broyden because none participates in this method.

## Public boundary and validation

The public method is defined in
[`src/simulation/methods/abba/order2_midpoint.py`](../../../../src/simulation/methods/abba/order2_midpoint.py).
Its physical branch accepts any object satisfying the runtime-checkable
[`DynamicalSystem`](../../../../src/dynamics/protocols.py) protocol, provided
that `state_dimension == 2`. It is not restricted to
`GuidingCenterDynamics`, and it never requests a vector-field Jacobian.

[`InitialValueProblem`](../../../../src/simulation/problem.py) supplies the
validated packed physical state and binds it to the dynamics.
[`SimulationRequest`](../../../../src/simulation/request.py) supplies the
integration interval, the upper bound on the main step, and the requested
saved times. [`SimulationRunner`](../../../../src/simulation/runner.py)
validates those public objects, calls `integrate(...)`, validates the returned
history against the source layout, and constructs the public `Solution`.

`ABBA2Midpoint` is a frozen dataclass with only three fields:

| Field | Role | Default |
|---|---|---:|
| `state_extension` | Chooses the physical, shared-time, or fully extended runtime | `"physical"` |
| `progress` | Enables the shared terminal progress display | `False` |
| `step_observer` | Receives accepted main-step observations | `None` |

Only the first row is a numerical-model axis. The diagram fixes it to
`"physical"`.

## One physical ABBA2 midpoint step

The implementation shares its explicit endpoint-time stage kernel with the
implicit ABBA methods. The neutral kernel lives in
[`src/simulation/methods/abba/_core.py`](../../../../src/simulation/methods/abba/_core.py),
while `_midpoint_abba_step(...)` owns duplication and arithmetic projection.

Let `z_n` be the accepted packed state, `h` the current main or shadow duration,
and `s=h/2`. Midpoint starts both copies on the physical diagonal:

\[
u_0=z_n,\qquad v_0=z_n.
\]

`_evaluate_unprojected_stages(...)` then applies the four explicit shears at
the two step endpoints:

\[
\begin{aligned}
u_1 &= u_0+s f(t_n,v_0),\\
v_1 &= v_0+s f(t_n,u_1),\\
v_f &= v_1+s f(t_n+h,u_1),\\
u_f &= u_1+s f(t_n+h,v_f).
\end{aligned}
\]

Every field evaluation passes through `_checked_vector_field(...)`, which
requires a finite result with exactly the candidate-state shape. The private
`_ABBAStages` record retains `u_initial`, `v_initial`, `u_first`, `v_final`,
`u_final`, and the unprojected separation `u_f-v_f`.

The accepted state is the arithmetic diagonal projection

\[
z_{n+1}=\frac{u_f+v_f}{2},
\]

and `_ABBA2MidpointStep` also returns

\[
d_n=\lVert u_f-v_f\rVert_\infty.
\]

This `d_n` is a copy-separation diagnostic. It is not a nonlinear residual, a
convergence tolerance, or an error estimate; no iteration tries to reduce it.
The name "midpoint" refers to the arithmetic midpoint of the two final copies,
not to the implicit midpoint Runge--Kutta rule.

One call of the map evaluates the vector field four times. An off-grid shadow
sample invokes another complete map and therefore incurs four additional
evaluations; `vector_field_evaluations_per_step=4` describes one map, not a
total-run evaluation counter.

## Fixed main grid and shadow samples

[`integrate_fixed_grid(...)`](../../../../src/simulation/_fixed.py) separates
the numerical trajectory from the requested output schedule. It chooses the
smallest uniform main-step count whose step does not exceed
`request.max_step`, then uses

\[
h_{\mathrm{main}}=\frac{t_f-t_0}{\text{step_count}}.
\]

For each main interval it calls the nested
`advance(t, state, step, step_index, observe)` callback with `observe=True`.
That returned state replaces the main state, contributes one copy-separation
value, advances progress, and may emit an observation.

Requested times are handled as follows:

- a time at the main-step start reuses the preceding main state;
- a time at the main-step end reuses the newly accepted main state;
- a time inside the interval triggers a shorter shadow advance from a copy of
  the preceding main state with `observe=False`.

A shadow state is saved but never replaces the main state. It does not affect
later main steps, progress, `copy_separation_norms`, or observations. Changing
`output_times` can therefore change the sampling work without changing the
underlying main-grid trajectory.

## Diagnostics

The physical branch returns
[`IntegrationData`](../../../../src/simulation/_result.py) with the requested
times, the physical packed history, and these diagnostics:

| Key | Physical-branch value or meaning |
|---|---|
| `step_count` | Number of accepted uniform main-grid steps |
| `copy_separation_norms` | One `||u_f-v_f||_infinity` value per main step |
| `projection_kind` | `"arithmetic_mean"` |
| `state_extension` | `"physical"` |
| `vector_field_evaluations_per_step` | `4` for one main or shadow map |
| `accepted_internal_state_dimension` | `2N` for `N` planar particles |
| `base_splitting_state_dimension` | `4N` for the two copies |
| `observer_state_dimension` | `2N` |
| `observer_state_kind` | `"physical_map"` |
| `nonlinear_unknown_dimension` | `0` |

The diagram abbreviates this mapping to its central midpoint quantities. The
dimension keys are generated by
[`_state_dimension_diagnostics(...)`](../../../../src/simulation/methods/abba/_configuration.py).

## Optional observation

When `step_observer` is set, the physical branch emits one
[`IntegrationStep`](../../../../src/simulation/observation.py) after every
accepted main step. It contains:

- method and dynamics names, step index, start time, end time, and duration;
- independent `state_before` and `state_after` snapshots;
- the exact dynamics instance; and
- `map_state`, a closure that reevaluates this same fixed-time, fixed-duration
  midpoint map on another packed physical state.

Shadow advances never emit events. The event does not expose the private ABBA
stage record; downstream code that needs a numerical tangent can evaluate or
differentiate `map_state` without importing private integrator helpers.

## Result boundary

After `ABBA2Midpoint.integrate(...)` returns, the existing runner call validates
`IntegrationData` and constructs
[`Solution`](../../../../src/simulation/solution.py). The public result owns
read-only copies of saved times, states, and diagnostic arrays and retains the
source initial configuration so its layout can split components and positions.

## Numerical properties and limitations

- The endpoint-time A--B--B--A base map is explicit, symmetric, and designed
  for second-order integration.
- Its four shear maps are symplectic on the duplicated phase space for the
  guiding-centre Hamiltonian structure, but arithmetic averaging does not in
  general preserve that structure on the physical diagonal. The physical
  midpoint map is therefore not guaranteed symplectic.
- The physical branch can advance packed planar states containing multiple
  particles; their concrete memory interpretation remains owned by the source
  layout, and the complete vector shares one integration time.
- A large copy separation is reported, not corrected. Users needing Hairer's
  implicit diagonal projection should select an implicit ABBA method.
- The diagram says nothing about the shared-time or fully extended midpoint
  branches, implicit ABBA formulations, higher-order compositions, potential
  internals, or downstream diagnostic algorithms.

The mathematical derivation in
[`ABBA2_implicit.tex`](../../abba2-implicit/ABBA2_implicit.tex) and its
[`compiled PDF`](../../abba2-implicit/ABBA2_implicit.pdf) derives the shared
endpoint-time A--B--B--A map, its symmetry, and its duplicated-space
symplecticity. Its later Hairer-projection proof applies to the implicit method,
not to the arithmetic midpoint closure documented here.

## Minimal public usage

```python
from simulation import ABBA2Midpoint, SimulationRequest, simulate

solution = simulate(
    problem,
    ABBA2Midpoint(
        state_extension="physical",
        progress=False,
    ),
    SimulationRequest.uniform(
        t_span=(0.0, final_time),
        max_step=max_step,
        sample_count=sample_count,
    ),
)
```

The caller supplies the validated `problem` and the physical time and sampling
parameters.
