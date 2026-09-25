# Shared extended-space implementation

ABBA and BM4 share the numerical implementation in
[`src/methods/extended/`](../../../../src/methods/extended/).
`extended` names an internal implementation family, not a new public method.
The public method names and exports from `simulation`, `methods`, and
`methods.extended` remain available. ABBA6 now uses one outer projection,
matching ABBA4; its numerical map and observation structure change accordingly.
The canonical package is `methods.extended`; `formulations`, `integration`,
and `contracts` are sibling packages. Old `simulation.methods.*`,
`methods.abba.*`, and `methods.bm4.*` routes have been removed. Explicit public
exports refer to the canonical implementation, without module aliases or import
hooks. See the [import migration table](../../../simulation/integration-architecture.md#public-api-and-imports).

![Shared extended-space architecture](extended-simulation-architecture.svg)

The canonical mathematical entry point is [theory.tex](../tex/theory.tex), with
its deliberate compiled [theory.pdf](../tex/theory.pdf). Diagram sources and
views can be regenerated with
`python scripts/render_extended_architecture.py`. That script writes matching
PlantUML/Graphviz graph specifications and fixed-layout SVG/PNG views from one
node-and-edge specification; an external Graphviz executable is not required.

## Execution and responsibilities

The ordinary implicit path is `advance -> solve_projection -> compose ->
direct_map / adjoint_map`. Initialization binds the recipe, spatial formulation,
solver options and Jacobian strategy once. The generic integration coordinator
continues to own step scheduling, output sampling, observer dispatch and history
collection. No per-order forwarding modules or separate ABBA composition drivers
remain. Diagnostic analysis consumes accepted events; independent residual
equations used only for verification live under `tests/_extended_reference/`.

| Module or component | Responsibility |
|---|---|
| `extended/core/composition.py` | Validated immutable signed recipes and one direct/adjoint traversal |
| `extended/core/projection.py` | Reduced and simultaneous spatial Hairer equations for any recipe |
| `methods/_nonlinear.py` | Shared Newton/Broyden convergence, iteration limits and counters |
| `extended/core/jacobians.py` | Exact ordered particle-block tangents and centered-difference fallback |
| `extended/core/energy.py` | Passive normalized momentum quadrature from accepted shear inputs |
| `extended/core/records.py` | Numerical stage traces, projection results and aggregate statistics |
| `extended/core/midpoint.py` | Explicit arithmetic projection after the complete composition |
| `extended/abba.py`, `extended/bm4.py` | Public implicit and midpoint configurations and accepted-step operations |
| `extended/configuration.py` | Shared selectors and configuration validation |
| `extended/observations.py` | On-demand ABBA pair and BM4 stage events from common numerical traces |
| `formulations/gc.py` | `GCDoubledMaps`: spatial direct/adjoint maps and optional coupling |
| `formulations/state.py` | `DoubledFormulation`: accepted internal state and physical/energy extraction |

The family has one shallow shared engine directory:

```text
extended/
  __init__.py
  abba.py
  bm4.py
  configuration.py
  observations.py
  core/
    __init__.py
    composition.py
    projection.py
    midpoint.py
    jacobians.py
    energy.py
    records.py
```

ABBA4 and ABBA6 inherit the same preparation, advancement and placement
validation. Only their immutable recipes and formal orders differ. Coefficients
and recipes live together in `core/composition.py`. ABBA-specific observation
views stay outside the numerical core. Public exports are explicit; retired
`order*_implicit`, `order2_midpoint`, `bm4_midpoint`, `abba_steps`, and composition
adapter modules are not recreated as compatibility routes.

## Recipes and numerical identity

| Public method | Unprojected recipe | Diagonal projection per complete step |
|---|---|---|
| `ABBA2Implicit` | Adjoint/direct pair with weights `(1/2, 1/2)` | One reduced or simultaneous Hairer solve |
| `ABBA4Implicit` | Three continuous ABBA pairs: `(gamma, delta, gamma)` | One reduced or simultaneous solve around all six stages |
| `ABBA6Implicit` | Seven continuous signed ABBA pairs with Yoshida weights | One reduced or simultaneous solve around all fourteen stages |
| `BM4Implicit` | Twelve alternating stages with mirrored BM4 weights | One reduced Hairer solve |
| `ABBA2Midpoint` | One ABBA pair | One arithmetic mean |
| `BM4Midpoint` | Complete twelve-stage BM4 recipe | One arithmetic mean |

`ABBA4Implicit` is the fourth-order public configuration. ABBA uses uncoupled maps. BM4 preserves its configurable harmonic
coupling and defaults (`0` for implicit BM4, `pi/8` for midpoint BM4).

In execution order, even-indexed stages are adjoint and odd-indexed stages are
direct. If `s = coefficient * h`, an adjoint evaluates at the current stage
clock and a direct map at `clock + s`; then the clock advances by `s`. Negative
weights reverse both the signed subflow and its clock. Reflection exchanges
adjoint and direct maps. Palindromic weights and unit total duration are checked,
but they alone do not prove fourth- or sixth-order accuracy.

Projection placement is part of method identity. ABBA6 intentionally changes
from seven projected pairs to one projection around fourteen stages. Historical
ABBA6 trajectories and timings must not be relabeled as results of the new map.
An arithmetic mean does not imply exact symplecticity or reversibility.

## State, projection and derivatives

For N independent planar particles the physical state has 2N components in
component-major order. The numerical composition uses two spatial copies with
4N components. Accepted copies are equal. `track_energy=True` appends N times
and N normalized momenta to the accepted workspace, producing 6N components;
the physical trajectory remains 2N dimensional. See the shared
[dynamics protocols](../../../dynamics/protocols.md).

With `E z = (z,z)`, `G = [I,-I]` and `N = G.T`, the reduced equation is
`r(mu) = G Psi_h(E z + N mu) + 2 mu = 0`. Its Jacobian is
`G D(Psi_h) N + 2 I`. Broyden starts from `4 I` and evaluates no analytic
Jacobians. The simultaneous ABBA equation retains both output copies and the
multiplier as 6N unknowns; its zero-step block Jacobian initializes Broyden.
Both formulations call the same general Newton/Broyden primitives.

Analytic Newton composes exact stage tangents from retained shear sources and
solves independent particle blocks (2 by 2 reduced, 6 by 6 simultaneous). It
replays no vector-field stages to obtain those sources. BM4's existing optional
centered-difference strategy differentiates the complete doubled map instead.
All roots retain the physical scale `atol + rtol * max(1, ||z||_inf)`.

## Energy, sampling and observation

Only the converged residual's trace is used for accepted energy quadrature.
`Delta kappa = (1/2) sum_j s_j (-partial_t H)(t_j, z_j)` sums individual
signed **shears**, two per direct/adjoint stage. The factor one half converts the
doubled Hamiltonian's summed momentum to physical kappa. Coupling does not alter
that passive momentum. The diagnostic is `H(t,z) + kappa - H(t0,z0)`;
it measures energy balance, not conservation of the time-dependent physical H.

Tracking neither enlarges a nonlinear root nor adds spatial field evaluations,
and all public observers receive independent physical snapshots. Numerical
traces do not allocate public events when no observer is requested. Shadow steps
for off-grid output remain independent and do not emit main-step observations.
Per-particle energy histories align with saved times; nonlinear work arrays align
with accepted main steps. ABBA6 reports `nonlinear_solves_per_step=1`, `implicit_substeps_per_step=1`,
`unprojected_abba_maps_per_step=7`, `composition_stage_count=14`, and
`base_composition="unprojected_abba6_yoshida"`. Its `substep_nonlinear_*` and
`substep_projection_multiplier_norms` histories have shape `(steps, 1)`.
`ABBA4ImplicitIntegrationStep` and `ABBA6ImplicitIntegrationStep` share the
`ABBAImplicitCompositionIntegrationStep` contract: one outer multiplier and
respectively three or seven `UnprojectedABBAIntegrationStep` records. Reduced
and simultaneous roots both expose the same physical observation domain.
The ABBA6 exact-tangent function and reversibility observer differentiate this
single outer root; they do not multiply seven projected physical Jacobians.

## Verification

`tests/test_extended_family.py` checks common solver binding, public class
identity, coefficient validation, projection placement, endpoint shear
agreement, signed nonautonomous reversibility, analytic tangents, and passive
tracking/observation without extra spatial work. Existing method tests retain
convergence order, projection equations, observer compatibility, energy balance,
vectorized particles and sampling contracts. `tests/test_abba6.py` additionally
checks sixth-order convergence for nonlinear nonautonomous dynamics with both
root formulations and Newton/Broyden, signed reversal, the outer constraint,
analytic versus finite-difference tangents and symplecticity. The ABBA6 change
is intentional; the other methods preserve their numerical maps.
