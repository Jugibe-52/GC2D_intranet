# ABBA2 implicit simulation architecture

This document accompanies
[`abba2-implicit-simulation-architecture-current-vscode.puml`](abba2-implicit-simulation-architecture-current-vscode.puml).
The diagram preserves the horizontal Graphviz structure of the earlier VS Code
view while describing the current unified `ABBA2Implicit` runtime.

## Configuration cube

`ABBA2Implicit` is the only public second-order implicit ABBA class. Its three
independent selector axes provide twelve supported configurations:

| Axis | Choices |
|---|---|
| `projection_formulation` | `"reduced_multiplier"`, `"simultaneous_state_multiplier"` |
| `state_extension` | `"physical"`, `"shared_time"`, `"fully_extended"` |
| `nonlinear_solver` | `"newton"`, `"broyden"` |

The frozen `_ABBAImplicitConfig` stores these selectors together with solver
tolerances, the iteration limit, progress selection, and the optional step
observer. Configuration names and state-dimension diagnostics are centralized
in `src/simulation/methods/abba/_configuration.py`.

The former `ABBA2SharedTimeExtendedImplicit` and
`ABBA2FullyExtendedImplicit` classes are not part of the current public API.
Their behavior is selected through `ABBA2Implicit(state_extension=...)`.

## Common public path

Every run begins with the same public composition:

```text
InitialValueProblem + SimulationRequest + ABBA2Implicit
                         |
                         v
              SimulationRunner.simulate(...)
                         |
                         v
                 ABBA2Implicit.integrate(...)
```

`InitialValueProblem` binds opaque dynamics to an `InitialConfiguration`.
The configuration exposes a `StateLayout`, which validates and interprets the
packed physical state. `SimulationRunner` validates the public arguments,
invokes the method, checks the returned physical arrays, and constructs an
immutable `Solution`.

## State-extension dispatch

`ABBA2Implicit.integrate(...)` selects one of two coordinators.

### Physical and shared-time path

The `physical` and `shared_time` variants call
`_integrate_projected_abba(...)` in
`src/simulation/methods/abba/_implicit.py`. The coordinator:

1. validates the guiding-center Jacobian capability;
2. selects a physical projection solver with `_step_solver_for(...)`;
3. creates the fixed-grid `advance(...)` callback;
4. collects nonlinear and projection diagnostics; and
5. returns physical output through `IntegrationData`.

The reduced formulation solves for one multiplier in `R^(2N)`. The
simultaneous formulation solves for `(u_f, v_f, mu)` in `R^(6N)`. Both use the
same endpoint-time A--B--B--A stage kernels and return `_ProjectedStep`.

For `shared_time`, the accepted internal state is `(z,t,kappa)` and
`_shared_time_kappa_increment(...)` advances `kappa = k/2` after the projected
physical step. The public trajectory still contains only the physical state;
extended coordinates are retained as diagnostics.

### Fully extended path

The `fully_extended` variant calls `_integrate_abba_fully_extended(...)` in
`src/simulation/methods/_fully_extended.py`. It accepts one-particle
`(z,t,k)` states in `R^4`, constructs the duplicated `(Z_1,Z_2)` base map in
`R^8`, and applies a full diagonal projection.

`_solve_abba_fully_extended_step(...)` builds the ABBA2 base map and dispatches
to either:

- `_solve_abba_full_reduced_projection(...)`, with an `R^4` multiplier; or
- `_solve_abba_full_simultaneous_projection(...)`, with an `R^12`
  `(Z_1,Z_2,mu)` nonlinear workspace.

The full branch retains the exact base-map Jacobian when observation requires
it. `_FullProjectedStep`, `_AcceptedFullSubstep`, and `_FullMethodStep` carry
the accepted internal result through the coordinator. Only the physical
`z` slice is transferred to the public `Solution`; extended coordinates and
generalized-energy quantities remain in diagnostics.

## Nonlinear solvers

Exact Newton iterations are implemented inside each reduced or simultaneous
formulation because each branch owns a different residual and analytic
Jacobian. Broyden iterations reuse `_solve_broyden(...)` from
`src/simulation/methods/_nonlinear.py`.

For one guiding-center particle, the relevant dimensions are:

| State extension | Accepted internal state | Base splitting state | Reduced unknown | Simultaneous unknown |
|---|---:|---:|---:|---:|
| `physical` | 2 | 4 | 2 | 6 |
| `shared_time` | 4 | 6 | 2 | 6 |
| `fully_extended` | 4 | 8 | 4 | 12 |

## Fixed grid and observations

Both coordinators delegate time scheduling to `integrate_fixed_grid(...)`.
Main-grid advances use `observe=True`, replace the accepted trajectory, update
diagnostics, and may emit one event. Off-grid requested samples use a shadow
advance from the preceding main node with `observe=False`; they do not modify
later states or emit events.

Physical and shared-time main steps emit
`ABBA2ImplicitIntegrationStep`. Fully extended main steps emit
`FullyExtendedImplicitIntegrationStep`, including `FullyExtendedBaseMap`
snapshots when an observer is installed.

## Principal files

| File | Responsibility |
|---|---|
| `src/simulation/methods/abba/order2_implicit.py` | Public method and state-extension dispatch |
| `src/simulation/methods/abba/_configuration.py` | Canonical selector axes and dimension diagnostics |
| `src/simulation/methods/abba/_implicit.py` | Shared configuration and physical/shared-time coordinator |
| `src/simulation/methods/abba/_projection_reduced.py` | Reduced physical projection |
| `src/simulation/methods/abba/_projection_simultaneous.py` | Simultaneous physical projection |
| `src/simulation/methods/abba/_core.py` | Projection-independent ABBA2 stages |
| `src/simulation/methods/_fully_extended.py` | Fully extended base map, projections, tangents, and coordinator |
| `src/simulation/methods/_nonlinear.py` | Shared nonlinear validation and Broyden service |
| `src/simulation/_fixed.py` | Main-grid and shadow-sample scheduling |
| `src/simulation/observation.py` | Physical and fully extended step events |
| `src/simulation/runner.py` | Public validation and `Solution` construction |
