# Trajectory calculation architecture

This document accompanies the single horizontal PlantUML overview in
[`outline-architecture.puml`](outline-architecture.puml). The diagram follows
the code path that turns a physical model, an initial state, a numerical method,
and a time request into an immutable `Solution`.

## Reading direction

Read the diagram from left to right through four groups:

1. **Dynamics** supplies the physical vector field. This group is deliberately
   compact because potentials and physical dynamics are documented separately
   in [`dynamics/gc2d-h5-import.md`](dynamics/gc2d-h5-import.md).
2. **Initial State** defines packed state layouts, concrete GC/FC initial
   configurations, optional area contours, and the `InitialValueProblem` that
   checks compatibility between a configuration and its dynamics.
3. **Numerical Model** contains the temporal request, the numerical-method
   contract, every public method implementation, BM4 formulation contracts,
   and the shared integration functions that coordinate a trajectory.
4. **Solution** shows the public `simulate(...)` façade, its runner, the
   internal `IntegrationData` transfer object, and the immutable public result.

Class boxes list attributes first and methods second. A method is written as
`name(inputs): output`. Standalone function boxes state `IN` and `OUT`
explicitly. Solid generalization arrows represent inheritance; dashed
generalization arrows represent structural protocol implementation; dependency
arrows identify calls or returned values.

## Canonical runtime path

```text
Potential
  -> GuidingCenterDynamics or FullCyclotronDynamics
  -> InitialValueProblem <- InitialConfiguration
  -> NumericalMethod.integrate(problem, request)
  -> integrate_fixed_grid(initial_state, request, advance, ...)
  -> IntegrationData(t, states, diagnostics)
  -> SimulationRunner validation
  -> Solution(t, states, source, diagnostics)
```

The public entry point wraps that path:

```python
solution = simulate(problem, method, request)
```

`SimulationRunner` calls the selected method's `integrate(...)` operation. The
method owns its internal state and stepping algorithm, but it must return
physical states aligned exactly with `request.output_times`. The runner checks
that contract before constructing `Solution` and retaining the original initial
configuration as `solution.source`.

## Numerical-method catalogue

All fifteen selectable classes implement the same operation:

```text
integrate(
    problem: InitialValueProblem,
    request: SimulationRequest,
): IntegrationData
```

| Family | Public classes |
| --- | --- |
| Classical | `ExplicitEuler`, `RK4`, `GaussLegendre4` |
| HBVM | `HBVM42` |
| ABBA | `ABBA2Midpoint`, `ABBA2Implicit`, `ABBA4Implicit`, `ABBA4ImplicitSingleProjection`, `ABBA6Implicit` |
| BM4 | `BM4Composition`, `ProjectedBM4Composition`, `MidpointBM4`, `BM4Implicit1`, `BM4Implicit2`, `BM4_implicit2` |

The shared private configuration classes in the diagram avoid repeating the
same inherited attributes on every ABBA or BM4 class. BM4's reusable
`DirectAdjointFormulation` boundary is shown separately because
`BM4Composition` accepts GC or FC formulations and prepares per-run direct and
adjoint maps.

## Scope

The overview includes the canonical, non-deprecated initial-state hierarchy,
all public numerical methods, formulation interfaces, family-level integration
coordinators, and the mandatory result path. It intentionally excludes:

- deprecated compatibility aliases such as `Trajectory`, `TrajectoryGC`, and
  `TrajectoryFC`;
- optional observer event records, which do not participate in the mandatory
  `IntegrationData -> Solution` path;
- nested step closures and most small validation, array-packing, residual,
  Jacobian, and matrix helpers (the shared dynamics packing helpers and
  `gc_coupling_matrix` remain visible because they define module boundaries);
  and
- private per-step payload dataclasses.

Those implementation details do not introduce another architectural boundary
and would obscure the four-group flow. Numerical algorithms, projection
variants, nonlinear systems, and observer payloads remain available in the
[model-specific documentation index](models/README.md).
