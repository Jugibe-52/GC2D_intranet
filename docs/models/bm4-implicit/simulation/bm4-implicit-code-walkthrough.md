# BM4 implicit: code walkthrough

Current BM4 advancement is inherited from ProjectedPalindromicMethod in `extended/palindromic.py`. The complete recipe executes explicit A/B/T operations. See [the current operation architecture](../../extended/simulation/palindromic-operations.md).

The public `BM4Implicit` class is implemented in
[`extended/bm4.py`](../../../../src/methods/extended/bm4.py).
The former `methods.bm4` import adapters have been removed; public names remain
available through explicit exports. See the
[detailed architecture diagram](bm4-detailed-architecture.md),
[compact architecture](bm4-simulation-architecture.md), and
[shared family contract](../../extended/simulation/extended-simulation-architecture.md).

1. Execution injects `f` and the optional physical Jacobian into a fresh run. `initialize` prepares `GCDoubledMaps`, `DoubleFormulation`, the `BM4` recipe and projection with the configured solver.
2. `advance` solves `u_final-v_final+2*mu=0`, evaluating `compose` from `(z+mu,z-mu)`.
3. Newton uses retained tangents or centered differences; Broyden starts from `4I` and updates by secants. The solver owns tolerances, differentiation, iteration limits and callbacks.
4. Only the converged trace is accepted. Its corrected copies are averaged, and quadrature points and weights are described when an observer needs them. Neither time nor momentum enters the root.
5. `advance` returns physical state and statistics. `build_observation` copies accepted stages without replaying the composition.
6. Execution supplies quadrature to `EnergyObserver`, which owns momentum and diagnostics. Shadow samples do not commit this state or main-step statistics.

The analytic path solves independent 2-by-2 reduced particle blocks. The
finite-difference path retains its dense complete-map differentiation. Neither
choice changes the reduced equation, signed coefficients,
projection placement or public result contract.

## Initial data and numerical components

The public problem constructor also accepts `states.InitialConditions`; existing
packed-array callers remain supported. See the [component interface](../../../simulation/initial-solvers-formulations.md)
for ownership, array shapes and unchanged request/result boundaries.

BM4 uses `ReducedProjection` for its existing 2N multiplier system and accepts
`solver=NewtonSolver(...)` or `solver=BroydenSolver(...)`. Composition,
physical scaling and the single outer projection remain unchanged.

## State representation and step preparation

`DoubleFormulation(physical_size)` is the only state formulation. It prepares temporary copies through `lift`, `join_copies` and `split_copies`, preserving NumPy/JAX arrays and complete component blocks. Classical, HBVM and adaptive methods use the physical vector directly and instantiate no formulation. The doubled representation stores only a positive physical size; it contains no dynamics, solver or diagnostic state.

Every accepted snapshot is `IntegrationState(time, values)`: scalar time and a physical vector of shape `(d*N,)`. GC remains 2N and FC 4N with or without energy observation. Temporary doubled GC work has 4N entries; generic doubled FC work has 8N. These workspaces never become accepted states. This generic copy support does not extend GC numerical methods to FC.

Execution binds the physical functions and calls `new_run(f, initial_conditions, request, jacobian=..., jacobian_structure=...)`. The method receives no `DynamicalSystem` or problem. `advance(t, Z, h)` returns physical state, statistics, optional details and optional quadrature data. `EnergyObserver` owns its independent momentum and accumulates only accepted quadrature; shadow samples derive from the accepted observer state without committing it. Step callbacks are passed through `simulate(..., observers=(...,))`.

Enable passive energy with `simulate(..., observers=(EnergyObserver(),))`, importing `EnergyObserver` from `diagnostics.methods.energy`. Execution supplies the observer with H and `-partial_t H`. It evaluates saved physical H once and constructs the existing diagnostics. Energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. Old configured Python objects must be recreated; saved numerical archives remain readable through versioned history conversion.
