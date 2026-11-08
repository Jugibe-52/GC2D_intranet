# BM4 implicit: code walkthrough

Current BM4 advancement is inherited from ProjectedPalindromicMethod in `palindromic/projected_base.py`. The complete recipe executes explicit A/B/T operations. See [the current operation architecture](../../palindromic/simulation/palindromic-operations.md).

The public `BM4Implicit` class is implemented in
[`palindromic/bm4_implicit.py`](../../../../src/methods/palindromic/bm4_implicit.py).
The former `methods.bm4` import adapters have been removed; public names remain
available through explicit exports. See the
[detailed architecture diagram](bm4-detailed-architecture.md),
[compact architecture](bm4-simulation-architecture.md), and
[shared family contract](../../palindromic/simulation/palindromic-simulation-architecture.md).

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

`DoubleFormulation(physical_size)` is the only state formulation. It prepares temporary copies through `lift`, `join_copies` and `split_copies`, preserving NumPy/JAX arrays and complete component blocks. Runge--Kutta and HBVM methods use the physical vector directly and instantiate no formulation. The doubled representation stores only a positive physical size; it contains no dynamics, solver or diagnostic state.

Every accepted snapshot is `IntegrationState(time, values)`: scalar time and a physical vector of shape `(d*N,)`. GC remains 2N and FC 4N with or without energy observation. Temporary doubled GC work has 4N entries; generic doubled FC work has 8N. These workspaces never become accepted states. This generic copy support does not extend GC numerical methods to FC.

`IntegrationMethod.new_run(f, initial_conditions)` prepares a fresh run without
an algebraic solver or Jacobian. `ImplicitIntegrationMethod` in `methods.implicit`
adds `new_run(f, initial_conditions, *, jacobian=None)` and the solver contract;
concrete families retain their defaults. Both bases are exported from `methods`.
Explicit methods use the common base and reject the `jacobian` keyword. There
is no separate explicit base. Execution supplies derivatives only to implicit
configurations, and JAX reads derivative settings only for those runs.

Both paths call `initialize(initial_conditions)` after binding their functions.
Initial data and layout are temporary inputs; the common base retains `f`,
metadata, status and copy count, while execution owns the initial snapshot.
Only implicit runs retain `jacobian`. Supplied derivatives have independent
particle shape `(N,d,d)`; finite differences remain dense. Solver strategies,
Broyden's projection matrix and deferred derivative validation are unchanged.
There is no `jacobian_structure`, `capture_quadrature` or method-level `progress` option.

`EnergyObserver()` records physical H and the positive analytic partial time derivative `hamiltonian_time_derivative` at accepted endpoints and requested physical samples. It does not consume internal stages or integrate momentum. Per-step diagnostics are `energy_start`, `energy_end`, `energy_change`, `energy_time_derivative_start` and `energy_time_derivative_end`, with shape `(steps,N)`. Saved histories `physical_hamiltonian` (alias `hamiltonian`), `energy_drift` and `hamiltonian_time_derivative` have shape `(N,T)`. Energy variation in a time-dependent Hamiltonian is not a numerical conservation error.
