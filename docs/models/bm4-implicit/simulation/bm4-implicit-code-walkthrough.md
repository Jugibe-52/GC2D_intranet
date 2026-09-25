# BM4 implicit: code walkthrough

The public `BM4Implicit` class is implemented in
[`extended/bm4.py`](../../../../src/methods/extended/bm4.py).
The former `methods.bm4` import adapters have been removed; public names remain
available through explicit exports. See the
[detailed architecture diagram](bm4-detailed-architecture.md),
[compact architecture](bm4-simulation-architecture.md), and
[shared family contract](../../extended/simulation/extended-simulation-architecture.md).

1. `initialize` binds `GCDoubledMaps`, `DoubledFormulation`, the twelve-stage
   `BM4` recipe and `solve_projection` with the existing solver options.
2. `advance` calls that bound projection directly. `solve_projection` embeds
   `z` as `(z+mu,z-mu)` and evaluates the complete recipe with `compose`.
3. The spatial residual is `u_final-v_final+2*mu`. Newton forms its tangent
   using the retained shear sources, or centered differences when selected.
   Broyden starts from `4I` and updates by secants. Both use `_nonlinear.py`.
4. Only the converged trace is accepted. Its corrected copies are averaged;
   optional `momentum_increment` accumulates signed shear quadrature once,
   normalized by one half. No time or momentum variable enters the root.
5. `advance` returns the accepted workspace and existing work counters. When
   requested, `build_observation` copies the retained stages into the existing
   `ImplicitBM4IntegrationStep`; it no longer replays the composition.
6. The common integration coordinator collects main steps and handles shadow
   output samples without adding main-step observations or work records.

The analytic path solves independent 2-by-2 reduced particle blocks. The
finite-difference path retains its dense complete-map differentiation. Neither
choice changes the reduced equation, signed coefficients, coupling setting,
projection placement or public result contract.
