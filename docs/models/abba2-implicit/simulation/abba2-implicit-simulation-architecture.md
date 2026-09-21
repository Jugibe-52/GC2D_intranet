# ABBA2Implicit: state formulations and execution

One endpoint-time A-B-B-A map with one spatial Hairer projection.

The canonical theoretical source is [theory.tex](../tex/theory.tex), with its
compiled [theory.pdf](../tex/theory.pdf). The four-formulation convention below
supersedes earlier diagrams showing a duplicated clock or full time/momentum projection.

## Responsibilities

| Component | Owns |
|---|---|
| Dynamics | Physical vector field, Hamiltonian and required derivatives |
| Formulation | Internal coordinates, spatial copies and physical/energy extraction |
| Method | Stages, signed coefficients, spatial projection and passive quadrature |
| Integration | Accepted intervals, sampling, observation and output collection |

## State contract

| Planar one-particle formulation | Internal coordinates | Dimension |
|---|---|---:|
| Physical | `(x, y)` | 2 |
| Physical with energy | `(x, y, t, kappa)` | 4 |
| Duplicated | `(x1, y1, x2, y2)` | 4 |
| Duplicated with energy | `(x1, y1, x2, y2, t, kappa)` | 6 |

ABBA2Implicit uses the **duplicated** rows. `track_energy=False` is the default;
`track_energy=True` enables the energy row. With N planar particles its internal
dimensions are 4N / 6N. There are N time entries and N energy momenta.
Classical FC runs use their actual physical size 4N, giving 4N / 6N.
All components remain component-major; energy states append N times, then N
normalized momenta. Every component block has the same particle dimension.
The time entries are copies of the integration time; physical maps still receive
one scalar time. The formulation owns clock validation and output alignment.
The physical output always has its original size.
Accepted duplicated copies are equal. They separate only inside the numerical map.

`PhysicalFormulation` and `DoubledFormulation` are constructed directly from the
problem, initial time and tracking flag. They are defined in
`src/simulation/formulations/state.py`. BM4 additionally uses directly bound
`GCDoubledMaps` for its spatial direct/adjoint stages; its legacy configuration
factory is only a compatibility entry point.

## Energy and nonlinear work

The accepted shear stages update the normalized passive momentum after the spatial projection solve.

The stored momentum is physical `kappa`, initialized at zero. Its derivative is
`-partial_t H`; splitting sums are normalized by one half. The diagnostic is
`H(t, z) + kappa - H(t0, z0)`. It measures a balance, not conservation of the
time-dependent physical Hamiltonian. Dynamics must implement
[`HamiltonianSystem`](../../../dynamics/protocols.md) when tracking is enabled.
It extends `DynamicalSystem` with `hamiltonian` and
`extended_momentum_derivative`, including an explicit zero derivative for an
autonomous Hamiltonian. With tracking disabled, only `DynamicalSystem` is
required for this energy choice; method-specific capabilities still apply.

Only spatial coordinates enter a Hairer constraint. The reduced multiplier has
2N components; the ABBA simultaneous spatial solve has 6N unknowns. Clock and
momentum never enlarge these roots or affect their stopping scale. Tracking also
leaves classical physical solves and adaptive acceptance decisions unchanged.
Passive energy quadrature and optional observer reconstruction are extra work
outside the physical solver counters; reported wall time still includes them.

## Lifecycle and output

`simulate(problem, method, request)` creates a fresh run via `new_run`, validates
its formulation and calls the shared `integrate_method`. Each `advance` returns
an internal state, small work counters and method-specific accepted details.
The common collector retains samples and counters; the formulation extracts the
physical trajectory and diagnostic histories. Run resources are isolated.

Fixed methods use independent shortened maps for off-grid samples. Adaptive
methods retain one live SciPy solver whose state is always physical. Their
energy quadrature follows accepted dense output and cannot affect the error norm.
Radau Jacobians are physical-sized even when energy tracking is enabled.

Observers receive the physical map and independent snapshots. Their shapes do
not change with tracking. All energy histories have shape `(N, saved_times)`,
including `extended_time` even for a single particle. Diagnostic arrays
`extended_time`, `extended_momentum`,
`physical_hamiltonian`, `generalized_energy` and `generalized_energy_error` follow
`Solution.t`; nonlinear and runtime work arrays follow `step_times`.
`extended_momentum_normalization` is `physical_kappa` and `energy_error` is the
maximum absolute sampled balance error over all particles.

## Migration and verification

`state_extension="fully_extended"` no longer runs a time/momentum projection.
It raises explicit migration guidance. Use `track_energy=True` with spatial
projection. The historical full-state symplecticity study is retired because it
measured a different map; existing saved artifacts can still be read.

Tests in `tests/test_state_formulations.py` cover all 13 methods, both tracking
settings, particle batches, physical-only observers, adaptive control, per-particle
time alignment and energy normalization. Model tests retain order, projection,
Jacobian and nonlinear-solver checks. The pre-change physical trajectories are
also compared with the migrated implementations on short nonautonomous runs.
