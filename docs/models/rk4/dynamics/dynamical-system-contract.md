# Dynamical-system contract for classical RK4

## Scope

`RK4` is a general explicit Runge--Kutta method. It advances any object that
structurally satisfies `DynamicalSystem`; it does not require a splitting
formulation, a Hamiltonian, a vector-field Jacobian, or a particular particle
layout.

For a physical state $z \in \mathbb{R}^d$, the required problem is

\[
\dot z=f(t,z).
\]

The [`DynamicalSystem`](../../../../src/dynamics/protocols.py) contract exposes
`state_dimension` and `vector_field(t, state)`. Every RK4 stage checks that the
returned derivative has the same packed shape as its input state.

## State ownership and compatibility

[`InitialValueProblem`](../../../../src/simulation/problem.py) binds the
dynamics to an independent initial configuration. Before integration it
requires

- a finite, one-dimensional initial state;
- a configuration whose layout satisfies `StateLayout`; and
- equality between `layout.state_dimension` and
  `dynamics.state_dimension`.

The layout, rather than RK4, defines how component blocks are packed and how
many particles are represented. Two production examples are:

| Dynamics | `state_dimension` | Component-major physical state |
|---|---:|---|
| `GuidingCenterDynamics` | 2 | `[x_1, ..., x_N, y_1, ..., y_N]` |
| `FullCyclotronDynamics` | 4 | `[x_1, ..., x_N, y_1, ..., y_N, vx_1, ..., vx_N, vy_1, ..., vy_N]` |

Consequently, the same immutable `RK4` instance can integrate guiding-centre,
full-cyclotron, or external dynamics without method-specific preparation. The
architecture tests exercise both production layouts and a minimal external
non-Hamiltonian rotation field.

## Optional extended-Hamiltonian capability

With `track_energy=False`, `DynamicalSystem` is the only dynamics capability
used. With `track_energy=True`, RK4 strictly requires
[`ExtendedHamiltonianSystem`](../../../../src/dynamics/protocols.py), which adds

- `hamiltonian(t, state)`, returning one value per particle and saved time; and
- `extended_momentum_derivative(t, state)`, returning one value per particle.

For a non-autonomous Hamiltonian, the internal triangular extension is

\[
K(t,z,k)=H(t,z)+k,
\qquad
\dot z=f(t,z),
\qquad
\dot k=-\partial_t H(t,z).
\]

RK4 appends one initially zero momentum value per particle and evaluates both
derivatives at the same four stage times and physical stage states. The
momentum never enters `f(t, z)`, so energy tracking does not change the
physical equations. Only the physical history is returned as `Solution.states`;
the saved momentum history is a diagnostic.

The implementation validates the momentum derivative shape as
`(particle_count,)`. After integration,
[`generalized_energy_error`](../../../../src/simulation/formulations/base.py)
computes the maximum saved-sample drift of $H+k$. RK4 does not claim exact
conservation of this quantity.

## Dependency boundary

RK4 depends only on the public dynamics capabilities above. Potential
evaluation, physical parameters, and component packing remain owned by the
dynamics. Initial geometry and layout remain owned by the configuration. Time
grids, output sampling, observations, and result validation remain in
`simulation`.

The companion source diagram is
[`dynamical-system-contract.puml`](dynamical-system-contract.puml).
