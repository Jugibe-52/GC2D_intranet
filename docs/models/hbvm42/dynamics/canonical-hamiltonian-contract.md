# Canonical Hamiltonian contract for HBVM(4,2)

## Scope

`HBVM42` advances any project `DynamicalSystem`. Its energy-preservation claim,
however, applies only when the vector field has canonical Hamiltonian form

\[
\dot y = J\nabla H(y), \qquad J^T=-J,
\]

and `dynamics.hamiltonian(t, state)` evaluates the same Hamiltonian that
generates the vector field. Physical energy is an invariant only for an
autonomous Hamiltonian. A time-dependent Hamiltonian may still be integrated,
but its physical value is not expected to remain constant without an explicit
extended-phase-space formulation.

## Required capabilities

- `state_dimension` and `vector_field(t, state)` satisfy `DynamicalSystem`.
- `hamiltonian(t, state)` satisfies `HamiltonianSystem` when
  `HBVM42(track_energy=True)` is selected.
- `particle_vector_field_jacobians(t, state)` may satisfy
  `GuidingCenterJacobianSystem`. `jacobian_method="auto"` then assembles the
  exact component-major Jacobian. Otherwise the method uses centered finite
  differences.

The packed planar layout is
`[q_1, ..., q_N, p_1, ..., p_N]`. Exact particle Jacobians have shape
`(N, 2, 2)` and are assembled into the corresponding dense component-major
matrix before the reduced Newton system is formed.

## Reproducible validation system

`studies.QuarticOscillatorDynamics` implements

\[
H(q,p)=\frac{p^2}{2}+a\frac{q^4}{4}, \qquad
\dot q=p, \qquad \dot p=-a q^3.
\]

This autonomous degree-four Hamiltonian is deliberate. Four-node HBVM(4,2)
integrates its discrete Hamiltonian line integral exactly, so the energy error
is limited by the nonlinear tolerance and floating-point arithmetic. It also
has a nonlinear flow, exposing the fact that HBVM(4,2) is not generally a
symplectic Runge--Kutta method.

## Boundaries

The dynamics owns physical equations and exact derivatives. It does not own
time grids, Newton tolerances, reference trajectories, benchmark repetition,
or plots. Those responsibilities remain in `simulation`, `studies`, and
`visualization`, respectively.
