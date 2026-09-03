# Dynamical-system contract for explicit Euler

## Scope

`ExplicitEuler` advances the physical state of any `InitialValueProblem` whose
dynamics satisfies the runtime-checkable `DynamicalSystem` protocol. For a
packed state `z in R^m`, that contract is the ordinary differential equation

\[
\dot z=f(t,z).
\]

The method advances this packed vector directly. It does not duplicate the
state, introduce a formulation-specific internal layout, or append an extended
Hamiltonian variable.

## Required capability

| `DynamicalSystem` member | Role in the explicit-Euler path |
|---|---|
| `state_dimension` | `InitialValueProblem` requires it to equal the state dimension declared by the initial-configuration layout. |
| `vector_field(t, state)` | `ExplicitEuler` evaluates it at the left endpoint of every main or shadow step. |

Both `InitialValueProblem` and `ExplicitEuler.integrate` perform a structural
runtime check for `DynamicalSystem`. The method itself does not interpret
particle blocks or component names; those remain properties of the initial
configuration and its `StateLayout`.

## Vector-field validation

The private `_checked_vector_field` boundary converts every derivative to a
floating-point NumPy array and requires

- exactly the same shape as the candidate state; and
- finite values in every component.

A shape change or a non-finite derivative raises `ValueError` before the Euler
update is accepted. The shared fixed-grid runner independently requires the
initial numerical state to be a finite, non-empty, one-dimensional vector, and
`SimulationRunner` validates the returned physical history against the source
layout.

## Capability boundary

Additional protocols implemented by a concrete system are allowed but unused.
In particular, this method does not request Hamiltonian values, extended
momentum derivatives, particle Jacobians, or cyclotron split operations. Its
only physical operation is `vector_field(t, state)`; grid construction,
observation delivery, and result ownership remain in `simulation`.

The complete integration lifecycle is documented in
[`../simulation/explicit-euler-simulation-architecture.md`](../simulation/explicit-euler-simulation-architecture.md).
The companion source diagram is
[`dynamical-system-contract.puml`](dynamical-system-contract.puml).

## Implementation sources

- [`src/dynamics/protocols.py`](../../../../src/dynamics/protocols.py)
- [`src/simulation/problem.py`](../../../../src/simulation/problem.py)
- [`src/simulation/methods/classical/euler.py`](../../../../src/simulation/methods/classical/euler.py)

