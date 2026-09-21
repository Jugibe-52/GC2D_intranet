# Physical dynamics contracts

`src/dynamics/protocols.py` defines two general runtime-checkable protocols.
Implementations satisfy them structurally, without explicit inheritance.

| Contract | Required members | Purpose |
|---|---|---|
| `DynamicalSystem` | `state_dimension`, `vector_field(t, state)` | Integrate physical trajectories. |
| `HamiltonianSystem` | All `DynamicalSystem` members, `hamiltonian(t, state)`, `extended_momentum_derivative(t, state)` | Support optional energy-balance tracking. |

`HamiltonianSystem` extends `DynamicalSystem`. Its energy methods return one
value per particle; Hamiltonian evaluation also supports saved time histories.
The momentum derivative is `-partial_t H` with the same energy normalization
as `hamiltonian`. Autonomous systems provide an explicit zero derivative.

The protocols describe available capabilities. The formulation's `track_energy`
flag selects whether to use them: `False` requires only the basic dynamics
contract, while `True` requires `HamiltonianSystem` and appends the per-particle
time and passive momentum `kappa`. The same dynamics instance supports either
setting. Energy tracking measures the balance `H(t, state) + kappa` and does not
change the physical vector field or the physical integration decisions.

`GuidingCenterJacobianSystem` and `CyclotronSplitSystem` remain separate,
specialized contracts for analytic Jacobians and cyclotron splitting. They are
independent of the energy-tracking choice.

The former `ExtendedHamiltonianSystem` protocol has been removed. Import
`HamiltonianSystem` for energy-tracked formulations. A custom implementation
that previously supplied only `hamiltonian` must also supply the basic dynamics
members and `extended_momentum_derivative` to satisfy the unified contract.
Evaluating a Hamiltonian directly remains possible without enabling tracking.
