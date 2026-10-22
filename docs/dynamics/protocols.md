# Physical dynamics contracts

`src/contracts/dynamics/protocols.py` defines two general runtime-checkable protocols.
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

## NumPy and JAX arrays

Built-in GC/FC equations work with either potential class. Direct compiled or
differentiated calls require dynamics constructed with `JaxPotential`.
`Potential` supplies the NumPy/SciPy evaluation path. JAX simulation preparation
binds the equations to JAX potentials sharing the same fitted splines, without
mutating the original dynamics. NumPy calls remain independent of the optional
JAX installation.

State layouts preserve the array backend when reshaping, splitting and packing
component-major blocks. The physical dimensions and ordering do not change.
Integration preparation selects devices and compilation; the dynamics do not
accept execution options or own a second JAX implementation. The SciPy adaptive
controller has an explicit adapter for transfers when JAX execution is selected.
Saved `States` arrays remain NumPy. See the
[shared potential contract](jax-potential-evaluation.md) for class selection and device
placement, including NumPy-only `evaluate_grid` reconstruction.

## Physical layouts and problem ownership

`formulations.layout` owns `PackedStateLayout`, `GCState`, `GCStateLayout`, `FCState` and `FCStateLayout`. Dynamics, formulations, generators and results share these component-major operations. `GCState` and `FCState` are named component tuples, not storage classes. The structural `StateLayout` protocol remains in `contracts.formulations.layout` and requires no inheritance from a built-in layout.

`split` reshapes the packed vector or history into component blocks without calling the explicit validator. `validate_packed_state_layout` checks nonempty complete blocks; `as_blocks` also validates them. The inferred reshape in `split` requires nonempty sample axes. Operations preserve NumPy/JAX backends.

`InitialValueProblem(dynamics, initial_state, layout=...)` accepts a finite one-dimensional physical array. It captures independent array and deepcopy-compatible layout snapshots and validates particle count and dynamics dimension. GC uses `[x, y]`; FC uses `[x, y, vx, vy]`. An omitted layout selects the canonical GC/FC layout from the dynamics dimension; other dimensions require an explicit structural layout. GC doubled maps require two physical components; FC split maps require four plus `CyclotronSplitSystem`. Hamiltonian capability is independent of this layout choice.

Problem accessors return independent initial-array and layout copies. `States` similarly preserves an independent initial snapshot and layout alongside its physical history. There is no retained provider identity or `source` object. Construct a new problem to use revised initial values. The initial time belongs to `request.SimulationRequest`, and the integration controller manages current internal values and time through `IntegrationState`.

Built-in GC and FC dynamics freeze their physical parameters. Construct a new
`GuidingCenterDynamics` or `FullCyclotronDynamics` for a different potential,
`rho`, or `eta`; this keeps the GC gyroaveraged field and compiled calculations
consistent with the declared system. These objects and problem snapshots remain
pickle-compatible for process and Modal execution. Custom dynamical systems
remain responsible for keeping their own physical parameters stable during a run.
