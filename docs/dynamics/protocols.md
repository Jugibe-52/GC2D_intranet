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
Saved `Solution` arrays remain NumPy. See the
[shared potential contract](jax-potential-evaluation.md) for class selection and device
placement, including NumPy-only `evaluate_grid` reconstruction.

## Physical layouts and problem ownership

`contracts.state_layout` owns `PackedStateLayout`, `GCState`, `GCStateLayout`,
`FCState`, and `FCStateLayout`. Dynamics, formulations, and initial-condition
builders use these same component-major operations. Existing imports from
`initial_conditions`, `initial_conditions.base`, `initial_conditions.gc`, and
`initial_conditions.fc` remain explicit reexports of the canonical classes;
the private duplicate `dynamics._layout` module has been removed.

`split` reshapes the state directly and returns its component blocks, without
calling the explicit layout validator. Use `validate_packed_state_layout` when
non-empty particle blocks must be checked; `as_blocks` still performs that
validation. The inferred reshape in `split` requires non-empty sample axes.

An external initial-state provider implements `InitialConfiguration` and supplies
a `StateLayout`. Layouts must be deepcopy-compatible and interpret component-major
particle blocks: GC uses `[x, y]`, and FC uses `[x, y, vx, vy]`. GC doubled maps
require two physical components; FC split maps require four components plus
`CyclotronSplitSystem`. Neither requires inheritance from a built-in initial
configuration. The optional Hamiltonian capability remains independent.

`InitialValueProblem` captures and validates the state, layout, and particle count
at construction. Its `initial_state` and `layout` properties return independent
copies. `initial_configuration` retains the original provider for provenance;
changing that provider afterward cannot change the problem or a formulation
prepared from it. `Solution` similarly owns independent `layout` and `initial_state`
snapshots while retaining its original `source` as provenance. Interpretation and
persistence use those snapshots, so editing an initial-condition builder after a
run cannot relabel its saved state. Construct a new problem to use a revised
initial condition.

Built-in GC and FC dynamics freeze their physical parameters. Construct a new
`GuidingCenterDynamics` or `FullCyclotronDynamics` for a different potential,
`rho`, or `eta`; this keeps the GC gyroaveraged field and compiled calculations
consistent with the declared system. These objects and problem snapshots remain
pickle-compatible for process and Modal execution. Custom dynamical systems
remain responsible for keeping their own physical parameters stable during a run.
