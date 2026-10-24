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

The method's `track_energy` flag selects optional Hamiltonian capabilities. Tracking integrates separate per-particle physical momentum and measures `H + kappa`; it changes neither physical states nor integration decisions.

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

Dynamics declare immutable `PhysicalCoordinates(component_names, position_names=(), velocity_names=())` from `dynamics.coordinates`. GC declares `(x,y)`; FC declares `(x,y,vx,vy)`. Position and velocity roles are explicit. The structural protocol is `contracts.dynamics.coordinates.StateLayout`; external implementations require no inheritance.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a physical model from dimension 2 or 4. It checks dimension, component order and position/velocity roles, even for equally sized arrays. Initial conditions and results retain an immutable structural description and owned arrays, without retaining live dynamics or a potential. Generic packing, component views and role access support `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)` while preserving NumPy/JAX tracing. Incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. Saved numerical archives retain existing GC/FC schemas and can be interpreted without constructing dynamics or loading a field.

## Dynamics-independent state representation

`SingleFormulation(physical_size)` and `DoubleFormulation(physical_size)` retain only a positive physical vector size. They prepare one or two temporary copies through `lift`, `join_copies` and `split_copies`. Shapes are `(d*N,*samples)` and `(copies*d*N,*samples)`; shifted copies remain distinct until the method projects them. NumPy/JAX backends and complete FC velocity blocks are preserved. There is no layout, particle count, tracking flag, clock, momentum, initial state, projection or solver in either representation.

Every accepted snapshot is `IntegrationState(time, values, momentum=None)`: scalar time, physical `values` of shape `(d*N,)`, and optional physical `momentum` of shape `(N,)`. GC remains 2N and FC 4N with or without tracking. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. This generic support does not extend GC numerical methods to FC.

The method initializes zero momentum, prepares copies, selects projection equations and solver, and integrates momentum from accepted stages. `advance(t, Z, h, momentum=kappa)` returns physical `StepResult.state` and separate `StepResult.momentum`. `StepInfo.momentum_before` records the incoming auxiliary. Shadow samples start from the same accepted physical state and momentum. JAX carries separate arrays through the compiled loop.

Methods evaluate saved physical H once. `states.energy.energy_diagnostics` combines supplied H and physical kappa without dynamics evaluation. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. Numerical archives require no migration; old pickled configured objects should be recreated.
