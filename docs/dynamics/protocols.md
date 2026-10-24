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

Registering `EnergyObserver()` through `observers=` selects optional Hamiltonian capabilities in execution. Tracking integrates separate per-particle physical momentum and measures `H + kappa`; it changes neither physical states nor integration decisions.

`GuidingCenterJacobianSystem` and `CyclotronSplitSystem` remain separate,
specialized contracts for analytic Jacobians and cyclotron splitting. They are
independent of the energy-tracking choice.

The former `ExtendedHamiltonianSystem` protocol has been removed. Import
`HamiltonianSystem` when preparing energy observers. A custom implementation
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

`DoubleFormulation(physical_size)` retains only a positive physical vector size.
It prepares two temporary copies through `lift`, `join_copies` and `split_copies`.
Shapes are `(d*N,*samples)` and `(2*d*N,*samples)`; shifted copies remain distinct
until projection. NumPy/JAX backends and complete FC velocity blocks are preserved.
The helper has no layout, particle count, tracking flag, clock, momentum, initial
state, projection or solver. Ordinary methods use physical arrays directly;
`SingleFormulation` is removed.

Each accepted `IntegrationState(time, values)` contains scalar time and physical
`values` of shape `(d*N,)`. GC remains 2N and FC 4N with or without observation.
Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are
temporary and never become accepted states. Generic packing does not extend
GC numerical methods to FC.

Execution retains the problem's source identity and binds `f(t, Z)` and optional
analytic derivatives before calling `new_run(f, initial_conditions, request, *,
jacobian=None)`. Derivatives explicitly distinguish dense matrices from
independent-particle blocks. Methods do not retain dynamics or inspect bound
function owners. FC splitting composes the supplied `cyclotron_flow(t, state,
duration)` and `electric_kick(t, state, duration)` functions; their physical
equations belong to `FullCyclotronDynamics`.

The method prepares copies, projection equations and physical stages.
`advance(t, Z, h)` returns physical `StepResult.state` and optional numerical
quadrature data. The energy observer owns separate zero or supplied initial
momentum, advances it once after acceptance, and evaluates saved H once.
Fixed shadow samples use shortened-map quadrature from the left observer state;
adaptive samples use the accepted interpolant. JAX carries a separate functional
energy accumulator through the compiled execution loop.

`states.energy.energy_diagnostics` combines supplied H and physical kappa without
dynamics evaluation. Existing energy labels and `(N,T)` auxiliary histories stay
unchanged; `extended_time` is broadcast from scalar sample times. Numerical
archives require no migration; old pickled configured objects should be recreated.
