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

`GuidingCenterDynamics` and `FullCyclotronDynamics` use NumPy coordinates and
normally a `Potential`. `JaxGuidingCenterDynamics` and `JaxFullCyclotronDynamics`
from `dynamics.jax_dynamics` inherit all physical methods and constructors,
declaring only `JaxGCCoordinates` or `JaxFCCoordinates`. Import these classes from their explicit
module; the base `dynamics` package does not load JAX. Direct compiled or
differentiated calls also require `JaxPotential`.

`PhysicalCoordinates` declares `array_namespace: ClassVar[ModuleType] = np`;
`JaxGCCoordinates` and `JaxFCCoordinates` declare the same attribute with `jnp`. These are fixed
module references, selected at class definition, without a property or runtime
input inspection. Importing `dynamics.jax_coordinates` explicitly loads optional
JAX; its inherited array operations do not import or choose a library.
Conversion, packing, shape validation and component access by index share the same
implementation. Dynamics access `self.coordinates`; GC Jacobian stacking and
FC rotation use that coordinate namespace, while the physical equations remain
in the dynamics. Backend choice no longer depends on input array types.

```python
import jax
import numpy as np
from dynamics.jax_dynamics import JaxGuidingCenterDynamics, JaxFullCyclotronDynamics
from potential import JaxPotential

jax.config.update("jax_enable_x64", True)
potential = JaxPotential.random(A=0.1, M=2, nx=12, ny=14, seed=27)
gc = JaxGuidingCenterDynamics(potential, rho=0.3, B=1.5)
fc = JaxFullCyclotronDynamics(potential, rho=0.3, eta=-0.4)
gc_field = jax.jit(gc.vector_field)(0.2, np.array([0.4, 0.5]))
fc_field = jax.jit(fc.vector_field)(0.2, np.array([0.4, 0.5, 0.1, -0.2]))
```

JAX simulation preparation calls the dynamics' internal `_as_jax()` method.
GC/FC prepare their own explicit subclasses and delegate potential conversion to
`Potential._as_jax()`, sharing samples, fitted splines and the GC effective field.
They neither repeat gyroaveraging nor modify the source. Already compatible JAX
objects are reused; unknown subclass overrides are rejected. Each component
bounds its preparation cache. Device selection, compilation caches and the SciPy
adaptive host adapter remain in execution. Base NumPy imports never load optional
JAX. Initial data and saved `States` retain structural NumPy snapshots and
unchanged component order. See the [potential contract](jax-potential-evaluation.md).

## Physical layouts and problem ownership

Dynamics declare `GCCoordinates()` for fixed `(x, y)` blocks or `FCCoordinates()` for fixed `(x, y, vx, vy)` blocks, from `dynamics.coordinates`. `PhysicalCoordinates` is a stateless base with common array operations; dimensions are fixed properties and no names or role-index fields are configured. `split()` returns an ordinary tuple. `JaxGCCoordinates` and `JaxFCCoordinates`, from `dynamics.jax_coordinates`, inherit their respective operations and only select `jnp` through `array_namespace`.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a model from dimension. `StateLayout.snapshot()` explicitly supplies an immutable NumPy convention; `describe_coordinates` validates that snapshot and its dimension. Compatibility compares the snapshot class and equality, even for equally sized arrays. Initial conditions and results own their arrays and conventions without retaining live dynamics or a potential. Custom conventions derive from `PhysicalCoordinates` and implement fixed operations instead of role tables; external adapters may supply these snapshots without inheritance. Packing and views preserve `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)`. The respective JAX classes support tracing; incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. States stores time rows `(T,d*N)`, with `states[i]` at `t[i]`; component accessors retain `(N,T)`. New solution archives use schema 2; the reader converts schema 1 columns explicitly. Archives can be interpreted without constructing dynamics or loading a field.

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
