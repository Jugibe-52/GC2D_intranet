# Physical dynamics contracts

`src/contracts/dynamics/protocols.py` defines two general runtime-checkable protocols.
Implementations satisfy them structurally, without explicit inheritance.

GC evaluates H from its already normalized Hamiltonian potential and obtains positive partial_t H with `evaluate(t,x,y,dt=1)`. Space is represented by splines and time by Fourier factors; time differentiation is analytic, including the frequency factors. FC differentiates the electric-potential term with its existing physical scaling. The result is a partial time derivative at fixed physical state, not a finite difference of sampled energy or a total derivative along the numerical trajectory.

`DynamicalSystem` requires `state_dimension` and `vector_field(t,state)`. `HamiltonianSystem` additionally requires `hamiltonian(t,state)` and positive `hamiltonian_time_derivative(t,state)`. Both energy operations return `(N,)` for a scalar time and `(N,T)` for a saved history; autonomous models explicitly return zeros with the same shape. Jacobian and exact-flow capabilities remain separate protocols.

`EnergyObserver()` records physical H and the positive analytic partial time derivative `hamiltonian_time_derivative` at accepted endpoints and requested physical samples. It does not consume internal stages or integrate momentum. Per-step diagnostics are `energy_start`, `energy_end`, `energy_change`, `energy_time_derivative_start` and `energy_time_derivative_end`, with shape `(steps,N)`. Saved histories `physical_hamiltonian` (alias `hamiltonian`), `energy_drift` and `hamiltonian_time_derivative` have shape `(N,T)`. Energy variation in a time-dependent Hamiltonian is not a numerical conservation error.

`GuidingCenterJacobianSystem` and `CyclotronSplitSystem` remain separate,
specialized contracts for analytic Jacobians and cyclotron splitting. They are
independent of the energy-tracking choice.

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
`Potential._as_jax()`, sharing samples, fitted splines and GC `hamiltonian_potential`.
They neither repeat gyroaveraging nor modify the source. Already compatible JAX
objects are reused; unknown subclass overrides are rejected. Each component
bounds its preparation cache. Device selection, compilation caches remain in execution. Base NumPy imports never load optional
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
analytic derivatives before calling `new_run(f, initial_conditions, *,
jacobian=None)`. Supplied derivatives always use independent particle blocks; numerical differences are dense. Methods do not retain dynamics or inspect bound
function owners. FC splitting composes the supplied `cyclotron_flow(t, state,
duration)` and `electric_kick(t, state, duration)` functions; their physical
equations belong to `FullCyclotronDynamics`.

Specialized historical generalized-energy studies explicitly use stage diagnostics under `diagnostics.methods.stage_energy` and `studies._generalized_energy`. They retain their signed quadrature, physical kappa normalization and archive labels. Generic execution has no momentum accumulator. `States.k` and `States.err` remain readers for historical diagnostics and return `None` when absent. Old numerical archives remain readable; recreate configured-object pickles after the API change.

## Guiding-center Hamiltonian

`GuidingCenterDynamics.hamiltonian_potential` is the prepared `Potential`
representing `H = Phi_rho / B`. Construction applies the selected gyro-radius
in the source coordinate units and divides the mean and modes by finite,
nonzero signed B. `vector_field` evaluates `H_x` and `H_y` through the existing
splines and packs `(-H_y, H_x)`. `hamiltonian(t, state)`, particle Jacobians and
`hamiltonian_time_derivative` all use the same object and normalization.
JAX preparation shares its splines without rebuilding them.

`Potential.electric_field` retains `(-Phi_x, -Phi_y)` and its full-cyclotron
use. GC drift uses Hamiltonian derivatives directly. The ABBA step diagnostic
retains its historical `effective_electric_fields_after` name and values,
which represent `-grad H`, including gyroaveraging and magnetic scaling.
Study results exposing the prepared object also use `hamiltonian_potential`.
See the [API migration guide](../simulation/api-migration.md#guiding-center-hamiltonian).
