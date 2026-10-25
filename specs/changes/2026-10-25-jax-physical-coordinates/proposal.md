# Coordinate classes select the array library

`PhysicalCoordinates` declares `array_namespace: ClassVar[ModuleType] = np`.
`JaxPhysicalCoordinates`, defined in `dynamics.jax_coordinates`, inherits the
immutable names, roles, shape validation and component operations, declaring
only the corresponding `jnp` module attribute. The module is fixed at class
definition; no property, input inspection or library lookup runs per operation.
JAX classes are imported from their explicit modules only when requested, so
base dynamics imports remain independent of optional JAX. Both retain component-major layouts
`(d*N, *samples)` and named component views; GC uses d=2 and FC d=4.

All GC/FC methods use `self.coordinates`. Array construction, GC Jacobian
stacking and the FC rotation use its namespace, with equations remaining in
the dynamics. `JaxGuidingCenterDynamics` and `JaxFullCyclotronDynamics` in
`dynamics.jax_dynamics` inherit the existing classes and declare only JAX
coordinates. They do not duplicate physical methods or introduce constructors.

Execution prepares these explicit subclasses with `JaxPotential`, sharing the
original samples, splines and GC effective potential without refitting,
gyroaveraging again or modifying the source. Exact supported types, bounded
reuse, device selection and the hybrid SciPy adapter remain explicit.

Existing `simulate` calls continue selecting JAX with `ExecutionOptions`.
Standalone compiled calls use a JAX dynamics class with `JaxPotential`; the
base coordinate class now consistently returns NumPy arrays. Initial conditions
and saved results keep their structural NumPy coordinate snapshots. Generic
formulation helpers retain their existing backend-preserving behavior.

No physical units, integration algorithms, output schemas or method support
change. No unresolved decisions remain. See the
[physical contracts](../../../docs/dynamics/protocols.md).
