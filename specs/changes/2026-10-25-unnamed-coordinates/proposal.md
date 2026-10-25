# Proposed change

Use a stateless `PhysicalCoordinates` base for common array operations.
`GCCoordinates()` fixes `(x, y)` and `FCCoordinates()` fixes `(x, y, vx, vy)`.
Their dimensions are fixed properties, not constructor fields. Position and
velocity extraction are implemented directly; there are no name or index tables.
GC has no stored velocities. `split()` returns ordinary tuples.

In `dynamics.jax_coordinates`, `JaxGCCoordinates(GCCoordinates)` and
`JaxFCCoordinates(FCCoordinates)` only select `jnp` through `array_namespace`.
The base and NumPy classes remain usable without importing JAX. Each physical
dynamics declares its respective coordinate class. Remove `JaxPhysicalCoordinates`
and the configurable constructor without compatibility aliases.

Initial data and results keep immutable NumPy coordinate snapshots. Add a
`snapshot()` operation to the structural layout contract so external adapters
explicitly supply their convention, without inferring a model from dimension.
Compatibility uses the snapshot class and equality. Fixed GC/FC map guards check
the explicit coordinate classes; they may import the shared coordinate types,
never the GC/FC equations or dynamics objects. Custom systems remain possible
through explicit coordinate implementations, not configurable role tables.

Preserve existing operation names, component-major arrays `(d*N, *samples)`,
sample axes, numerical arithmetic and GC/FC archive schemas. Older configured
object pickles must be recreated. There are no unresolved scope decisions: the
user accepted this hierarchy with its respective JAX classes before requesting
preparation and execution again.
