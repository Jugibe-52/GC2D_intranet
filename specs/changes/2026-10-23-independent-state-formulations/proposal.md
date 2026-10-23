# Proposed behavior

## Representations

Define `SingleFormulation` and `DoubleFormulation` in `formulations.state` and
export them through `formulations` and `simulation`. They replace the state
representation names `PhysicalFormulation` and `DoubledFormulation`; migrate
supported consumers and document the rename without retaining retired aliases.

Both accept structural layout information and particle count, rather than
owning an `InitialValueProblem`, initial values or an initial time. They provide
explicit operations to build, split, pack and read internal arrays. A `lift(z)`
operation embeds a physical state into one copy or two equal copies. Its input
must have physical size, so an already doubled state cannot be doubled again
accidentally. All operations preserve NumPy/JAX array semantics and input values.

For N particles with d physical components, each physical copy has shape
`(d*N, *sample_axes)` in component-major order. GC has d=2 with `[x, y]`; FC has
d=4 with `[x, y, vx, vy]`. A double state concatenates two complete physical
copies, including velocities for FC. The representation does not identify the
model through a concrete dynamics type.

| Representation | GC coordinates per particle | FC coordinates per particle |
| --- | ---: | ---: |
| Single | 2 | 4 |
| Single with passive energy | 4 | 6 |
| Double | 4 | 8 |
| Double with passive energy | 6 | 10 |

Energy tracking remains independent. It appends one time block and one kappa
block of N entries each, after the physical copies. Kappa satisfies
`d kappa / dt = -partial_t H` in the existing physical Hamiltonian normalization.
Those auxiliary blocks are not duplicated or included in spatial projection
unknowns. Preserve physical units, energy normalization and saved diagnostics.

## The method owns initialization and each step

The method receives the physical initial values from the problem and the initial
time from the request. It selects the representation, creates its initial
internal state explicitly, and initializes passive momentum to zero when
tracking is enabled. The representation no longer owns an `initial_state`
factory tied to a problem. Execution continues to own the complete run and its
controller; this change does not move scheduling into the method.

For every projected doubled step, the method extracts the accepted physical
state and prepares the temporary copies. The basic sequence is:

```text
z_n -> lift to (z_n, z_n) -> compose elementary maps -> project -> z_(n+1)
```

A symmetric implicit projection instead prepares `(z_n + mu, z_n - mu)` for
each candidate multiplier. The method and its projection helper own those
guesses and corrections. Formulation helpers construct or split the arrays
when called; they do not reset copies, advance time, select a projection or
invoke a solver automatically.

Preserve diagonal accepted internal snapshots and their current dimensions;
the method explicitly repacks the accepted result and passive auxiliaries.
Each step begins from its supplied accepted state, including independent shadow
steps used for sampling. Storage never grows with the number of steps, and
reusing a method configuration does not share mutable run or solver work.

## Projections, solvers and elementary maps

Methods select and invoke separate projection equation objects. Keep the
existing public projection routes, reduced/simultaneous choices, initial guesses
and residual equations. These reusable helpers are not policies or members of
`SingleFormulation` or `DoubleFormulation`. Keep one outer symmetric projection
around the complete implicit composition and the existing arithmetic projection
for midpoint variants. Extracting a physical representative from an accepted
diagonal state must not silently average an unprojected state.

Neither new representation accepts a `solver` parameter. Newton and Broyden
remain reusable objects supplied to the method that owns the nonlinear solve.
Existing method `solver=None` retains its current meaning: use the method's
configured defaults. Explicit operations and arithmetic projection perform no
nonlinear solve. This change does not reinterpret `None` as disabling an
implicit method's required projection.

Factor reusable copy handling out of `GCDoubledMaps` and the specialized split
helpers. A/B field evaluations obtain their copies through the generic
representation. Keep specialized public split-map interfaces supported, while
the calling method controls when their operations run. Exact cyclotron flow,
planar harmonic coupling, analytic particle Jacobians and method compatibility
checks retain their existing capabilities and mathematical assumptions.
Generic FC duplication therefore does not advertise new FC support for ABBA or
BM4, or establish symplecticity for an arbitrary dynamics.

Follow the existing [state and integration conventions](../../../docs/simulation/integration-architecture.md)
and [explicit operation ordering](../../../docs/models/extended/simulation/palindromic-operations.md).
Keep stage times, coefficients, solver tolerances and observation counts.
