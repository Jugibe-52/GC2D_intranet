# Architecture

GC2D has four public entities and two dynamical variants.

```text
Area --> TrajectoryGC --+
                        +--> SystemGC --+
Potential --------------|               |--> Solution
                        +--> SystemFC --+
TrajectoryFC -----------+
```

## Responsibilities

### Potential

Contains the periodic grid and the electrostatic field. It evaluates the
potential and its spatial or temporal derivatives. It also contains the
visualizations used by the notebooks.

`Potential` has no knowledge of particles, states, or integrators. `SystemGC`
builds a gyroaveraged version for its effective dynamics; `SystemFC` uses the
physical potential directly.

### Trajectory

Contains the particle parameters and its initial state:

- `TrajectoryGC`: `[x, y]` blocks and Larmor radius `rho`.
- `TrajectoryFC`: `[x, y, vx, vy]` blocks, `rho`, and `eta`.

The initial state can be assigned in the constructor or later with
`set_initial_state(...)`. `split(...)`, `pack_components(...)`, and
`particle_count(...)` centralize the physical state format. The results of
`split(...)` have named components: `[x, y]` for GC and `[x, y, vx, vy]` for FC.
A trajectory has no knowledge of the potential or the integration algorithm.

### Area

`Area` inherits from `TrajectoryGC`: its `[x, y]` blocks are points ordered
counterclockwise that delimit a square or circle. Its `Area.square(...)` and
`Area.circle(...)` constructors generate the contour, and `calculate_area(...)`
computes the initial or transported oriented area. As a GC trajectory, it can
also be used directly with `SystemGC`.

### System

Combines exactly one `Potential` with a compatible `Trajectory`:

- `SystemGC` builds the effective potential and guiding-center equations.
- `SystemFC` builds the full-cyclotron equations and the electric acceleration
  consumed by the integrator.

Both systems expose `hamiltonian(...)` and `simulate(...)`. The numerical BM4
implementation is private: it does not constitute a separate API and cannot be
selected externally. Its BM4 time grid depends only on `t_span` and `step`,
while `n_output_samples` defines a separate output grid. An output between BM4
nodes is computed by a shadow BM4 advance from the preceding integration state,
without modifying the integration trajectory or notifying stage observers.

When it contains an `Area`, `SystemGC.animate_area(...)` combines the solution
with the effective potential and electric field. The animation transports the
contour and simultaneously displays the relative error
`(A(t) - A(0)) / abs(A(0))`.

The graphical implementation lives in a private module; `SystemGC` retains the
public method because it has simultaneous access to the effective potential,
trajectory, and solution.

### Solution

This is the result of `simulate(...)`. It only carries integration times,
states, and diagnostics. It does not decide how to display, save, or analyze
the results.

## Dependencies

```text
Potential       Trajectory
     \             /
      \           /
        System GC/FC
             |
       BM4 integration
             |
          Solution
```

The integrator maintains private structures that differ from the physical state:

- GC uses two copies of the state and an optional extended momentum.
- FC only adds the optional extended momentum.

These structures, GC coupling, and the forward/adjoint FC flows belong to
`_integration`. They are assembled and disassembled through
`Trajectory.split(...)` and `Trajectory.pack_components(...)`, without
duplicating the physical layout inside `System`.

The interface supports building trajectories with `from_components(...)`, so
the notebooks do not need to know the flat ordering. Within the algorithms,
`Trajectory.as_blocks(...)` interprets the state as
`(components, particles, *samples)`, and `from_blocks(...)` restores the flat
format. Both transformations are views when the memory layout permits it.
`Solution` retains the trajectory that produced the result, and
`solution.components()` directly returns its named physical blocks.

Component-major ordering keeps all values of a quantity contiguous and favors
vectorized potential evaluations. The flat vector is retained as the stable
contract of the composition engine; the flows avoid repacking the same physical
representation, and GC coupling directly flattens the blocks produced by the
matrix operation.

Dependencies only move toward composition and integration. There are no
dependencies from `Potential` to `Trajectory` or from `Trajectory` to
`Potential`.

## Public surface

The notebooks import only:

```python
from classes import (
    Potential,
    Area,
    SystemFC,
    SystemGC,
    TrajectoryFC,
    TrajectoryGC,
)
```

Grid, interpolation, and symplectic-composition details remain internal. There
are no compatibility aliases, workflows, or command-line entry points.
