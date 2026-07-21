# GC2D

Research code for studying particle trajectories in a two-dimensional
electrostatic potential.

The project is organized around four entities:

```text
Potential + Trajectory -> System -> Solution
```

- `Potential` represents the physical field on a periodic grid.
- `TrajectoryGC` and `TrajectoryFC` describe the state of one or more
  particles.
- `Area` specializes a GC trajectory as a square or circular contour.
- `SystemGC` and `SystemFC` combine a potential with a trajectory and perform
  the time integration.

The notebooks in `notebooks/developements/` are the only entry points. There
are no command-line executables, external configuration, or workflow API.

## Installation

Python 3.11 or later is required.

```bash
python -m pip install -r requirements.txt
```

The package is installed in editable mode, so changes in `src/` are immediately
available to the notebooks.

## GC example

```python
import numpy as np

from classes import Potential, SystemGC, TrajectoryGC

potential = Potential.random(
    A=0.7,
    M=25,
    nx=64,
    ny=64,
    seed=27,
    interpolation_order=5,
)

x0 = potential.grid.xmin + potential.grid.period / 2
y0 = potential.grid.ymin + potential.grid.period / 2

trajectory = TrajectoryGC.from_components(
    x=np.array([x0]),
    y=np.array([y0]),
    rho=0.3,
)

system = SystemGC(potential, trajectory, coupling_frequency=2.0)
solution = system.simulate(
    t_span=(0.0, 6 * np.pi),
    step=0.001,
    n_output_samples=361,
    check_energy=True,
    progress=True,
)
```

`SystemGC` integrates guiding-center motion over the gyroaveraged effective
potential. The original physical potential remains in `potential`.
`coupling_frequency` controls the numerical coupling between the two internal
copies in the GC integrator; it does not represent a physical frequency. Its
default value is `pi / 8`, and it can be adjusted when the system is built.

## FC example

```python
import numpy as np

from classes import Potential, SystemFC, TrajectoryFC

potential = Potential.random(
    A=0.7,
    M=25,
    nx=64,
    ny=64,
    seed=27,
    interpolation_order=5,
)

x0 = potential.grid.xmin + potential.grid.period / 2
y0 = potential.grid.ymin + potential.grid.period / 2

trajectory = TrajectoryFC.from_components(
    x=np.array([x0]),
    y=np.array([y0]),
    vx=np.array([1.0]),
    vy=np.array([0.0]),
    rho=0.3,
    eta=0.01,
)

system = SystemFC(potential, trajectory)
solution = system.simulate(
    t_span=(0.0, 2 * np.pi),
    step=0.001,
    n_output_samples=101,
    check_energy=True,
)
```

`SystemFC` integrates the full cyclotron orbit over the physical potential.

## State organization

States use blocks rather than interleaved values. For `N` trajectories:

```text
GC: [x_1, ..., x_N, y_1, ..., y_N]
FC: [x_1, ..., x_N, y_1, ..., y_N,
     vx_1, ..., vx_N, vy_1, ..., vy_N]
```

For example, two GC states are written as follows:

```python
trajectory = TrajectoryGC(rho=0.3)
state = trajectory.pack_components(
    np.array([x1, x2]),
    np.array([y1, y2]),
)
trajectory.set_initial_state(state)

components = trajectory.split(state)
print(components.x, components.y)
print(trajectory.particle_count(state))  # 2
```

For user code, the semantic constructor is recommended because it avoids a
dependency on that internal ordering:

```python
trajectory = TrajectoryGC.from_components(
    x=np.array([x1, x2]),
    y=np.array([y1, y2]),
    rho=0.3,
)
```

`as_blocks(...)` exposes a view with shape
`(components, particles, *samples)`, and `from_blocks(...)` performs the inverse
transformation. The integrator uses these explicit axes internally, while
retaining the flat vector as the stable input and output format.
`pack_components(...)` obtains that vector directly from the class;
`from_components(...)` uses it to build the trajectory in a single step.

Trajectories accept the initial state in the constructor and also allow it to
be replaced with `set_initial_state(...)`. Reusable contour geometries belong
to `Area`; other experiment-specific conditions are prepared in the notebook.

`TrajectoryGC.split(...)` returns components named `x` and `y`;
`TrajectoryFC.split(...)` adds `vx` and `vy`. The `pack_components(...)` class
method performs the inverse operation. This makes the physical state format the
trajectory's responsibility, so the integrator does not need to duplicate its
structure.

## Areas

`Area` is a GC trajectory whose points delimit an oriented contour. It can be
built as a square or circle and passed directly to `SystemGC`:

```python
from classes import Area

area = Area.square(
    center=(np.pi, np.pi),
    side=1.0,
    points_per_side=40,
    rho=0.3,
)
system = SystemGC(potential, area)
solution = system.simulate(step=0.005)

transported_area = area.calculate_area(
    solution.y,
    period=potential.grid.period,
)

animation = system.animate_area(
    solution,
    frames=120,
    interval=50,
)
```

The alternative `Area.circle(...)` constructor takes `center`, `radius`, and the
total number of points on the contour. `calculate_area(...)` accepts both the
initial state and a complete time series and applies the shoelace formula; with
`period`, it also handles crossings of the periodic boundary correctly.
`SystemGC.animate_area(...)` displays the contour over the effective potential
and its electric field, together with
`(A(t) - A(0)) / abs(A(0))` in a second panel.
`SystemGC.animate_area_comparison(...)` overlays several solutions saved at the
same times and uses a consistent color for each solution across the contour,
area error, and optional projected diagnostics.

## Integration and results

`System.simulate(...)` uses the symplectic BM4 composition. The method is fixed
to maintain a single, understandable numerical path. Its uniform BM4 grid
depends only on `t_span` and `step`; changing `n_output_samples` does not change
that trajectory. Samples between BM4 nodes are obtained with independent
shadow BM4 advances from the preceding node.

The usual arguments are:

- `t_span`: initial and final time.
- `step`: maximum internal step size.
- `n_output_samples`: number of saved samples, including the endpoints.
- `check_energy`: computes the generalized energy and its error.
- `progress`: shows the progress of long GC integrations.
- `stage_observer`: optional callback for instrumenting the twelve internal
  stages of each BM4 step without changing the numerical result; shadow
  output advances are excluded.

The solution provides at least:

- `solution.t`: saved times.
- `solution.y`: states, with one column per time.
- `solution.n_steps`: number of complete BM4 steps.
- `solution.k`: extended momentum when the energy is checked.
- `solution.err`: maximum energy error when requested.
- `solution.components()`: named physical blocks for the trajectory.

For example:

```python
components = solution.components()
x = components.x  # shape (particles, times)
y = components.y
```

The Hamiltonian can also be evaluated directly:

```python
energy = system.hamiltonian(solution.t, solution.y)
```

## Potential

`Potential.random(...)` generates the reproducible periodic potential used by
the development notebooks. The potential supports:

- evaluating the field and its derivatives;
- obtaining the gyroaverage required by GC;
- plotting the field with `potential.plot()`;
- animating it with `potential.animate(...)`.

## Development notebooks

- `test_generalized_energy_.ipynb`: convergence and conservation of generalized
  energy in GC, plus a short FC check.
- `test_dX_dY.ipynb`: area conservation of a contour transported in GC.
- `study_gc_symplecticity.ipynb`: Jacobians of `flow` and `adjoint_flow`,
  symplectic defect, and block persistence under `outputs/`.
- `study_gc_projected_symplecticity_area.ipynb`: accumulated symplecticity of
  the projected physical map, copy separation, and area evolution.
- `study_gc_area_and_projected_symplecticity.ipynb`: animated comparison of a
  16-point GC circle for three BM4 step sizes, tracking area, projected
  symplecticity, and internal separation through `4*pi`.

The diagnostics specific to these studies live in `research/symplecticity/`
and `research/projection/`. Outputs are organized as
`outputs/<notebook folder>/<notebook>/<date>/` and are not versioned.

The internal structure is summarized in
[`docs/architecture.md`](docs/architecture.md).
