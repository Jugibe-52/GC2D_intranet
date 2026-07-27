# GC2D

Research code for studying particle trajectories in a two-dimensional
electrostatic potential.

The interoperable API separates physical dynamics, initial configurations,
numerical formulations, and methods:

```text
Dynamics + InitialConfiguration -> InitialValueProblem
         + NumericalMethod      -> SimulationRunner -> Solution
         + SimulationRequest
```

- `Potential` represents the physical field on a periodic grid.
- `GuidingCenterDynamics` and `FullCyclotronDynamics` define the equations.
- `TrajectoryGC` and `TrajectoryFC` provide compatible initial configurations.
- `Area` specializes a GC configuration as a square or circular contour.
- `InitialValueProblem` binds dynamics to one initial configuration.
- `NumericalMethod` implementations integrate that problem independently.

The notebooks in `notebooks/developements/` remain the primary examples. There
are no command-line executables or external configuration files.

## Installation

Python 3.11 or later is required.

```bash
python -m pip install -r requirements.txt
```

The package is installed in editable mode, so changes in `src/` are immediately
available to the notebooks.

## Notebook workflows

The `workflows` package keeps repeated initialization and experiment
orchestration out of notebooks without hiding reproducibility parameters. For
example, a centered area study starts with:

```python
from workflows import (
    AreaComparisonConfig,
    RandomPotentialConfig,
    centered_circle,
    pi_area_steps,
    run_area_comparison,
)

potential_config = RandomPotentialConfig(
    amplitude=0.7,
    max_wave_number=25,
    nx=64,
    ny=64,
    seed=27,
    interpolation_order=5,
)
potential = potential_config.build()
area = centered_circle(potential, radius=0.5, points=16, rho=0.3)
config = AreaComparisonConfig(
    steps=pi_area_steps(40, 80, 160),
    t_span=(0.0, 4 * np.pi),
    save_interval=np.pi / 8,
    coupling_frequency=0.0,
)
result = run_area_comparison(
    potential,
    area,
    notebook_path="notebooks/developements/area_study.ipynb",
    config=config,
    metadata=potential_config.metadata(),
)
```

Set `method_kind="stage_projected_bm4"` with `coupling_frequency=0.0` to run
the alternative that projects both GC copies after every direct or adjoint map.

The notebook remains responsible for declaring potential, geometry,
integration, and sampling parameters. Workflows construct centered initial
conditions, assemble problems, methods and observers, run repeated integrations, and
prepare summaries and visualizations.

## GC example

```python
import numpy as np

from classes import (
    BM4Composition,
    GCExtendedFormulation,
    GuidingCenterDynamics,
    InitialValueProblem,
    Potential,
    SimulationRequest,
    TrajectoryGC,
    simulate,
)

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

problem = InitialValueProblem(
    GuidingCenterDynamics(potential, rho=trajectory.rho),
    trajectory,
)
request = SimulationRequest.uniform(
    t_span=(0.0, 6 * np.pi),
    max_step=0.001,
    sample_count=361,
)
method = BM4Composition(
    GCExtendedFormulation(coupling_frequency=2.0),
    track_energy=True,
    progress=True,
)
solution = simulate(problem, method, request)
```

`GuidingCenterDynamics` evaluates guiding-center motion over its gyroaveraged
effective potential. The original physical potential remains in `potential`.
`coupling_frequency` controls the numerical coupling between the two internal
copies in `GCExtendedFormulation`; it does not represent a physical frequency.
Its default value is `pi / 8`.

## FC example

```python
import numpy as np

from classes import (
    BM4Composition,
    FCSplitFormulation,
    FullCyclotronDynamics,
    InitialValueProblem,
    Potential,
    SimulationRequest,
    TrajectoryFC,
    simulate,
)

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

problem = InitialValueProblem(
    FullCyclotronDynamics(
        potential,
        rho=trajectory.rho,
        eta=trajectory.eta,
    ),
    trajectory,
)
request = SimulationRequest.uniform(
    t_span=(0.0, 2 * np.pi),
    max_step=0.001,
    sample_count=101,
)
solution = simulate(
    problem,
    BM4Composition(FCSplitFormulation(), track_energy=True),
    request,
)
```

`FullCyclotronDynamics` describes the full cyclotron orbit over the physical
potential, while `FCSplitFormulation` supplies its exact split maps to BM4.

## Explicit numerical composition

New simulations can select a general method or a structure-preserving
formulation without changing the physical dynamics:

```python
import numpy as np

from classes import (
    BM4Composition,
    GCInitialConfiguration,
    GCExtendedFormulation,
    GCStageProjectedFormulation,
    GuidingCenterDynamics,
    InitialValueProblem,
    ProjectedBM4Composition,
    RK4,
    SimulationRequest,
    simulate,
)

configuration = GCInitialConfiguration.from_components(
    x=np.array([potential.grid.xmin + potential.grid.period / 2]),
    y=np.array([potential.grid.ymin + potential.grid.period / 2]),
    rho=0.3,
)
problem = InitialValueProblem(
    GuidingCenterDynamics(potential, rho=configuration.rho),
    configuration,
)
request = SimulationRequest.uniform(
    t_span=(0.0, 6 * np.pi),
    max_step=0.001,
    sample_count=361,
)

rk4_solution = simulate(problem, RK4(), request)
bm4_solution = simulate(
    problem,
    BM4Composition(
        GCExtendedFormulation(coupling_frequency=2.0),
        track_energy=True,
    ),
    request,
)
projected_solution = simulate(
    problem,
    ProjectedBM4Composition(
        GCStageProjectedFormulation(),
        track_energy=True,
    ),
    request,
)
```

`RK4` consumes any compatible vector field. `BM4Composition` consumes a
direct/adjoint formulation; the current choices are
`GCExtendedFormulation` and `FCSplitFormulation`.
`ProjectedBM4Composition` consumes `GCStageProjectedFormulation`, omits harmonic
copy coupling, and re-embeds the mean of both GC copies after every direct or
adjoint map, for twelve projections per complete BM4 step. This projection is
not invertible, so this variant does not inherit the doubled-space
symplecticity guarantee of the standard BM4 composition. The architecture-level names
`GCInitialConfiguration` and `FCInitialConfiguration` are available alongside
the compatible legacy names `TrajectoryGC` and `TrajectoryFC`.

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
to `Area`; common experiment composition belongs to `workflows`, while each
notebook keeps the parameters that define its scientific study explicit.

`TrajectoryGC.split(...)` returns components named `x` and `y`;
`TrajectoryFC.split(...)` adds `vx` and `vy`. The `pack_components(...)` class
method performs the inverse operation. This makes the physical state format the
trajectory's responsibility, so the integrator does not need to duplicate its
structure.

## Areas

`Area` is a GC initial configuration whose points delimit an oriented contour.
It can be built as a square or circle and integrated like any other GC
configuration:

```python
from classes import (
    Area,
    BM4Composition,
    GCExtendedFormulation,
    GuidingCenterDynamics,
    InitialValueProblem,
    SimulationRequest,
    simulate,
)
from workflows import animate_gc_area_solution

area = Area.square(
    center=(np.pi, np.pi),
    side=1.0,
    points_per_side=40,
    rho=0.3,
)
dynamics = GuidingCenterDynamics(potential, rho=area.rho)
solution = simulate(
    InitialValueProblem(dynamics, area),
    BM4Composition(GCExtendedFormulation()),
    SimulationRequest.uniform(max_step=0.005),
)

transported_area = area.calculate_area(
    solution.states,
    period=potential.grid.period,
)

animation = animate_gc_area_solution(
    dynamics.effective_potential,
    area,
    solution,
    frames=120,
    interval=50,
)
```

The alternative `Area.circle(...)` constructor takes `center`, `radius`, and the
total number of points on the contour. `calculate_area(...)` accepts both the
initial state and a complete time series and applies the shoelace formula; with
`period`, it also handles crossings of the periodic boundary correctly.
`animate_gc_area_solution(...)` displays the contour over the effective
potential and its electric field, together with
`(A(t) - A(0)) / abs(A(0))` in a second panel.
`animate_gc_area_comparison(...)` overlays several solutions saved at the
same times and uses a consistent color for each solution across the contour,
area error, and optional projected diagnostics.

## Integration and results

`SimulationRequest` owns `t_span`, `max_step`, and the output times. Its
`uniform(...)` constructor accepts `sample_count`, including both endpoints.
Changing the output samples does not alter the uniform main integration grid.
Off-grid outputs use independent shadow advances from the preceding node.

Method instances own numerical options. `track_energy=True` computes extended
momentum and generalized-energy drift; `progress=True` enables terminal
progress. BM4 methods also accept `stage_observer`, which instruments their
twelve internal stages without observing shadow output advances.

The canonical solution interface provides:

- `solution.t`: saved times.
- `solution.states`: physical states, with one column per time.
- `solution.source`: the initial configuration and state layout.
- `solution.diagnostics`: named method and formulation diagnostics.
- `solution.components()`: named physical blocks for the trajectory.
- `solution.positions()`: the two position histories.

For compatibility, `solution.y` aliases `states`, `solution.trajectory` aliases
`source`, and `solution.n_steps`, `solution.k`, and `solution.err` expose the
corresponding diagnostics. For example:

```python
components = solution.components()
x = components.x  # shape (particles, times)
y = components.y
```

The Hamiltonian can also be evaluated directly:

```python
energy = problem.dynamics.hamiltonian(solution.t, solution.states)
```

## Potential

`Potential.random(...)` generates the reproducible periodic potential used by
the development notebooks. The potential supports:

- evaluating the field and its derivatives;
- obtaining the gyroaverage required by GC;
- plotting the field with `potential.plot()`;
- animating it with `potential.animate(...)`.

## Notebooks

- `developements/gc_area_and_projected_symplecticity_1_dev.ipynb`: short or
  full validation of the uncoupled projected-area workflow.
- `experiments/gc_area_and_projected_symplecticity_1.ipynb`: three-step area
  comparison with zero numerical copy coupling.
- `experiments/gc_area_and_projected_symplecticity_2.ipynb`: the same
  comparison with coupling frequency 10.
- `experiments/generalized_energy.ipynb`: generalized-energy convergence over
  four successively refined BM4 steps.

The diagnostics specific to these studies live in `research/symplecticity/`
and `research/projection/`. Outputs are organized as
`outputs/<notebook folder>/<notebook>/<date>/` and are not versioned.

The implemented architecture is documented by the editable PlantUML diagram
[`docs/architecture.puml`](docs/architecture.puml).
