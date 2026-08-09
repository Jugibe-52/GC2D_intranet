# GC2D

GC2D simulates particle trajectories in a two-dimensional, time-dependent
electrostatic potential. Library code lives directly under `src/` in packages
grouped by responsibility:

- `potential`: periodic electrostatic fields;
- `dynamics`: guiding-center and full-cyclotron equations;
- `initial_conditions`: state layouts and initial geometry;
- `simulation`: numerical formulations, methods, requests, and results;
- `diagnostics`: opt-in numerical observers and persistence;
- `studies`: reusable experiment composition;
- `visualization`: optional Matplotlib presentation.

The simulation lifecycle keeps the physical model, initial state, numerical
algorithm, temporal request, and computed result separate:

```text
Potential -> Dynamics --\
                       +-> InitialValueProblem --\
InitialConfiguration -/                         \
                                                  -> SimulationRunner -> Solution
NumericalMethod ---------------------------------/
SimulationRequest -------------------------------/
```

`Solution` is an immutable computed trajectory. Its initial configuration is
available as `solution.source`, while diagnostics are attached as read-only
data. See the [package layout](docs/package-layout.md) and the
[detailed](docs/architecture.puml) and
[conceptual](docs/architecture-overview.puml) architecture diagrams.

## Installation

Python 3.11 or later is required. Install the reproducible notebook and
development environment with:

```bash
python -m pip install -r requirements.txt
```

For a core editable installation without plotting or notebook dependencies:

```bash
python -m pip install -e .
```

Runtime compatibility ranges are declared in `pyproject.toml`; tested direct
dependency versions are recorded in `constraints.txt`. Matplotlib support is
available through the `visualization` extra.

## Guiding-center example

```python
import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
    BM4Composition,
    GCExtendedFormulation,
    InitialValueProblem,
    SimulationRequest,
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
configuration = GCInitialConfiguration.from_components(
    x=np.asarray([np.pi]),
    y=np.asarray([np.pi]),
)
problem = InitialValueProblem(
    GuidingCenterDynamics(potential, rho=0.3),
    configuration,
)
request = SimulationRequest.uniform(
    t_span=(0.0, 6 * np.pi),
    max_step=0.001,
    sample_count=361,
)
solution = simulate(
    problem,
    BM4Composition(
        GCExtendedFormulation(coupling_frequency=2.0),
        track_energy=True,
    ),
    request,
)
```

Physical parameters belong to the dynamics object. Changing the initial
configuration therefore does not change the model. The effective gyroaveraged
potential is available as `problem.dynamics.effective_potential`.

## Full-cyclotron example

```python
from dynamics import FullCyclotronDynamics
from initial_conditions import FCInitialConfiguration
from simulation import BM4Composition, FCSplitFormulation

configuration = FCInitialConfiguration.from_components(
    x=np.asarray([np.pi]),
    y=np.asarray([np.pi]),
    vx=np.asarray([1.0]),
    vy=np.asarray([0.0]),
)
problem = InitialValueProblem(
    FullCyclotronDynamics(potential, rho=0.3, eta=0.01),
    configuration,
)
solution = simulate(
    problem,
    BM4Composition(FCSplitFormulation(), track_energy=True),
    SimulationRequest.uniform(
        t_span=(0.0, 2 * np.pi),
        max_step=0.001,
        sample_count=101,
    ),
)
```

## State organization

States use component-major blocks. For `N` particles:

```text
GC: [x_1, ..., x_N, y_1, ..., y_N]
FC: [x_1, ..., x_N, y_1, ..., y_N,
     vx_1, ..., vx_N, vy_1, ..., vy_N]
```

Named constructors hide this ordering. `as_blocks(...)` exposes a view with
shape `(components, particles, *sample_axes)`, while `from_blocks(...)`
performs the inverse transformation.

`Area` is a guiding-center initial configuration with square or circular
boundary geometry:

```python
from initial_conditions import Area

area = Area.circle(center=(np.pi, np.pi), radius=0.5, points=64)
problem = InitialValueProblem(
    GuidingCenterDynamics(potential, rho=0.3),
    area,
)
```

`Area.calculate_area(...)` applies the shoelace formula and can unwrap
periodic boundary crossings when a period is supplied.

## Reusable studies

The `studies` package assembles repeatable experiments while notebooks retain
their scientific parameters explicitly:

```python
from studies import (
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
area = centered_circle(potential, radius=0.5, points=16)
config = AreaComparisonConfig(
    steps=pi_area_steps(40, 80, 160),
    t_span=(0.0, 4 * np.pi),
    save_interval=np.pi / 8,
    rho=0.3,
    coupling_frequency=0.0,
)
result = run_area_comparison(
    potential,
    area,
    notebook_path="notebooks/experiments/area_study.ipynb",
    config=config,
    metadata=potential_config.metadata(),
)
```

These helpers validate grids, assemble simulations and observers, execute
repeated runs, persist diagnostics, and prepare summaries. Potential seeds,
initial geometry, physical and numerical parameters, integration spans, and
sampling choices remain visible in the calling notebook.

## Results and visualization

The result interface provides:

- `solution.t`: read-only saved times;
- `solution.states`: read-only physical states;
- `solution.source`: initial configuration and state layout;
- `solution.diagnostics`: read-only numerical diagnostics;
- `solution.components()` and `solution.positions()`: physical views.

Optional presentation is kept outside the physical model:

```python
from visualization import plot_potential

figure, axis = plot_potential(potential, t=0.0, show=False)
```

Area, particle, comparison, potential, animation, and notebook-display helpers
are exposed by `visualization`.

## Examples and notebooks

Fast supported scripts live in `examples/`. Git versions notebooks only in:

- `notebooks/experiments/`: reproducible scientific experiment artifacts;
- `notebooks/sympy/`: symbolic derivations supporting numerical methods.

`notebooks/developements/` contains local working notebooks and remains
ignored. Generated outputs live below
`outputs/<notebook folder>/<notebook>/<date>/` and are not versioned.

## Quality checks

Run the same checks used by continuous integration:

```bash
python -m mypy src
MPLBACKEND=Agg python -m unittest discover -s tests -v
python examples/gc_orbit.py
python examples/projected_abba.py
python -m build
```
