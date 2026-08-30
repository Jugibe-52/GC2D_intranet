# GC2D

GC2D simulates particle trajectories in a two-dimensional, time-dependent
electrostatic potential. Library code lives directly under `src/` in packages
grouped by responsibility:

- `potential`: GC2D HDF5 imports and periodic electrostatic fields;
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
data. Architecture documentation is organized by numerical model. The
`ABBA2Implicit` documentation currently provides separate
[dynamics](docs/models/abba2-implicit/dynamics/gc2d-h5-potential-architecture.puml)
and
[simulation](docs/models/abba2-implicit/simulation/abba2-implicit-simulation-architecture.puml)
diagrams.

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

## GC2D HDF5 potential

The primary GC2D field format stores a real mean potential and complex
positive-frequency modes in HDF5. Load it through the public potential API:

```python
from potential import load_gc2d_h5_potential

potential = load_gc2d_h5_potential(
    "data/potential/V1/PHI_2.h5",
    interpolation_order=3,
)
```

The primary-file defaults are `B=1.5` and `indx=(0, 1)`, selecting the mean
field and its dominant declared positive-frequency mode.

See the
[HDF5 import contract](docs/models/abba2-implicit/dynamics/gc2d-h5-import.md)
and its
[architecture diagram](docs/models/abba2-implicit/dynamics/gc2d-h5-potential-architecture.puml)
for the dataset schema, normalization, interpolation, gyroaveraging, and
guiding-center integration.

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

## Implicit projection formulations and nonlinear solvers

`ABBA2Implicit` is the single public second-order implementation of Hairer's
symmetric projection. Its `projection_formulation` field chooses how the same
accepted physical map is solved:

| `projection_formulation` | Nonlinear unknown for one particle | Exact Newton block |
|---|---|---:|
| `"reduced_multiplier"` | Projection multiplier `mu` | `2 x 2` |
| `"simultaneous_state_multiplier"` | Final duplicated state and multiplier `(u_f, v_f, mu)` | `6 x 6` |

The implementation keeps those branches in separate private modules:
`_projection_reduced.py` owns formulation 1, `_projection_simultaneous.py` owns
formulation 2, and `_projection_common.py` contains only their shared stage and
tangent records. The public selector remains unified because the two branches
solve the same projected map.

The `6 x 6` simultaneous system is a nonlinear-solver workspace. It is **not**
an `R^6` accepted state and it does not define another numerical method. Both
formulations return the same physical state up to the nonlinear stopping
tolerance. The reduced derivation is documented in
[`ABBA2_implicit`](docs/models/abba2-implicit/ABBA2_implicit.pdf); the
simultaneous algebra remains in the historical
[simultaneous-formulation note](docs/tex/ABBA_implicit_2/ABBA_implicit_2.pdf).

```python
from simulation import ABBA2Implicit

method = ABBA2Implicit(
    projection_formulation="simultaneous_state_multiplier",
    nonlinear_solver="broyden",
    newton_absolute_tolerance=1e-14,
    newton_relative_tolerance=1e-13,
    newton_max_iterations=40,
)
```

`ABBA2Midpoint` reuses the same endpoint-time A-B-B-A stage map but replaces
the nonlinear symmetric projection with an arithmetic mean. The implicit
composition family uses the reduced multiplier root as its base step:

- `ABBA4Implicit` is Yoshida's fourth-order triple jump with signed durations
  `(gamma h, delta h, gamma h)`;
- `ABBA4ImplicitSingleProjection` applies one reduced projection around a
  complete unprojected fourth-order triple jump; and
- `ABBA6Implicit` is the symmetric seven-stage sixth-order composition, with
  two mirrored negative substeps.

All projected implicit ABBA methods accept `nonlinear_solver="newton"` or
`nonlinear_solver="broyden"`. Newton uses exact independent particle blocks.
Broyden evaluates the residual and applies a good rank-one secant update. The
generic solver construction is documented in
[`broyden_generic_method.tex`](docs/tex/broyden/broyden_generic_method.tex).

### Time extensions and their state spaces

The two extension families are deliberately separate because they lift a
different state and, in the fully extended case, define a different projected
map:

| Method | Accepted internal state | Base splitting state | Momentum convention | Physical map |
|---|---|---|---|---|
| `ABBA2SharedTimeExtendedImplicit` | `(z,t,kappa) in R^4` | `(u,v,t,k) in R^6` | `kappa = k/2` | Identical to `ABBA2Implicit` for the same formulation |
| `ABBA2FullyExtendedImplicit` | `(z,t,k) in R^4` | `(Z_1,Z_2) in R^8` | Direct `k` | Full-state projected ABBA2 map |
| `ABBA4FullyExtendedImplicit` | `(z,t,k) in R^4` | `(Z_1,Z_2) in R^8` | Direct `k` | Fourth-order composition of full-state ABBA maps |

Here `z=(x,y)` and `Z=(z,t,k)`. The shared-time method duplicates only the
physical state and shares one time--momentum pair, so its splitting state is
genuinely in `R^6`. The fully extended methods duplicate the complete
autonomous state and therefore operate on `R^8`. These extension classes
currently require exactly one guiding-centre particle.

This distinction also prevents a common ambiguity: the `R^6` simultaneous
Newton unknown of `ABBA2Implicit` is an algebraic solve vector, whereas the
`R^6` value in `ABBA2SharedTimeExtendedImplicit` is the state of the lifted
splitting map.

`run_implicit_generalized_energy_study` can reconstruct the normalized
conjugate momentum `kappa=k/2` from accepted projected-ABBA stages without
changing the physical trajectory. `run_fully_extended_implicit_study` instead
audits the analytic `R^8` splitting tangent and the projected `R^4` tangent of
the fully duplicated methods.

The independent-trajectory diagnostics cover `ABBA2Midpoint`,
`ABBA2Implicit`, `ABBA4Implicit`, and the BM4 family. They form exact local
particle Jacobians, compose accumulated flow tangents, and use finite
differences only as independent verification. The reversibility study performs
a genuine signed reverse step rather than defining the reverse tangent as the
inverse of the forward matrix.

Solver-neutral diagnostics include `projection_formulation`,
`state_extension`, `nonlinear_solver`, `nonlinear_iterations`,
`residual_evaluations`, `nonlinear_residual_norms`, and
`nonlinear_tolerances`. Shared-time runs additionally expose `extended_time`,
`extended_kappa`, and the `kappa_equals_k_over_2` normalization. Fully extended
runs report the `fully_extended` state extension and direct-`k` normalization.

`ExplicitEuler` provides the classical forward map
`z_next = z + h * f(t, z)` on the same output-independent fixed grid.

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

`run_ten_method_trajectory_comparison` advances two explicit midpoint methods
and all four implicit ABBA/BM4 formulations with both Newton and Broyden. Its
ten solutions share one initial configuration and saved-time grid. The result
provides all 45 pairwise periodic-distance summaries, runtimes for every
variant, and aligned nonlinear-work summaries for the eight implicit runs.
The companion animation presents sampled trajectories as points without
connecting lines.

`run_high_precision_reference_trajectory` constructs a versioned numerical
reference for the same interpolated guiding-center ODE with adaptive DOP853 and
audits its resolution independently with Radau. The stored NPZ/JSON/README
artifact includes exact initial data, complete solver controls, periodic audit
distances, checksums, and a fingerprint of the actual gyroaveraged interpolated
field. `run_ten_method_accuracy_study` verifies that artifact and reports
minimum-image trajectory errors for all ten fixed-step variants, including
particle-RMS error over time, final and worst-case distances, resolution-floor
ratios, and the measured accuracy--runtime trade-off.
`run_ten_method_accuracy_refinement_study` repeats all ten variants on nested
complete steps while saving one common set of genuine main-grid nodes. It
reports the error-reduction factor and observed order between every adjacent
refinement without introducing shadow steps or trajectory interpolation.
The focused `run_abba4_implicit_accuracy_study` performs the same verified
reference comparison for fourth-order ABBA, while
`run_abba4_implicit_trajectory_symplecticity_study` analyzes its exact
ideal-root tangent product across a requested step refinement.
`run_abba6_accuracy_study` applies the same reference, periodic-error,
nonlinear-work, and runtime protocol to the seven-stage method and reports its
observed sixth-order refinement slope.
The focused `run_implicit_generalized_energy_study` runs any one of the three
projected implicit GC methods over an ordered step refinement. Energy is
recorded at every accepted main-grid node even when the physical solution uses
a coarser saved-time grid; output-only shadow steps never enter the energy
history. It also records the time-extended splitting symplecticity defect at
every accepted step and the complete projected-map defect on the reduced
four-dimensional extended space.

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
