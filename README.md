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

The [common integration architecture](docs/simulation/integration-architecture.md)
shows the complete six-phase lifecycle, shared by all 13 methods. Each model
also has its own detailed diagram in the [method catalog](docs/models/README.md).

`Solution` is an immutable computed trajectory. Its initial configuration is
available as `solution.source`, while diagnostics are attached as read-only
data. Shared physical dynamics are documented independently of the numerical
models:

- [GC2D HDF5 potential and guiding-center dynamics](docs/dynamics/gc2d-h5-import.md).

Numerical architecture is organized by model:

- ABBA: [family theory](docs/models/abba/tex/theory.pdf),
  [family simulation](docs/models/abba/simulation/abba-numerical-architecture.md),
  and focused views for
  [`ABBA2Midpoint`](docs/models/abba2-midpoint/simulation/abba2-midpoint-simulation-architecture.md),
  [`ABBA2Implicit`](docs/models/abba2-implicit/simulation/abba2-implicit-simulation-architecture.md),
  [`ABBA4Implicit`](docs/models/abba4-implicit/simulation/abba4-implicit-simulation-architecture.md),
  including its [exterior-projection derivation](docs/models/abba4-implicit/tex/exterior-projection.pdf),
  and [`ABBA6Implicit`](docs/models/abba6-implicit/simulation/abba6-implicit-simulation-architecture.md);
- `BM4Implicit`: [theory](docs/models/bm4-implicit/tex/theory.pdf),
  and [simulation architecture](docs/models/bm4-implicit/simulation/bm4-simulation-architecture.md);
- `BM4Midpoint`: [theory](docs/models/bm4-midpoint/tex/theory.pdf),
  and [simulation architecture](docs/models/bm4-midpoint/simulation/bm4-midpoint-simulation-architecture.md).
  It averages the two copies after each complete twelve-stage cycle, with
  spatial duplication and optional passive energy tracking;
- `ExplicitEuler`: [theory](docs/models/explicit-euler/tex/theory.pdf),
  and [simulation](docs/models/explicit-euler/simulation/explicit-euler-simulation-architecture.md);
- `GaussLegendre4`: [theory](docs/models/gauss-legendre4/tex/theory.pdf),
  and [simulation](docs/models/gauss-legendre4/simulation/gauss-legendre4-simulation-architecture.md);
- `HBVM42`: [theory](docs/models/hbvm42/tex/theory.pdf),
  and [simulation](docs/models/hbvm42/simulation/hbvm42-simulation-architecture.md); and
- `RK4`: [theory](docs/models/rk4/tex/theory.pdf),
  and [simulation](docs/models/rk4/simulation/rk4-simulation-architecture.md).

The [model documentation index](docs/models/README.md) links every canonical
theory source and PDF, including each focused ABBA model.

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
    characteristic_length=0.06,
    interpolation_order=3,
)
```

The primary-file defaults are `B=1.5`, `characteristic_length=0.06`, and
`indx=(0, 1)`, selecting the mean field and its dominant declared
positive-frequency mode. The loader maps that dominant frequency to one cycle
per normalized time unit, so its temporal period is `1`, and maps each
characteristic spatial length to `2*pi`.

Loaded and artificial fields are instances of the same `Potential` class. Both
use a real mean plus complex positive-frequency modes with runtime phase
`exp(+i*2*pi*f*t)`; HDF5-specific provenance is retained in
`potential.metadata`.

See the
[HDF5 import contract](docs/dynamics/gc2d-h5-import.md)
and its
[architecture diagram](docs/dynamics/gc2d-h5-potential-architecture.puml)
for the dataset schema, normalization, interpolation, gyroaveraging, and
guiding-center integration.

## Guiding-center example

```python
import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
    BM4Implicit,
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
    BM4Implicit(coupling_frequency=2.0),
    request,
)
```

See the BM4 model-specific
[theory](docs/models/bm4-implicit/tex/theory.pdf)
and [simulation architecture](docs/models/bm4-implicit/simulation/bm4-simulation-architecture.md)
for the twelve-stage composition, its single reduced Hairer projection around
each complete cycle, the physical-state contract, and nonlinear diagnostics.

Physical parameters belong to the dynamics object. Changing the initial
configuration therefore does not change the model. The effective gyroaveraged
potential is available as `problem.dynamics.effective_potential`.

## ABBA models and canonical configuration space

The public API contains exactly four ABBA numerical-method classes:

1. `ABBA2Midpoint`;
2. `ABBA2Implicit`;
3. `ABBA4Implicit`;
4. `ABBA6Implicit`.

State-space choices are parameters of those methods, not additional public
method classes. The three implicit methods share two nonlinear selector axes
and one constrained state/energy strategy:

| Axis | Canonical values | Meaning |
|---|---|---|
| `projection_placement` | `"around_complete_composition"` | ABBA4 uses one outer projection; the removed per-map selector raises an error. |
| `projection_formulation` | `"reduced_multiplier"`, `"simultaneous_state_multiplier"` | Chooses the nonlinear residual representation. |
| `nonlinear_solver` | `"newton"`, `"broyden"` | Chooses how that residual is solved. |
| `state_extension` | `"physical"` | Compatibility selector: only spatial coordinates are duplicated. |
| `track_energy` | `False`, `True` | Adds time and passive normalized energy momentum per particle. Available on all 13 methods. |

`ABBA2Implicit`, `ABBA4Implicit` and `ABBA6Implicit` each admit eight
configurations: two spatial residual formulations, two nonlinear solvers and
tracking off/on. `ABBA2Midpoint` has two tracking configurations, giving 26
configurations across the four classes. The deprecated
`ABBA4ImplicitSingleProjection` factory returns `ABBA4Implicit`.

The former `state_extension="fully_extended"` projected time and momentum as
well as space. It is rejected explicitly; use `track_energy=True` for the new
energy-monitoring contract. Historical full-state studies are not silently
reinterpreted as physical studies.

The exported tuples `ABBA4_PROJECTION_PLACEMENTS`,
`ABBA_PROJECTION_FORMULATIONS`, `NONLINEAR_SOLVERS`, and
`ABBA_STATE_EXTENSIONS` expose the canonical values programmatically.

```python
from simulation import ABBA4Implicit

method = ABBA4Implicit(
    projection_placement="around_complete_composition",
    projection_formulation="simultaneous_state_multiplier",
    nonlinear_solver="broyden",
    state_extension="physical",
    track_energy=True,
    newton_absolute_tolerance=1e-14,
    newton_relative_tolerance=1e-13,
    newton_max_iterations=40,
)
```

The formulation, solver, state and energy selections apply to the complete
step. ABBA4 always carries both copies continuously through three signed maps
inside one outer projection. Its legacy single-projection factory selects the
same algorithm as `ABBA4Implicit()`.

| Method | ABBA maps per outer step | Projection policy |
|---|---:|---|
| `ABBA2Midpoint` | 1 | Arithmetic mean; no nonlinear solve |
| `ABBA2Implicit` | 1 | One implicit symmetric projection |
| `ABBA4Implicit` | 3 | One outer projection around the complete unprojected triple jump |
| `ABBA6Implicit` | 7 | One implicit projection after each signed map |

The former per-map ABBA4 implementation is removed. Current ABBA4 has one
nonlinear solve per step and three base maps per residual evaluation.

### Four state formulations

| Formulation | One planar particle | N planar particles | Methods |
|---|---|---:|---|
| Physical | `(x,y)` in R2 | 2N | Classical and adaptive |
| Physical with energy | `(x,y,t,kappa)` in R4 | 4N | Classical and adaptive |
| Spatially duplicated | `(u,v)` in R4 | 4N | ABBA and BM4 |
| Duplicated with energy | `(u,v,t,kappa)` in R6 | 6N | ABBA and BM4 |

Every public method accepts `track_energy=True`, including `ExplicitEuler`,
`HBVM42` and `BM4Implicit`. The concrete `PhysicalFormulation` and
`DoubledFormulation` classes own the layouts. Accepted duplicated states store
equal spatial copies. `Solution.states` and observer maps expose physical
coordinates only. For FC classical methods the physical size is 4N, or 6N
with tracking. Every component block has N entries, so
`formulation.components(state)` has shape `(coordinates, N, *sample_axes)`.
The time block repeats the integration time for each particle; physical maps
continue to receive that scalar time. Clock validation and output alignment
belong to the formulation.

Hairer projects spatial copies only. Its reduced multiplier remains 2N, and
the ABBA simultaneous spatial unknown remains 6N regardless of tracking.
The energy clock and momentum never enter the nonlinear root or stopping scale.
The energy variant's R6 state is different from the simultaneous solver's R6
unknown for a single particle.

The auxiliary obeys `kappa'=-partial_t H`, starting from zero. Every family
reports the same normalization: `physical_kappa`. The histories
`extended_time`, `extended_momentum`, `physical_hamiltonian`,
`generalized_energy` and `generalized_energy_error` follow `Solution.t`.
All have shape `(N, saved_times)`, including `extended_time` for one particle.
Use `Solution.t` when a one-dimensional integration grid is needed.
`energy_error` is the maximum sampled absolute change in `H+kappa`.
Physical H need not be conserved in a time-dependent potential.

Tracking preserves physical trajectories, nonlinear work and adaptive grids.
DOP853/Radau integrate only physical coordinates in SciPy and evaluate passive
eight-point Gauss quadrature along accepted dense output. Their diagnostic
quadrature has no separate adaptive error tolerance and should be audited by
refinement. Radau Jacobians are always physical-sized. BM4Implicit replays the
converged spatial stages once for energy; that replay is not nonlinear work.

See the [integration contract](docs/simulation/integration-architecture.md)
and [model index](docs/models/README.md) for current guides and compiled theory.

`ABBA4Implicit` composes three unprojected ABBA maps with signed durations
`(gamma h, delta h, gamma h)` inside one symmetric projection. Its
[canonical theory](docs/models/abba4-implicit/tex/theory.pdf) includes the outer
residual, simultaneous formulation and complete ideal-root tangent. Earlier
per-map companions are retained as historical derivations.


Solver-neutral diagnostics include `projection_formulation`,
`state_extension`, `track_energy`, `nonlinear_solver`, `nonlinear_iterations`,
`residual_evaluations`, `nonlinear_residual_norms`, and
`nonlinear_tolerances`. They also record
`accepted_internal_state_dimension`, `base_splitting_state_dimension`, and
`nonlinear_unknown_dimension`, plus `observer_state_dimension` and
`observer_state_kind`. Energy-tracked physical runs expose
`extended_momentum`, `energy_error`, and the `kappa_equals_k_over_2`
normalization; their time coordinate is the ordinary `Solution.t`. Fully
extended runs expose direct `extended_momentum`, `extended_time`, the same
scalar `energy_error`, and detailed generalized-energy histories.

### Method instances, collection and adaptive methods

Every method inherits `IntegrationMethod.integrate`. Its `new_run` creates an instance of the same numerical class. Its `initialize`,
`advance`, `build_observation` and `export_history` own the method behavior.
`integrate_method` owns one run;
`IntegrationCollector` stores samples and accepted-step metrics. Method-specific
traces are consumed only when an observer needs them.

`DOP853` and `Radau` are public methods with `relative_tolerance`,
`absolute_tolerance`, `first_step`, `dense_output` and optional energy tracking.
They retain one live SciPy solver over the complete interval and share the same
collector and result boundary as all fixed methods. Accepted steps can vary;
requested samples use dense output. `AdaptiveIntegrationStep` exposes an accepted
interpolant instead of a fixed-step `map_state`.

```python
from simulation import DOP853, Radau, simulate

reference = simulate(problem, DOP853(relative_tolerance=1e-10,
                                     absolute_tolerance=1e-12), request)
audit = simulate(problem, Radau(relative_tolerance=1e-11,
                                absolute_tolerance=1e-13), request)
```

The common [architecture guide](docs/simulation/integration-architecture.md)
explains preparation, variable steps, observation domains and data ownership.
The [DOP853](docs/models/dop853/simulation/dop853-simulation-architecture.md) and
[Radau](docs/models/radau/simulation/radau-simulation-architecture.md) model guides
cover their numerical sessions and work counters. The study reference and
recurrence pipelines now use these public methods.

### Classical explicit methods

`ExplicitEuler` provides the first-order forward map
`z_next = z + h * f(t, z)`. `RK4` uses the four classical stages at
`t`, `t + h/2`, `t + h/2`, and `t + h`, and has designed global order four.
Both consume the general `DynamicalSystem` protocol and use the shared
output-independent fixed grid. RK4 can additionally advance the
time-conjugate momentum when `track_energy=True` and the dynamics implements
`HamiltonianSystem`. This extends `DynamicalSystem` with both `hamiltonian`
and `extended_momentum_derivative`; see the shared
[dynamics contracts](docs/dynamics/protocols.md).

See the model-specific documentation for
[`ExplicitEuler`](docs/models/explicit-euler/simulation/explicit-euler-simulation-architecture.md)
and [`RK4`](docs/models/rk4/simulation/rk4-simulation-architecture.md).

### Two-stage Gauss--Legendre method

`GaussLegendre4` is a symmetric, fourth-order implicit Runge--Kutta method.
For guiding-center dynamics it uses exact Hessian-derived field Jacobians and
solves one independent `4 x 4` coupled-stage Newton system per particle. Other
`DynamicalSystem` implementations use a dense centered-difference fallback.

```python
from simulation import GaussLegendre4

solution = simulate(
    problem,
    GaussLegendre4(
        track_energy=True,
        newton_absolute_tolerance=1e-14,
        newton_relative_tolerance=1e-13,
        newton_max_iterations=40,
        newton_jacobian_method="analytic",
    ),
    SimulationRequest.uniform(
        t_span=(0.0, 2.0),
        max_step=0.05,
        sample_count=41,
    ),
)
```

The observer event retains both converged collocation stages. For planar
guiding-center dynamics with exact particle Jacobians, diagnostics can therefore
calculate the exact ideal-root step tangent and audit the actual finite-tolerance
map separately. Energy tracking advances the direct conjugate
momentum `k` with the same Gauss nodes and reports drift of `K=H+k` without
changing the physical trajectory.

See the model-specific
[simulation derivation](docs/models/gauss-legendre4/simulation/gauss-legendre4-simulation-architecture.md).

## Full-cyclotron example

```python
from dynamics import FullCyclotronDynamics
from initial_conditions import FCInitialConfiguration
from simulation import InitialValueProblem, RK4, SimulationRequest, simulate

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
    RK4(),
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

`run_ten_method_trajectory_comparison` retains its historical public name and
advances `ABBA2Midpoint`, both implicit ABBA2 projection formulations with
Newton and Broyden, and `BM4Implicit` with Newton and Broyden. Every solution
shares one initial configuration and saved-time grid. The result provides all
pairwise periodic-distance summaries, runtimes for every variant, and aligned
nonlinear-work summaries for the implicit runs. The companion animation
presents sampled trajectories as points without connecting lines.

`run_high_precision_reference_trajectory` constructs a versioned numerical
reference. The local viewer
`notebooks/developements/accuracy/h5_three_radial_reference.ipynb` uses the
measured HDF5 field and three initial positions on one radius, constructed by
`studies.initial_conditions.radial_gc_configuration`. It targets normalized
time `T=200` with 20,001 saved states, initial coverage, and per-particle audit
discrepancies, and saves versioned artifacts for reuse. This development
notebook remains local and ignored by Git.

The reference runner accepts an optional `progress_callback(method, time,
nfev, status)` observing accepted steps without changing adaptive integration.
The viewer uses `diagnostics.reference_progress.reference_progress_log` to flush
progress to its cell output and a versioned `progress.log` every ten seconds
after an accepted step. The log identifies Radau (phase 1/2) and DOP853 (phase
2/2), accepted simulation time, percentage, elapsed wall time and approximate
ETA for the current solver. ETA excludes the other solver and persistence;
the final completion message is emitted after artifacts are written. Failures
and interruptions are logged. Logs are not resumable checkpoints.

The reference pipeline constructs a numerical
reference for the same interpolated guiding-center ODE with adaptive DOP853 and
audits its resolution independently with Radau. The stored NPZ/JSON/README
artifact includes exact initial data, complete solver controls, periodic audit
distances, checksums, and a fingerprint of the actual gyroaveraged interpolated
field. `run_ten_method_accuracy_study` verifies that artifact and reports
minimum-image trajectory errors for every retained fixed-step variant,
including particle-RMS error over time, final and worst-case distances,
resolution-floor ratios, and the measured accuracy--runtime trade-off.
`run_ten_method_accuracy_refinement_study` repeats those variants on nested
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

`run_gauss_legendre4_evaluation` combines a DOP853/Radau reference audit with
trajectory accuracy, robust runtime samples, generalized-energy drift, Newton
work, exact ideal-root symplecticity, sparse finite-stopping-rule map audits,
and resolved order-deficit detection backed by a tighter-Newton trajectory
audit. Its energy history is sampled at every complete step.
`run_gauss_bm4_comparison` applies the
same reference and alternated timing protocol to `GaussLegendre4` and
`BM4Implicit`, reporting both equal-step ratios and log--log interpolated
runtime ratios at equal trajectory accuracy.

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

Local and reusable AWS notebook execution is selected by one attribute; see
[`docs/cloud-notebook-execution.md`](docs/cloud-notebook-execution.md).

## Quality checks

Run the same checks used by continuous integration:

```bash
python -m mypy src
MPLBACKEND=Agg python -m unittest discover -s tests -v
python examples/gc_orbit.py
python examples/projected_abba.py
python -m build
```
