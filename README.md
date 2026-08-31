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
data. Architecture documentation is organized by numerical model, with
separate dynamics contracts and simulation explanations:

- ABBA: [dynamics](docs/models/abba2-implicit/dynamics/gc2d-h5-import.md)
  and [simulation](docs/models/abba/simulation/abba-numerical-architecture.md);
- BM4: [dynamics and formulation contract](docs/models/bm4/dynamics/direct-adjoint-formulation-contract.md)
  and [simulation architecture](docs/models/bm4/simulation/bm4-simulation-architecture.md);
- `GaussLegendre4`: [dynamics](docs/models/gauss-legendre4/dynamics/guiding-center-contract.md)
  and [simulation](docs/models/gauss-legendre4/simulation/gauss-legendre4-simulation-architecture.md); and
- `HBVM42`: [dynamics](docs/models/hbvm42/dynamics/canonical-hamiltonian-contract.md)
  and [simulation](docs/models/hbvm42/simulation/hbvm42-simulation-architecture.md).

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

See the BM4 model-specific
[dynamics and formulation contract](docs/models/bm4/dynamics/direct-adjoint-formulation-contract.md)
and [simulation architecture](docs/models/bm4/simulation/bm4-simulation-architecture.md)
for the twelve-stage composition, projection variants, nonlinear solves,
diagnostics, and supported state extensions.

Physical parameters belong to the dynamics object. Changing the initial
configuration therefore does not change the model. The effective gyroaveraged
potential is available as `problem.dynamics.effective_potential`.

## ABBA models and canonical configuration space

The public API contains exactly five ABBA numerical-method classes:

1. `ABBA2Midpoint`;
2. `ABBA2Implicit`;
3. `ABBA4Implicit`;
4. `ABBA4ImplicitSingleProjection`; and
5. `ABBA6Implicit`.

State-space choices are parameters of those methods, not additional public
method classes. The four implicit methods share three independent
configuration axes:

| Axis | Canonical values | Meaning |
|---|---|---|
| `projection_formulation` | `"reduced_multiplier"`, `"simultaneous_state_multiplier"` | Chooses the nonlinear residual representation. |
| `nonlinear_solver` | `"newton"`, `"broyden"` | Chooses how that residual is solved. |
| `state_extension` | `"physical"`, `"shared_time"`, `"fully_extended"` | Chooses which state is duplicated by the ABBA splitting. |

Consequently, each implicit class admits

```text
2 projection formulations x 2 nonlinear solvers x 3 state extensions = 12
```

configurations. `ABBA2Midpoint` has no nonlinear residual and therefore accepts
only the three `state_extension` values. The complete public family contains
`4 x 12 + 3 = 51` valid configurations while retaining five method classes.
The exported tuples `ABBA_PROJECTION_FORMULATIONS`, `NONLINEAR_SOLVERS`, and
`ABBA_STATE_EXTENSIONS` expose the canonical values programmatically.

```python
from simulation import ABBA4Implicit

method = ABBA4Implicit(
    projection_formulation="simultaneous_state_multiplier",
    nonlinear_solver="broyden",
    state_extension="fully_extended",
    newton_absolute_tolerance=1e-14,
    newton_relative_tolerance=1e-13,
    newton_max_iterations=40,
)
```

The projection and solver selections are global for a composed step. Thus all
three signed substeps of `ABBA4Implicit`, and all seven signed substeps of
`ABBA6Implicit`, use the same selected formulation, solver, and state
extension. They do not make independent per-substep choices.

The five methods differ in their base composition and projection placement:

| Method | ABBA maps per outer step | Projection policy |
|---|---:|---|
| `ABBA2Midpoint` | 1 | Arithmetic mean; no nonlinear solve |
| `ABBA2Implicit` | 1 | One implicit symmetric projection |
| `ABBA4Implicit` | 3 | One implicit projection after each signed map |
| `ABBA4ImplicitSingleProjection` | 3 | One implicit projection around the complete unprojected triple jump |
| `ABBA6Implicit` | 7 | One implicit projection after each signed map |

This makes `ABBA4ImplicitSingleProjection` a different numerical map, not an
alias for an `ABBA4Implicit` parameter choice. Its formulation and solver are
selected once for its single outer projection.

### Residual and state dimensions

For one guiding-centre particle, the complete dimensional convention is:

| `state_extension` | Accepted internal state | Base splitting state | Reduced unknown | Simultaneous unknown |
|---|---|---|---|---|
| `"physical"` | `z in R^2` | `(u,v) in R^4` | `mu in R^2` | `(u_f,v_f,mu) in R^6` |
| `"shared_time"` | `(z,t,kappa) in R^4` | `(u,v,t,k) in R^6` | `mu in R^2` | `(u_f,v_f,mu) in R^6` |
| `"fully_extended"` | `Z=(z,t,k) in R^4` | `(Z_1,Z_2) in R^8` | `mu in R^4` | `(Z_1f,Z_2f,mu) in R^12` |

These literal `R^2/R^4/R^6/R^8/R^12` entries are the one-particle dimensions.
For `physical` with `N` independent particles, the four columns scale to
`2N`, `4N`, `2N`, and `6N`; `shared_time` and `fully_extended` currently
require `N=1`.

The simultaneous unknown is a nonlinear-solver workspace, not an accepted
trajectory state. In particular, the physical formulation's temporary `R^6`
solve vector is unrelated to the genuine `R^6` splitting state used by
`state_extension="shared_time"`.

The shared-time strategy duplicates only `z`, shares one `(t,k)` pair, and
stores the accepted momentum as `kappa=k/2`. Its triangular momentum update
does not feed back into `z`, so it preserves the physical trajectory of the
corresponding `state_extension="physical"` configuration. The fully extended
strategy duplicates the complete autonomous state `Z`, advances direct `k`,
and can define a different physical map. Both non-physical extensions currently
require exactly one guiding-centre particle.

The accepted internal dimension is not always the dimension seen by a step
observer. Physical and shared-time configurations expose the closed physical
map `z -> z_next`, so their observer states are in `R^2`; for shared time,
`extended_time` and `extended_kappa` remain in diagnostics. Fully extended
configurations expose the accepted internal map `Z -> Z_next`, so their
observer states are in `R^4`. This distinction applies to midpoint and implicit
methods alike.

`ABBA2Implicit`'s two residual formulations define the same exact projected
map at convergence. The reduced branch lives in `_projection_reduced.py`; the
simultaneous branch lives in `_projection_simultaneous.py`; and their shared
physical stage records live in `_projection_common.py`. Fully extended
counterparts operate on the `R^8` base map in `methods/_fully_extended.py`.
Newton uses exact independent-particle blocks, whereas Broyden applies a good
rank-one secant update to the selected residual. The derivations are documented
in [`ABBA2_implicit`](docs/models/abba2-implicit/ABBA2_implicit.pdf), the
[simultaneous-formulation note](docs/tex/ABBA_implicit_2/ABBA_implicit_2.pdf),
and [`broyden_generic_method.tex`](docs/tex/broyden/broyden_generic_method.tex).

Solver-neutral diagnostics include `projection_formulation`,
`state_extension`, `nonlinear_solver`, `nonlinear_iterations`,
`residual_evaluations`, `nonlinear_residual_norms`, and
`nonlinear_tolerances`. They also record
`accepted_internal_state_dimension`, `base_splitting_state_dimension`, and
`nonlinear_unknown_dimension`, plus `observer_state_dimension` and
`observer_state_kind`. Shared-time runs expose `extended_time`,
`extended_kappa`, and the `kappa_equals_k_over_2` normalization. Fully extended
runs expose direct `extended_momentum` and generalized-energy diagnostics.

`ExplicitEuler` provides the classical forward map
`z_next = z + h * f(t, z)` on the same output-independent fixed grid.

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

See the model-specific [dynamics contract](docs/models/gauss-legendre4/dynamics/guiding-center-contract.md)
and [simulation derivation](docs/models/gauss-legendre4/simulation/gauss-legendre4-simulation-architecture.md).

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

`run_gauss_legendre4_evaluation` combines a DOP853/Radau reference audit with
trajectory accuracy, robust runtime samples, generalized-energy drift, Newton
work, exact ideal-root symplecticity, sparse finite-stopping-rule map audits,
and resolved order-deficit detection backed by a tighter-Newton trajectory
audit. Its energy history is sampled at every complete step.
`run_gauss_bm4_comparison` applies the
same reference and alternated timing protocol to `GaussLegendre4` and
`BM4Implicit1`, reporting both equal-step ratios and log--log interpolated
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

## Quality checks

Run the same checks used by continuous integration:

```bash
python -m mypy src
MPLBACKEND=Agg python -m unittest discover -s tests -v
python examples/gc_orbit.py
python examples/projected_abba.py
python -m build
```
