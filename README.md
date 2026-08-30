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
`ImplicitABBA1` documentation currently provides separate
[dynamics](docs/models/implicit-abba-1/dynamics/gc2d-h5-potential-architecture.puml)
and
[simulation](docs/models/implicit-abba-1/simulation/implicit-abba-simulation-architecture.puml)
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
[HDF5 import contract](docs/models/implicit-abba-1/dynamics/gc2d-h5-import.md)
and its
[architecture diagram](docs/models/implicit-abba-1/dynamics/gc2d-h5-potential-architecture.puml)
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

The simulation API provides two implementations of Hairer's symmetric ABBA
projection:

- `ImplicitABBA1` solves the reduced multiplier equation with exact `2 x 2`
  Newton blocks;
- `ImplicitABBA2` solves the equivalent simultaneous output--multiplier system
  from equation (21), using exact `6 x 6` blocks.

They define the same exact projected physical map but expose distinct solver
diagnostics. Their derivations are documented in
[`ABBA_implicit_1`](docs/models/implicit-abba-1/ABBA_implicit_1.pdf) and
[`ABBA_implicit_2`](docs/tex/ABBA_implicit_2/ABBA_implicit_2.pdf).

`ABBA4Implicit1` is the fourth-order Yoshida triple jump of three complete
reduced `ImplicitABBA1` roots. Its signed durations are
`(gamma h, delta h, gamma h)`, so the middle solve runs backward; every
substep owns an independent projection multiplier and nonlinear solve.

`ABBA6` is Yoshida's symmetric sixth-order composition of seven complete
reduced `ImplicitABBA1` roots. Its palindromic real coefficients have unit
first moment and vanishing third and fifth moments; two mirrored substeps are
negative. One outer step therefore performs seven independent projected ABBA
solves while cancelling the degree-three and degree-five modified-flow errors.

`BM4Implicit1` and `BM4Implicit2` use the corresponding reduced and
simultaneous projections around one complete BM4 cycle. All six implicit
methods accept `nonlinear_solver="newton"` or `nonlinear_solver="broyden"`.
Newton retains the existing residual-Jacobian path: analytic particle blocks
for ABBA and centered differentiation of the complete BM4 map for BM4.
Broyden evaluates only the formulation's explicit residual after its initial
small-step Jacobian approximation, then applies the good rank-one secant
update. The generic construction is documented in
[`broyden_generic_method.tex`](docs/tex/broyden/broyden_generic_method.tex).

`run_projected_bm4_symplecticity_study` provides a separate finite-difference
audit for `ProjectedBM4Composition`, which applies arithmetic-mean projection
and diagonal re-embedding after every internal BM4 stage. It distinguishes the
local defect of each complete twelve-stage step from the accumulated physical
flow defect.

`MidpointBM4` provides the explicit uncoupled midpoint projection: it applies
the complete twelve-stage BM4 cycle and averages the two copies once per
complete step. `run_midpoint_bm4_symplecticity_study` observes those twelve
stages with exact guiding-center field Jacobians, propagates one `2 x 2`
physical tangent per independent trajectory (through an intermediate `4 x 2`
doubled tangent), and plots the arithmetic mean of their local and accumulated
symplecticity errors for each requested step size.

The matching independent-trajectory study API also covers `MidpointABBA`,
`ImplicitABBA1`, `ABBA4Implicit1`, and `BM4Implicit1`. It forms an exact
`2 x 2` local Jacobian for every trajectory and accepted step, composes the
physical flow tangents, and reports the arithmetic mean of the individual
symplecticity defects. The explicit ABBA map is differentiated as four analytic
shears. The implicit maps use the implicit-function theorem at the converged
projection root; `ABBA4Implicit1` composes its factors as `J3 @ J2 @ J1`, and
`BM4Implicit1` additionally composes the twelve exact coupled-BM4 stage
factors. Finite differences are used only in tests as an independent audit of
these observer formulas.

`run_implicit_generalized_energy_study` reconstructs the normalized
time-conjugate momentum `kappa = k / 2` for `ImplicitABBA1`,
`ABBA4Implicit1`, and `BM4Implicit1` from their accepted stage snapshots. It
reports the physical Hamiltonian `h(t, z)`, the autonomous extended quantity
`K = h + kappa`, signed and running-envelope errors, adjacent refinement
orders, and nonlinear-solver work without changing the physical trajectory.
The same accepted-step observer chain evaluates the numerical Jacobian of the
splitting map on `(u_x, u_y, v_x, v_y, t, k)` and reports the relative
`6 x 6` defect `||D Psi.T Omega_6 D Psi - Omega_6||_F / ||Omega_6||_F` plus
`|det(D Psi) - 1|`. This extended-space measurement explicitly excludes the
dimension-reducing diagonal projection. A second observer re-solves every
centered perturbation of the complete implicit step and tests the projected
physical map on `(x, y, t, kappa)` against its `4 x 4` form. The latter
includes the nonlinear projection and distinguishes physical area preservation
from the stronger mixed space-time/momentum symplecticity conditions.
The matching visualization separates the expected time variation of `h` from
its conjugate compensation and the residual drift of `K`.

The separate `ABBA_implicit2`, `ABBA4_implicit2`, and `BM4_implicit2`
methods duplicate the complete autonomous state `Z = (x, y, t, k)` rather
than only the physical state `z`. Their internal splitting therefore acts on
`(Z_1, Z_2) in R^8`, and the symmetric implicit projection solves for a
four-component multiplier so that both complete copies agree. The accepted
state lies in `R^4`, advances `t` and `k` directly, and uses
`K(Z) = h(t, z) + k`; no momentum reconstruction or factor of two is involved.
These names intentionally remain distinct from the pre-existing
`ImplicitABBA2` and `BM4Implicit2`, whose suffix denotes a simultaneous
nonlinear-solver formulation rather than full state duplication.

`run_fully_extended_implicit_study` audits the three fully duplicated methods
over a step refinement. It records generalized-energy error and nonlinear
work, constructs every unprojected `8 x 8` splitting Jacobian as an analytic
product of shear and binding factors, and forms the `4 x 4` projected tangent
with the implicit-function theorem. Newton uses the analytic residual matrix
`D_lambda R = G D Psi A + 2 I`; centered perturbations do not enter the solve.
The observer retains them as an independent audit of `D Psi`, `D R`, and
`D Phi`. The associated plots keep the analytic `R^8` and `R^4` form defects,
determinant checks, and centered-difference audit errors separate.

`run_implicit_abba_reversibility_study` complements those tangent diagnostics
by solving a genuine signed reverse ABBA step from every selected accepted
endpoint. Its `abba4_implicit_1` formulation composes the three forward
tangent factors as `J3 @ J2 @ J1` and independently repeats the complete
Yoshida composition with outer duration `-h`. It compares the resulting
tangents through
`J_minus @ J_plus - I` and evaluates the proposed forward and backward
increments `h * zdot + h**2 / 2 * J @ zdot`. Increment closure is reported as
`norm(Delta_plus + Delta_minus) / max(norm(Delta_plus), norm(Delta_minus))`.
The reverse tangent is never defined by
inverting the forward matrix, so the reported closure includes nonlinear-solve
tolerance and floating-point effects.

`ImplicitABBA1TangentTaylor` and `ABBA4Implicit1TangentTaylor` implement the
physical update
`z_next = z + h * f + h**2 / 2 * D(Psi_base) @ f`. The first differentiates
one converged reduced `ImplicitABBA1` root. The second solves all three signed
Yoshida factors and composes their exact physical tangents as `J3 @ J2 @ J1`.
Every base root and tangent is recalculated at the current state of the new
trajectory. `run_implicit_abba1_tangent_taylor_comparison` and
`run_abba4_implicit1_tangent_taylor_comparison` compare each new trajectory
with its original map on an aligned grid using minimum-image periodic
distances.

`ExplicitEuler` provides the classical forward map
`z_next = z + h * f(t, z)` on the same output-independent fixed grid. The
`run_tangent_taylor_euler_accuracy_study` refinement study compares it with
both tangent-Taylor methods on nested complete steps. One in-memory DOP853
trajectory supplies the reference, an independent Radau solve measures its
resolution floor, and every reported error uses periodic minimum-image
particle distances.

```python
from simulation import ImplicitABBA1

method = ImplicitABBA1(
    nonlinear_solver="broyden",
    newton_absolute_tolerance=1e-14,
    newton_relative_tolerance=1e-13,
    newton_max_iterations=40,
)
```

Solver-neutral diagnostics are available as `nonlinear_solver`,
`nonlinear_iterations`, `residual_evaluations`,
`nonlinear_residual_norms`, and `nonlinear_tolerances`. The former
`newton_iterations` and `newton_residual_norms` keys remain as compatibility
aliases.

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
The focused `run_abba4_implicit_1_accuracy_study` performs the same verified
reference comparison for fourth-order ABBA, while
`run_abba4_implicit_1_trajectory_symplecticity_study` analyzes its exact
ideal-root tangent product across a requested step refinement.
`run_abba6_accuracy_study` applies the same reference, periodic-error,
nonlinear-work, and runtime protocol to the seven-stage method and reports its
observed sixth-order refinement slope.
The focused `run_tangent_taylor_euler_accuracy_study` instead compares
classical Euler with both proposed tangent-Taylor updates and reports their
time-integrated RMS error, final and maximum errors, runtimes, and adjacent
observed convergence orders.

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
