# Source package layout

The Python implementation is stored directly below `src/` and divided by
responsibility. There is no umbrella import package: callers import each public
name from the package that owns it.

| Package | Responsibility | Representative public names |
| --- | --- | --- |
| `potential` | Periodic electrostatic field model | `Potential` |
| `dynamics` | Equations of motion and capability protocols | `GuidingCenterDynamics`, `FullCyclotronDynamics` |
| `initial_conditions` | Initial state layouts and geometry | `GCInitialConfiguration`, `FCInitialConfiguration`, `Area` |
| `simulation` | Problems, methods, formulations, nonlinear-solver selection, requests, and solutions | `InitialValueProblem`, `RK4`, `MidpointBM4`, `BM4Implicit1`, `BM4Implicit2`, `ImplicitABBA1`, `ImplicitABBA2`, `ABBA4Implicit1`, `NonlinearSolver`, `SimulationRequest`, `Solution` |
| `diagnostics` | Optional observers and diagnostic persistence | `StoredReferenceTrajectory`, `GCTrajectorySymplecticityObserver`, `MidpointBM4SymplecticityObserver`, `ImplicitABBAJacobianObserver`, ABBA/BM4 iteration observers, projection and symplecticity observers |
| `studies` | Reusable experiment assembly and summaries | `HighPrecisionReferenceConfig`, `TenMethodAccuracyResult`, `TenMethodTrajectoryComparisonConfig`, symplecticity configurations, and `run_*_study` functions |
| `visualization` | Optional plots, animations, tables, and notebook display | `plot_reference_trajectory_points`, `plot_trajectory_accuracy_over_time`, `plot_accuracy_summary`, `plot_accuracy_runtime_tradeoff`, `animate_trajectory_points`, symplecticity plots, and `plot_potential` |

## Import pattern

```python
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import InitialValueProblem, RK4, SimulationRequest, simulate

potential = Potential.random(A=0.7, M=25, nx=64, ny=64, seed=27)
configuration = GCInitialConfiguration.from_components(x=[1.0], y=[2.0])
problem = InitialValueProblem(
    GuidingCenterDynamics(potential, rho=0.3),
    configuration,
)
solution = simulate(
    problem,
    RK4(),
    SimulationRequest.uniform(
        t_span=(0.0, 1.0),
        max_step=0.01,
        sample_count=101,
    ),
)
```

Imports from implementation submodules are reserved for functionality that is
intentionally specialized. For example, symbolic or diagnostic code may use a
specific method module when it needs an internal map that is not part of the
general simulation API.

`ImplicitABBA1` and `ImplicitABBA2` expose the two nonlinear formulations of
Hairer's symmetric ABBA projection. The first solves the reduced multiplier
equation with independent `2 x 2` Newton blocks; the second implements the
equivalent simultaneous equation (21) with independent `6 x 6` blocks.

`ABBA4Implicit1` is the public fourth-order Yoshida composition of three
complete reduced `ImplicitABBA1` roots. One outer step uses the signed
durations `(gamma h, delta h, gamma h)`, with
`gamma = 1 / (2 - 2**(1/3))` and `delta = -2**(1/3) * gamma`; the middle
substep therefore runs backward. Each substep performs a separate nonlinear
solve and owns a separate projection multiplier. Its complete-step observation
contains the three accepted `ImplicitABBAIntegrationStep` values, and exact
diagnostics compose their physical tangents in flow order as `J3 @ J2 @ J1`.

`BM4Implicit1` and `BM4Implicit2` apply the same two Hairer projection
formulations around one complete twelve-stage `BM4Composition` cycle. Their
Newton matrices use centered differences of the doubled BM4 map; projection is
performed once per complete cycle and is distinct from `ProjectedBM4Composition`,
which averages and re-embeds the copies after every internal stage.

`run_projected_bm4_symplecticity_study` analyzes that stage-projected method.
Its observer differentiates each projected internal stage, composes the twelve
stage tangents into one local physical-step Jacobian, and separately propagates
the accumulated physical-flow Jacobian. The result reports symplecticity and
determinant defects, transported-area error, copy separation, and refinement
slopes without treating area preservation alone as a symplecticity proof.

`MidpointBM4` is the uncoupled explicit midpoint-projection variant. It starts
each complete step on the doubled diagonal, executes all twelve BM4 stages,
and averages the copies exactly once after that cycle. Its dedicated observer
uses exact guiding-center field Jacobians to compose independent `4 x 4` stage
factors into local and accumulated `2 x 2` physical tangents. The reusable
study reports the arithmetic mean of the individual trajectory defects; it
does not substitute the packed-system Frobenius defect, which is an RMS.

`GCTrajectorySymplecticityObserver` provides the same independent-trajectory
measurement protocol for `MidpointABBA`, `ImplicitABBA1`, `ABBA4Implicit1`, and
`BM4Implicit1`. It differentiates the explicit ABBA shears directly, applies the
implicit-function theorem to the accepted implicit projection, and, for BM4,
first composes the twelve exact coupled extended-stage factors. Every local
physical `2 x 2` tangent is then accumulated in time before the observer takes
the arithmetic mean of the per-trajectory relative defects. Its reusable
study runners share one initial configuration and saved-time grid across all
requested step sizes; trajectory plots use sampled markers without connecting
lines.

All public implicit ABBA/BM4 methods select either `newton` or `broyden`
through the same `nonlinear_solver` field. The Broyden path is shared across
method families and depends only on each formulation's residual evaluator.
Reduced formulations start from `4 I`; simultaneous formulations use the
corresponding zero-step
output--multiplier Jacobian. Per-step diagnostics distinguish nonlinear
corrections from explicit residual evaluations.

`ImplicitABBAJacobianObserver` evaluates the local physical tangent
`D Psi_{h,t_n}(z_n)` from converged ABBA stage data. Its spectral analysis is
separate from the area and symplecticity observers: it records the matrix,
trace, determinant, discriminant, eigensystem, and SVD for each independent
particle without accumulating the tangent flow.

`ImplicitABBAIterationObserver` consumes solver metrics emitted by each
accepted implicit ABBA step. It records how many nonlinear corrections were
required and the final residual relative to the effective stopping tolerance;
it does not reevaluate the step or its nonlinear equations.

`ImplicitBM4IterationObserver` records the equivalent metrics around each
complete projected BM4 cycle. Its records share the generic
`ImplicitIterationRecord` schema, allowing the same plots and persistence
format to present either method family.

`run_implicit_trajectory_comparison` advances the same initial configuration
with the four implicit methods, one common step, and one nonlinear solver. Its
result provides six pairwise periodic trajectory summaries and four aligned
iteration summaries. `animate_implicit_method_trajectories` assigns one color
to each method while displaying every particle trajectory.
`plot_implicit_trajectory_differences` presents the four methods on both axes
of a symmetric matrix and annotates every cell with the mean periodic particle
distance over all trajectories and saved states.

`run_ten_method_trajectory_comparison` extends the aligned comparison to
`MidpointABBA`, `MidpointBM4`, and Newton/Broyden instances of both implicit
ABBA and BM4 formulations. It exposes 45 pairwise periodic-distance summaries,
ten wall-clock runtimes, and eight nonlinear-work summaries. The matching
visualization uses a grouped `10 x 10` logarithmic distance matrix, a runtime
chart, aligned implicit-work plots, and an animation of sampled trajectory
points without connected paths.

`run_high_precision_reference_trajectory` integrates the same interpolated GC
ODE with DOP853 and an independent Radau audit on a prescribed output grid. It
persists a stable `example_trajectory/vN` artifact as checksummed NPZ arrays, a
JSON manifest, and a standalone explanation. The manifest records both solver
controls, exact initial-condition reconstruction data, the periodic audit
floor, and a fingerprint formed from the gyroaveraged sampled field and
canonical off-grid electric-field probes. `run_ten_method_accuracy_study`
validates the exact initial state, time grid, physical parameter, grid,
metadata, and dynamics fingerprint before measuring minimum-image errors for
the ten variants. The refinement runner evaluates nested steps on one saved
cadence that is an integer multiple of every complete step, then reports error
gains and observed time-integrated/final RMS orders. Its visualization shows
error evolution, log-log accuracy against step, global/final RMS accuracy, and
the runtime trade-off; reference paths use markers without connected lines.

`run_abba4_implicit_1_accuracy_study` provides a focused refinement of the
fourth-order composition against the same versioned reference, including
periodic RMS errors, observed orders, nonlinear work, runtime, and the margin
above the reference audit floor.
`run_abba4_implicit_1_trajectory_symplecticity_study` uses the exact ideal-root
product `J3 @ J2 @ J1` to report local and accumulated per-trajectory defects
for the same initial configuration across step sizes.

## Dependency direction

The core dependency flow is intentionally one-way:

```text
potential <- dynamics
initial_conditions <- simulation -> dynamics
              diagnostics -> simulation
visualization -> potential + initial_conditions + simulation + diagnostics
studies -> potential + dynamics + initial_conditions + simulation + diagnostics
```

The core simulation packages do not import notebook or Matplotlib helpers.
This keeps a minimal installation usable with NumPy and SciPy alone.

## Type information and packaging

Each top-level package carries a `py.typed` marker. Package discovery, static
checking, tests, examples, and distribution builds all operate on the same
flat `src/` layout:

```bash
python -m mypy src
python -m unittest discover -s tests -v
python -m build
```
