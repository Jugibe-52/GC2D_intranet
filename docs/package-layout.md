# Source package layout

The Python implementation is stored directly below `src/` and divided by
responsibility. There is no umbrella import package: callers import each public
name from the package that owns it.

| Package | Responsibility | Representative public names |
| --- | --- | --- |
| `potential` | Periodic electrostatic field model | `Potential` |
| `dynamics` | Equations of motion and capability protocols | `GuidingCenterDynamics`, `FullCyclotronDynamics` |
| `initial_conditions` | Initial state layouts and geometry | `GCInitialConfiguration`, `FCInitialConfiguration`, `Area` |
| `simulation` | Problems, methods, formulations, nonlinear-solver selection, requests, and solutions | `InitialValueProblem`, `RK4`, `BM4Implicit1`, `BM4Implicit2`, `ImplicitABBA1`, `ImplicitABBA2`, `NonlinearSolver`, `SimulationRequest`, `Solution` |
| `diagnostics` | Optional observers and diagnostic persistence | `ImplicitABBAJacobianObserver`, ABBA/BM4 iteration observers, projection and symplecticity observers |
| `studies` | Reusable experiment assembly and summaries | `ImplicitTrajectoryComparisonConfig`, `ImplicitABBAIterationStudyConfig`, `BM4ImplicitIterationStudyConfig`, and `run_*_study` functions |
| `visualization` | Optional plots, animations, tables, and notebook display | `animate_implicit_method_trajectories`, `plot_implicit_trajectory_differences`, `plot_implicit_method_iterations`, `plot_potential` |

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

`BM4Implicit1` and `BM4Implicit2` apply the same two Hairer projection
formulations around one complete twelve-stage `BM4Composition` cycle. Their
Newton matrices use centered differences of the doubled BM4 map; projection is
performed once per complete cycle and is distinct from `ProjectedBM4Composition`,
which averages and re-embeds the copies after every internal stage.

All four implicit methods select either `newton` or `broyden` through the same
`nonlinear_solver` field. The Broyden path is shared across method families and
depends only on each formulation's residual evaluator. Reduced formulations
start from `4 I`; simultaneous formulations use the corresponding zero-step
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
