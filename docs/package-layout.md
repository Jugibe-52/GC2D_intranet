# Source package layout

The Python implementation is stored directly below `src/` and divided by
responsibility. There is no umbrella import package: callers import each public
name from the package that owns it.

| Package | Responsibility | Representative public names |
| --- | --- | --- |
| `potential` | Periodic electrostatic field model | `Potential` |
| `dynamics` | Equations of motion and capability protocols | `GuidingCenterDynamics`, `FullCyclotronDynamics` |
| `initial_conditions` | Initial state layouts and geometry | `GCInitialConfiguration`, `FCInitialConfiguration`, `Area` |
| `simulation` | Problems, methods, formulations, requests, and solutions | `InitialValueProblem`, `RK4`, `BM4Implicit1`, `BM4Implicit2`, `ImplicitABBA1`, `ImplicitABBA2`, `SimulationRequest`, `Solution` |
| `diagnostics` | Optional observers and diagnostic persistence | projection and symplecticity observers |
| `studies` | Reusable experiment assembly and summaries | configuration objects and `run_*_study` functions |
| `visualization` | Optional plots, animations, and notebook display | `plot_potential`, `animate_gc_area_solution` |

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

## Dependency direction

The core dependency flow is intentionally one-way:

```text
potential <- dynamics
initial_conditions <- simulation -> dynamics
              diagnostics -> simulation
visualization -> potential + initial_conditions + simulation
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
