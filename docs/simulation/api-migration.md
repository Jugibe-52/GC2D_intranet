# Retired API migration

New code uses one execution function and one public class for each numerical
method. The following interfaces have been removed, including their exports:

| Removed interface | Current interface |
|---|---|
| `SimulationRunner().simulate(...)` | `simulate(...)` |
| `ABBA4ImplicitSingleProjection(...)` | `ABBA4Implicit(...)` |
| `ABBA4ImplicitSingleProjectionIntegrationStep` | `ABBA4ImplicitIntegrationStep`, describing one outer projection |
| `abba4_implicit_single_projection_step_particle_jacobians` | `abba4_implicit_step_particle_jacobians` |
| `Trajectory`, `TrajectoryGC`, `TrajectoryFC` | `StateConfiguration`, `GCInitialConfiguration`, `FCInitialConfiguration` |
| `centered_gc_trajectory` | `centered_gc_configuration` |
| Physical `rho`/`eta` metadata on initial configurations and `Area` | Explicit dynamics parameters or study configuration |
| `diagnostics.symplecticity.jacobians` and `.paths` | `diagnostics.jacobians` and `diagnostics.paths` |

The unused BM4 map/Jacobian wrappers, full-state projection implementations,
full-state energy/symplecticity observers, and retired full-state and per-factor
projection comparison studies have been removed. Tests of physical-field and
passive-energy derivatives now exercise the active dynamics directly.

```python
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from simulation import ABBA4Implicit, InitialValueProblem, SimulationRequest, simulate

configuration = GCInitialConfiguration.from_components(x=x0, y=y0)
problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=0.3), configuration)
request = SimulationRequest.uniform(t_span=(0.0, 1.0), max_step=0.01, sample_count=101)
solution = simulate(problem, ABBA4Implicit(track_energy=True), request)
```

The current method projects only the two spatial copies. Energy tracking is
passive; it does not reproduce the retired projection of time and momentum.
ABBA4 observations now always describe the three continuous unprojected factors
and one outer multiplier. ABBA6 retains its seven separate projected pairs.

New comparison records use the method name `ABBA4Implicit`. Configuration keys
use the `abba4_outer_projection` prefix to distinguish them from historical
three-projection records. Old measurements must retain their original method
identity. Recalculate a study to obtain records under the new interface; do not
relabel historical three-projection or full-state measurements.

Experiment notebooks now use the current interfaces, explicit physical settings,
and four ABBA4 energy-tracking configurations (two projection formulations times
two solvers). The former R6/R8 notebook filenames remain stable, but their
opening notes explain the changed scientific scope. Stale outputs were cleared.
Exact original versioned notebooks and their outputs are archived locally under
`outputs/legacy_notebooks/`, with source commit and content hashes in a manifest;
the same records remain in Git history. Development notebooks remain ignored.

Validation uses short disposable notebook executions. Their reduced settings and
outputs are never saved into the canonical experiment notebooks.
