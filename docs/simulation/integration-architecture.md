# Integration and the four state formulations

All 13 public methods use `IntegrationMethod.new_run` and `integrate_method`.
Dynamics defines physics; formulation defines coordinates; method defines one
numerical step; integration controls accepted intervals and collects output.

| Formulation | One planar particle | N planar particles |
|---|---:|---:|
| Physical | 2 | 2N |
| Physical with energy | 4 | 4N |
| Spatially duplicated | 4 | 4N |
| Spatially duplicated with energy | 6 | 6N |

Use `track_energy=True` on any public method. Classical methods use
`PhysicalFormulation`; ABBA and BM4 use `DoubledFormulation`. These concrete
objects live in `src/simulation/formulations/state.py`. Energy variants append one
time entry and one normalized momentum per particle. Projected methods
store diagonal accepted copies and project spatial variables only. Every
component block has N entries, and `components(state)` exposes a uniform
`(coordinates, N, *sample_axes)` view. The time entries repeat the scalar
integration time. `extended_time` has shape `(N, saved_times)`; `Solution.t`
remains the one-dimensional grid.

`StepResult.state` is internal; `Solution.states` and observer states are physical.
`StepResult.statistics` contains accepted physical work. Details are transient
unless an observer explicitly retains a copied event. `export_history` delegates
to the formulation's common physical and energy extraction.

Tracking does not affect physical stages, Newton/Broyden scales or adaptive error
control. DOP853 and Radau own physical-only SciPy solvers and use eight-point
Gauss quadrature over accepted dense output for passive momentum. This quadrature
has no independent error tolerance; audit it by refinement. Radau's optional
Jacobian always differentiates physical coordinates only. Diagnostic interpolation
work and BM4's accepted energy replay are excluded from physical work counters.

Fixed controllers compute off-grid samples with independent shortened maps;
adaptive controllers evaluate accepted dense output. The formulation validates each
particle time and aligns diagnostic times to the requested grid within round-off.
The coordinator and collector do not inspect clock indices or tracking flags.
Observer snapshots cannot mutate
samples or counters. Repeated runs own fresh solver resources.

The former `fully_extended` selector is rejected. It described spatial, time and
momentum projection in eight duplicated coordinates and is not an energy-monitoring
option. Existing historical full-state diagnostics are readers of that former
record type, not an active simulation path. Canonical model guides and theory
PDFs document the current methods; older rendered diagrams are historical.
