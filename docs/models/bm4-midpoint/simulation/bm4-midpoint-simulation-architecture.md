# BM4Midpoint: shared extended-space execution

Twelve alternating signed stages; Optional harmonic coupling. Arithmetic mean after the full cycle; No nonlinear solve.

![BM4Midpoint architecture](bm4-midpoint-simulation-architecture.svg)

The numerical implementation now lives in `src/methods/extended/`.
Public method names and exports from `simulation` remain unchanged. ABBA6
now uses one outer projection; its observer substeps are unprojected pairs and
its nonlinear-work arrays have one column per accepted step. The former `methods/abba/` and
`methods/bm4/` import adapters have been removed. Internal imports point directly
to the defining modules in `methods.extended`.

See the [shared architecture](../../extended/simulation/extended-simulation-architecture.md)
for the module responsibilities, state layouts, solver equations, energy
normalization and validation contracts. The model's mathematical entry point
remains [theory.tex](../tex/theory.tex), with its compiled
[theory.pdf](../tex/theory.pdf).

## Numerical path

Initialization binds this model's recipe and options. The ordinary implicit
path is `advance -> solve_projection -> compose -> direct_map / adjoint_map`.
Explicit midpoint methods use `midpoint_step` in place of the nonlinear
projection. ABBA4 and ABBA6 both project once around the complete
unprojected composition; their recipes contain six and fourteen stages.

The arithmetic projection averages the two copies once after the complete
recipe. It has no nonlinear solver and does not imply exact symplecticity or
reversibility.

The shared composition keeps signed durations and the nonautonomous stage
clock: adjoint maps evaluate at the start of their substep and direct maps at
its end. ABBA uses no harmonic coupling. BM4 retains its configurable coupling
and the existing defaults for each public class.

## Accepted states and diagnostics

For N planar particles, both methods use two spatial copies with 4N components.
Optional passive energy adds N time entries and N physical momenta, producing
6N accepted internal components. Physical outputs and observer snapshots remain
2N dimensional. Time and momentum never enter a nonlinear root.

The accepted shear trace feeds common normalized momentum quadrature after the
spatial computation. The diagnostic `H + kappa - H_initial` measures energy
balance for the time-dependent Hamiltonian. Tracking does not alter the spatial
map or its nonlinear stopping scale. Analytic tangents and BM4 stage events reuse
the retained spatial stages; no extra spatial replay is required for them.

The integration coordinator still owns accepted intervals, off-grid output
sampling, history collection and observer dispatch. Energy arrays follow saved
times and nonlinear-work arrays follow accepted main steps. Existing method
and observer tests remain applicable, supplemented by
`tests/test_extended_family.py`.

## Diagram maintenance

Regenerate this diagram and the shared family view with
`python scripts/render_extended_architecture.py`. The script writes the
PlantUML/Graphviz source and corresponding SVG/PNG views from the same graph
specification. Historical files marked `old` or `proposed` are archival diagrams;
the figure above describes the current implementation.

## Optional JAX execution

This model supports `simulate(..., options=ExecutionOptions(backend="jax", device="cpu"))`
(or a configured `device="gpu"`) with explicit float64. The complete fixed time
loop, particle stages, applicable nonlinear/projection solves and passive energy
quadrature remain on that device until final NumPy `Solution` export. Its
coefficients, physical-state tolerance scaling and sampling conventions are
unchanged. JAX rejects unconverged solves explicitly. Diagnostics report
`execution_mode="device_resident"`. Python step observers and progress reporting
require the default CPU path.

Import `ExecutionOptions` from `contracts.execution_options`. Omission preserves existing
SciPy/NumPy execution. See the [shared execution contract](../../../simulation/jax-execution.md)
for supported physical systems, solver details, timing scope and validation.

## Dimensional HDF5 star calculation

The local calculation notebook at
`notebooks/developements/poincare_section/bm4_midpoint_jax_star/calculation.ipynb`
uses `studies.poincare_star.PoincareStarConfig` and `run_poincare_star` to compose
the existing dimensional HDF5 field with BM4Midpoint and JAX. The generic
`initial_conditions.star.radial_star` constructor places equally spaced
particles on equally spaced straight arms, excluding repeated center points.
Its canonical import is from `initial_conditions.star`.

The example has eight arms of 16 particles, each extending 40% of the source
cell width from its center. Coordinates remain in meters; time is `tau=t/T0`.
The established `rho_hat=0.3` is converted to meters using `lambda/(2*pi)`.
Five forcing cycles at 40 steps per cycle produce 200 complete steps and six
saved states, including initialization. The default destination preserves the
notebook hierarchy in the configured bucket. The paired `visualisation.ipynb`
loads that canonical archive without integration and exports `poincare_star.html`.
It uses `visualization.poincare_star` and the shared comparison viewer for fixed
initial positions, animated once-per-cycle returns, arm/particle selection and
synchronized zoom/panning. Axes remain in meters; optional periodic folding
changes only display copies, never the saved physical coordinates.

## Complete integration executor

`simulate(problem, method, request, execution=Execution())` delegates the complete
integration to `Execution.run`. Import the executor with `from execution import
Execution`. It uses the existing local method implementation and does not
configure runtime resources. A subclass can replace the whole integration while
`simulate` retains final `Solution` construction and validation. Backend choices
remain separate in `options=ExecutionOptions(...)`. See the
[executor contract](../../../simulation/execution.md) for the extension interface.
The numerical stages, projection equations and integration controllers described
above remain inside this execution boundary.

`Execution_Modal` now implements this boundary for remote NumPy/SciPy and JAX CPU
integrations. Results return to the local machine for validation and subsequent
persistence; see the [Modal executor guide](../../../simulation/modal-execution.md).

## Modal JAX CPU star study

The local development study
`notebooks/developements/poincare_section/modal_jax_bm4midpoint_star_48_10_cycles_50_steps/`
uses the canonical `BM4Midpoint` through `simulation.runner.simulate` and
`Execution_Modal`. The reusable composition is `studies.modal_cpu_comparison`;
resource settings belong to `examples/modal_jax_cpu_app.py`.
It compares JAX float64 on CPU allocations of 1, 2 and 4 cores with identical
48-particle inputs: eight arms, six inclusive radii from 0.05 to 0.35 of the
cell radius R=L/2, centered in the periodic cell, first arm along +x. The generic
`initial_conditions.star.radial_star` accepts an optional `inner_radius`; its
existing default spacing is unchanged.

The field is reconstructed from the verified PHI_2 HDF5 data with B=1.5,
characteristic length 0.06, source selection (0,1), cubic interpolation,
rho=0.3 and coupling frequency pi/8. Ten forcing cycles at 50 complete steps
per cycle give h=0.02, 500 steps and 501 aligned states. BM4Midpoint uses an
arithmetic-mean projection and no Newton solve. Each worker runs one first
integration and three warm repetitions; metadata records timings, hardware
and cross-CPU trajectory discrepancies. The paired visualization notebook
only loads saved bucket archives and plots their integer-cycle sections.
This is a short resource-scaling experiment with no independent accuracy
reference. See the [Modal execution guide](../../../simulation/modal-execution.md)
for deployment, transport recovery and persistence behavior.

## Local JAX long-time star study

`studies.local_jax_star` prepares the same verified HDF5 physics for a local
`Execution` with JAX CPU float64. Its matched manual-run notebooks are under
`notebooks/developements/poincare_section/local_jax_bm4midpoint_star_41_5000_cycles_50_steps/`.
They specify one shared central particle and five particles per arm on eight
arms, at positive radii (0.14, 0.28, 0.42, 0.56, 0.70) R with R=L/2: 41
particles in total. The canonical geometry factory `radial_star` now accepts
`include_center=True`, which prepends one central particle and preserves the
existing arm-major order. Its default remains center-free.

The local study performs 5000 forcing cycles at 50 complete steps per cycle,
retaining all 250001 states. Consecutive 50-cycle blocks continue from the
previous accepted physical state at absolute time. This is valid for the
arithmetic-projected BM4Midpoint map with energy tracking disabled: both
accepted spatial copies coincide after every step, and there is no nonlinear
solver history to preserve. The same dynamics object keeps JAX bindings cached.
The block schedule is recorded because floating-point time evaluation can
differ from one uninterrupted compiled loop; no bitwise long-time equivalence
is claimed. The study checks every block's step count, aligned output,
arithmetic projection and absence of nonlinear unknowns.

`diagnostics.run_progress.progress_log` writes flushed timestamped messages to
the notebook and a local log file. It records completed cycles, percentage,
block wall time, elapsed time and an estimated remaining time. A separate
heartbeat thread reports during compilation, integration and upload, without
calling Python observers inside JAX. Errors and interrupts retain tracebacks
and propagate. `COMPLETED AND SAVED` follows successful publication only.
Progress blocks are not restart checkpoints. The numerical result returns
locally and is then saved to the configured bucket; failed uploads retain the
ordinary persistence recovery archive and are reported as failures.

`studies.h5_provenance.prepare_verified_h5_field` is shared with the Modal CPU
comparison; the existing `prepare_cpu_star` name remains its supported study
facade. `visualization.local_jax_star` reads saved states to plot initial
geometry and block timings and export every integer-cycle return. This study
does not extrapolate a short accuracy reference or certify 5000-cycle accuracy.
