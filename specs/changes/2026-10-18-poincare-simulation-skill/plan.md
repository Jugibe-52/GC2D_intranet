+++
schema_version = 2
kind = "change-plan"
id = "2026-10-18-poincare-simulation-skill"
revision = 1
spec_revision = 1
sources = ["src/studies/poincare_rho_sweep.py", "src/studies/poincare_gap_probes.py", "src/initial_conditions/star.py", "src/visualization/poincare_comparison.py", "src/visualization/_poincare_comparison.html", "src/visualization/poincare_gap_overlay.py", "src/diagnostics/persistence.py", "src/simulation/runner.py", "src/contracts/execution_options.py", "tests/test_poincare_rho_sweep.py", "tests/test_poincare_gap_viewers.py", "tests/test_h5_rescaling.py"]
documents = ["specs/changes/2026-10-18-poincare-simulation-skill/spec.md", "docs/dynamics/gc2d-h5-import.md", "docs/simulation/execution.md", "AGENTS.md"]
open_questions = []
+++

# Poincare simulation skill: implementation plan

## Design

### Shared study and fixed geometry

Add a study module, provisionally `src/studies/poincare_automation.py`, owning
an immutable configuration, the fixed particle fixture, case expansion,
preparation, validation and archive reuse. Keep configuration and algorithms
out of notebooks and the skill prompt. Reuse `ranked_radial_star` for the first
40 fractions and append the exact eight probe fractions from the specification.
Own a versioned particle-set identifier and include geometry in provenance.

Create one GCInitialConfiguration for all 48 particles. Build one
InitialValueProblem and SimulationRequest per `(rho, B, method)` case. Reuse
canonical method classes and `simulation.runner.simulate`; factor a small
shared method-construction helper only if needed to avoid duplicating the
existing dispatch. Do not invoke the historical builder and then undo its
Hamiltonian/time normalization.

Use `Potential.load` with explicit source selection, rescaling flags and sigma;
derive coordinates, runtime rho and cycle schedules directly from its metadata.
Pass B to `GuidingCenterDynamics`. Cache only immutable source preparation or
field samples shared by matching inputs. Validate scalar types, positive
integer counts, signed nonzero B, unique finite rho values and supported
method/solver combinations before expensive execution.

Coupling is configured per forcing-cycle time, default zero. If a runtime
cycle lasts D, pass `coupling_runtime=coupling_cycle/D`; retain both values.
Document absolute Newton tolerance units. In equivalence checks transform
dimensional absolute tolerances consistently, rather than expecting the same
numeric stopping threshold to mean the same physical error in different units.

### Workflow entry point and persistence

Add a thin CLI, provisionally `scripts/run_poincare_automation.py`, with prepare,
run and view operations over a persisted configuration. Allow an output path;
otherwise create a collision-safe study directory under ignored
`notebooks/developements/poincare_section/`. Store a readable configuration,
case manifest, local solution directories and one `viewer.html`. A notebook
wrapper is optional and must call the same helpers, not duplicate their logic.

The prepare operation resolves defaults and lists exact cases without loading
or integrating trajectories. The run operation verifies the source and writes
each completed case using existing persistence, validating states, schedule,
method diagnostics and initial identity before marking completion. Stable
case keys must include a canonical input digest, not rounded rho/B strings.
Record raw frequencies, selected source indices, source SHA256, source/physical
time convention, scales, rho units, B, numerical controls, particle set,
backend/device and actual executor. Case status distinguishes requested,
running, failed and complete; never treat a partial archive as complete.

Reuse identical validated results without calling simulate. Reject destination
conflicts explicitly. Do not treat a historical archive with a legacy clock as
a new physical-time case even when its folded returns are numerically similar.
After interruption, rerun only absent/failed cases; no trajectory-checkpoint
implementation is required. Preserve successful cases when another fails.

Use existing local execution as the baseline and test available NumPy/JAX
paths. Remote execution remains an explicit choice through existing executor
capabilities; if a requested combination lacks support, report it before
submission. Do not add cloud provisioning, paid deployment, scheduling or
bucket synchronization to this change.

### Adapter to the same visualizer

Add a small exporter, provisionally
`src/visualization/poincare_automation.py`, that validates the new archives and
builds datasets for `export_poincare_panel_comparison` and its existing HTML
template. Its generic dataset selector already supports alternative cases,
so rho/B/method labels do not need a new visualization engine or separate
pages. With only rho varying, retain the familiar rho selector label.

Each dataset has exactly two panels: the static field panel and the animated
48-particle return panel. Keep IDs/colors/shapes consistent across datasets.
Reuse the existing palette and probe presentation, adapting any description
that currently assumes separate probe archives. The first 40 retain their
40-entry color scale; all 48 are initially visible. Unavailable datasets keep
their selector entries without fabricated trajectories. With no completed
cases, report that there is no renderable viewer yet.

Sample dynamics through the prepared GC object using the saved source hash and
configuration, including gyroaveraging and B. Apply the specification's
runtime-to-cell/cycle map exactly once. Cache fields across methods only when
source, rho, B, normalization and other field inputs match. Preserve 51 phases,
endpoint audit, fixed per-case scales, playback, selection and start rings.
Export/view must never call simulate. Reuse the notebook link helper when
appropriate and open the generated viewer in Codex on a future run/view request.

### Skill packaging and documentation

Create `.agents/skills/simular-poincare/SKILL.md` and its
`agents/openai.yaml` as a project-local skill once implementation is authorized
with the required `&` prefix. Use the skill-creator guidance and validator at
implementation time. Include concise operation routing, defaults, rho/physical
model definitions, missing-input questions, supported controls, output paths
and tested CLI examples. A local reference Markdown file may carry the longer
parameter table; physics and particle coordinates remain owned by study code.

Use normal skill discovery; do not install a global duplicate or change
personal configuration. Resolve repository paths from the project, not a
hard-coded home directory. Execution follows the user's requested operation;
creation/installation is never a production run. Ordinary parameter changes
within the implemented contract do not need a fresh SDD change.

At closure establish `specs/current/poincare-automation.md` with PNC-001 through
PNC-004. Add focused usage documentation under `docs/studies/` and update the
relevant model documents only where the implemented composition changes their
documented workflow. Follow the memory skill for durable implementation
lessons. Maintain Spanish copies for every new/edited canonical Markdown file.
Future paths in this plan are not checked file references until created.

## Compatibility

The existing `RhoStarConfig`, historical loaders, star/probe archives and
notebooks retain their conventions and entry points. The new study has its own
schema and physical time provenance; it does not pretend that unit-cell
coordinates are the old `characteristic_length` representation. It retains
the user's requested rho meaning through `rho_m=rho*lambda/(2*pi)`.

The default normalized Hamiltonian matches the historical radial dynamics at
the same cycle index after mapping coordinates. Compare this identity on
sample fields, derivatives and short runs; do not use 5000-cycle chaotic
pointwise equality as a compatibility criterion. Keep the old recorded
seconds and source data untouched. One combined new solution replaces the
need for two calculations without rewriting the old separate archives.

The working tree already contains the HDF5 frequency implementation and other
uncommitted changes. Capture the pre-implementation state as the regression
baseline, preserve it, and attribute only this change's edits to this work.
No branch switch, commit or push is part of preparation or implementation.

## Verification plan

Use independent geometry expectations for all 48 fractions and exact probe
IDs/colors, then check invariance across rho, B, method and rescaling choices.
Check the default 250000-step/5001-sample schedule without executing it.

Use a small smooth single-frequency HDF5 fixture with a nonzero cell origin
and a non-unit physical period. Exercise all four rescaling combinations,
rho=0 and 0.3, B=1.5 and another signed nonzero value, and all four methods.
Check transformed Hamiltonian/velocity samples at relative tolerance `1e-10`
with absolute tolerance `1e-12` times each reference field scale. Short-run
coordinate equivalence must have maximum displacement divided by L <= `1e-8`.
Use a DOP853 reference (`rtol=1e-11`, `atol=1e-13` in cell coordinates), audited
by tighter Radau (`rtol=1e-12`, `atol=1e-14`), with a measured reference
discrepancy <= `1e-10` cell. Refine the fixture's validation steps if needed
to distinguish integration error from a normalization defect; record all
actual parameters and discrepancies. Do not change production defaults to
make a check pass.

Run at most 1-2 cycles for representative execution checks, retaining all 48
particles. On the measured source, compare one reduced normalized case with
historical radial fields/short evolution and verify the recorded clocks differ
as expected. Record optional dependency limitations explicitly.

Round-trip new archives, reject mismatched source/geometry/B/rho/method/scales,
and prove a matching rerun calls no integrator. Decode generated HTML payloads
to verify all cases, 48 colors/IDs, `(cycles+1,48,2)` returns and 51 field phases.
Audit endpoint closure at `rtol=1e-11`, `atol=1e-13*field_scale`, as in the
existing sampler. Exercise selection, phase, Start cycle and case switching in
the browser and retain a screenshot of the reduced viewer.

Validate skill routing with representative prepare/run/view prompts, absent
rho/method, explicit overrides, unsupported models and a read-only question.
Run the focused tests, strict mypy and repository-required checks after the
implementation; record only results actually obtained. The structural SDD
checker passing during preparation does not verify these scientific criteria.
