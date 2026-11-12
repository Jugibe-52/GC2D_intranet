# Transverse neighbours of saved Poincare curves

This study extends the historical four-method, 48-particle viewer with 36 fixed
initial positions. The reference particles are 1, 2, 41, 42, 43, 44, 45, 47 and
48 at rho=0.30. Their full-integration Qx agrees closely among the four methods;
this selection does not prove that a trajectory is periodic.

## Geometry and identity

`studies.poincare_parallel` estimates a local tangent using the 40 spatially
nearest saved returns, with the initial point as origin. The unit normal points
away from the component's centroid. The reference is GaussLegendre4; independent
fits from all four methods must differ by at most two degrees. Curve width is
the range of projections onto that normal, in cell fractions. Each new point
lies at minus 2%, minus 4%, plus 2% or plus 4% of this width along the normal.
A polar-ordered polygon of the compact reference component checks that the
negative offsets lie inside and the positive offsets outside. This construction
is validated for these curves; it is not a general reconstruction of arbitrary
self-intersecting or non-star-shaped trajectories.

Particle 2 has a seven-island chain. Returns 0, 7, 14, ... define the initial
component for its normal and width. The family zoom includes the whole chain.
The other references use all saved returns. The same seeds are used for every
rho: inside/outside labels always refer to the original rho=0.30 component,
not to a newly fitted curve at the currently displayed rho. Nearby initial
conditions need not produce trajectories that remain parallel.

| Reference | New IDs |
| --- | --- |
| 1 | 49–52 |
| 2 | 53–56 |
| 41 | 57–60 |
| 42 | 61–64 |
| 43 | 65–68 |
| 44 | 69–72 |
| 45 | 73–76 |
| 47 | 77–80 |
| 48 | 81–84 |

Within each row the order is inner-near, inner-far, outer-near, outer-far.
`seeds.json` records physical positions, cell fractions, unit normals, widths,
classification and cross-method fit checks. Original IDs and colors remain
unchanged. Related light/dark colors distinguish the four new members.

## Numerical execution

`scripts/run_poincare_parallel.py` separates `prepare`, `benchmark`, `run`,
`recover`, `retry-timeout` and `audit`. `examples/modal_parallel_poincare_app.py` deploys CPU JAX float64 workers
with two CPUs, 2 GiB requested memory, 4 GiB maximum memory and no automatic
retries. A full job has a 2400-second limit. The client also monitors a USD 6
study budget and a bounded wall time. Receipts identify submitted calls; a
restart retrieves those calls rather than silently submitting duplicates.

The study reuses the original field fingerprint, historical radial Hamiltonian,
normalization arithmetic, B=1.5, selected source field 15 and solver settings via
`prepare_horizontal_reproduction`. It does not switch the archived experiment
to the later reflection model or the generic physical-cycle import convention.
The initial conditions change, with the recorded recovery attempts below.
The initial batch integrates the 36 new particles jointly for
each of eleven rho values, 0 through 0.50 in increments of 0.05, with
BM4Midpoint, BM4Implicit, RK4 and GaussLegendre4: 44 jobs and 1584 trajectories.
Each job covers 5000 cycles with 50 complete fixed steps per cycle and stores all
5001 returns. The original 48 particles are reused without integration.

BM4Implicit at rho=0.05 failed the original undamped Newton convergence check.
The explicit `recover` mode requires that recorded failure and runs the same
implicit equations with backtracking and an 80-iteration cap instead of 40.
It preserves the fixed step, seeds, duration and absolute/relative tolerances
(1e-12 and 1e-11). The failed result is excluded. A successful recovery records
`solver_recovery` in the archive and exported metric case, and the viewer's
rho note identifies the exception. A 20-cycle Modal comparison of the two
solver configurations passed; its maximum state difference was 8.01e-9 m and
maximum Qx difference 1.82e-7. This is a short consistency check, not a bound
on long-time trajectory differences.

In this run the full damped attempt also failed. The diagnostic worker stops
at the first rejected step and returns its state and residual without exporting
trajectory metrics. With the original controls it rejected step 67577
(time 1351.54): residual 7.07824e-8 versus tolerance 1.1e-11.
`scripts/recover_parallel_individual.py` requires the recorded failed damped
attempt, then runs the 36 independent particles separately with the original
Newton controls and fixed step. Its explicit single-seed configuration has no
radial-star geometry. Every result is checked independently; the script merges
the physical x/y histories and five transport vectors only if all 36 pass.
The merged archive records all call IDs, `grouping_recovery` and a comparison
of the first twenty cycles with the joint short run. The viewer identifies
the changed grouping. Independent equations can still accumulate different
rounding errors when their Newton stopping decisions are made separately.

`retry-timeout` requires a confirmed remote timeout and uses a separate durable
receipt. It leaves numerical settings unchanged. The rho=0.15 BM4Midpoint
retry passed; the timed-out call contributes no published data.

`HorizontalTransportObserver` measures Qx from all accepted full-step endpoints
in physical unwrapped x. See [the metric contract](poincare-horizontal-transport-viewer.md).
The benchmark includes one cycle at 50 and 100 steps and twenty cycles at 50
steps for every method. This short refinement checks the execution and scale
of discretization effects; it is not a long-time convergence certificate or a
proof of periodicity. Implicit stopping across a joint particle block can affect
rounding, so the original archives remain separate and unchanged.

## Publication and verification

After every full archive passes its schedule, source, initial-state, backend,
precision and diagnostic checks, `scripts/export_parallel_poincare.py` appends
new returns and Qx values. It checks that every original browser float32 return
is unchanged, validates all 3696 metrics and checks asset round trips. Scientific
archives retain float64; the browser uses the existing float32 plotting format.

The static site loads one rho asset on demand. Each asset stays below the Pages
25 MiB file limit and has a content hash in its filename. Only the current
encoded trajectory dataset is retained in memory. A failed request leaves the
previous dataset visible and reports the failure. The eleven choices share the
same 84 IDs, four method panels, field backgrounds, cycle controls and viewport.
Nine family groups contain their reference and four neighbours; a separate group
contains the 39 other originals. Group and individual selection, indeterminate
checkboxes, isolate and reference-curve zoom are synchronized. Qx remains fixed
when the displayed cycle interval changes.

Run data, receipts, `audit.json`, `refinement.json`, `export_audit.json` and the
static `site/` live under the ignored development directory
`notebooks/developements/poincare_section/parallel_poincare_families/`.
The site exposes initial positions, all metrics and provenance as downloadable
JSON. Cloudflare publication is a separate authorized CLI operation to a new
Pages project; the original site is preserved. Actual run and deployment evidence
is recorded in the change's results document.

The published [84-particle viewer](https://gc2d-poincare-families.pages.dev/)
contains the audited results. See the
[execution record](../../specs/changes/2026-11-12-parallel-poincare-families/results.md)
for recovery details, validation and deployment evidence.
