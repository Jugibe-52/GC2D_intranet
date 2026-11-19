# Main reflected campaign

## Current thesis reference

The thesis reference is `main-20261117-rho-div20`, published at
[gc2d-main-reflected-rho-div20.pages.dev](https://gc2d-main-reflected-rho-div20.pages.dev/).
The sibling thesis's [reference selection](../../../TFM_redaccion/datos/campana_referencia.json)
is the single selection record for its campaign, website, B2 prefix, radius grid
and representative radius. Chapter 4 sources read that record and verify it
against the saved campaign. Use `rho=0.015` for its representative figures and
planned comparisons; the sweep is `rho_i=i/400`, i=0,...,10.
This is a choice of smaller input radii, not an additional runtime normalization.

The reference contains 86 verified results out of 88 expected cases. At rho=0,
ImplicitMidpoint and ABBA2Implicit have recorded nonlinear convergence failures.
Keep those gaps explicit. `main-20261116-direct-rho`, `main-20261115` and their
websites remain historical references with their original parameters and data.
Do not relabel their archives, change the generic campaign defaults or change
unrelated study protocols when updating the thesis reference. The particle JSON
files preserve the historical selection of the initial positions; their provenance
radii do not select the current simulation radius.

## Shared protocol

The campaign compares ABBA2Midpoint, ABBA2Implicit, BM4Midpoint, BM4Implicit,
ImplicitMidpoint, RK2 (explicit midpoint), RK4 and GaussLegendre4. This explicitly
chosen fixed-step protocol supersedes the general fourth-order comparison defaults
for this experiment only. No adaptive reference is included.

PHI_2.h5 supplies constant field 0 and variable field 15, without smoothing.
Both loader rescalings are enabled: time is measured in physical forcing cycles,
and space in base-cell widths. Guiding-center dynamics applies B=1.5 exactly once.
Cubic clamped reflected splines give a spatial period of two base widths; see
[spatial extension](../dynamics/spatial-extension.md). The reference rho ranges from
0 to 0.025 in increments of 0.0025, in base-cell widths. The campaign passes this
value directly to the gyroaverage: `rho_runtime == rho`. The physical equivalent
`rho_m = rho * spatial_scale_m` is recorded only as metadata; for a base width
of 0.18 metres, rho=0.015 means a radius of 0.0027 metres.
The exact physical positions, IDs and colors of the existing 84 particles are
read from the sibling thesis's `datos/conjuntos_particulas_rho030.json`. Their
selection is historical, not recomputed for the new reflected field.

Each of 88 cases advances all particles jointly for 5000 cycles, with 40 fixed
steps/cycle. Ten aligned samples/cycle give 50001 state rows `(time, 168)`;
no shadow samples or interpolated outputs are permitted. All numerical execution
and diagnostic reductions use JAX float64 on Modal CPU. Analytic Newton uses
absolute tolerance 1e-12, relative tolerance 1e-11 and at most 40 corrections.

## Scientific records

Every method evaluates the same physical Hamiltonian directly at its saved
states and the common output times. Archives retain unwrapped coordinates,
`physical_hamiltonian`, positive analytic `hamiltonian_time_derivative` and
`energy_drift = H-H0`, with energy arrays shaped `(84, saved_times)`.
No energy quadrature is computed: new schema-2 archives omit `energy_integral`,
`energy_balance` and `energy_quadrature_difference`. The variation H-H0 is not
a conservation error because the Hamiltonian depends on time. Historical
schema-1 archives remain readable with their original diagnostics.

For the two implicit projection methods, `projection_multipliers_sampled` has
shape `(saved_times-1, 2, 84)`, with x/y components and signs preserved.
`projection_particle_norms` and `projection_particle_block_max` have shape
`(saved_times-1, 84)`. The latter includes all four accepted steps between samples.
There is no initial multiplier. Nonlinear iteration/residual counts refer to
the joint particle batch, not independent per-particle Newton counts.

## Execution and recovery

```bash
.venv/bin/modal deploy examples/modal_main_campaign_app.py
.venv/bin/python scripts/run_main_campaign.py prepare notebooks/developements/main_reflected_campaign/main-20261117-rho-div20 --rhos 0 0.0025 0.005 0.0075 0.01 0.0125 0.015 0.0175 0.02 0.0225 0.025
.venv/bin/python scripts/run_main_campaign.py run notebooks/developements/main_reflected_campaign/main-20261117-rho-div20 --workers 16
.venv/bin/python scripts/run_main_campaign.py view notebooks/developements/main_reflected_campaign/main-20261117-rho-div20 --allow-partial
```

These commands document campaign preparation, recovery and return-only export;
the saved reference results already exist. For thesis figures, extract them
without integration. For the reference website's field backgrounds, use the
visualization notebook or the source-aware export described below after `view`.
The older eight-method rho=0.30 pilot uses a separate directory prepared with
`prepare --cycles 100 --pilot`; it does not select the thesis reference.
Preparation fingerprints the numerical source tree and scientific
inputs. With unchanged code and inputs, reuse the original directory and `run` to recover confirmed Modal receipts;
never delete receipts to force resubmission. An unconfirmed submission requires
inspection. Local result collection is serialized to bound RAM, while up to 16
cases execute remotely. Complete local archives survive upload failures.

The existing `main-20261115` results and its 100-cycle pilot used the legacy
physical radius `rho*0.06/(2*pi)` metres, divided by the spatial scale before
gyroaveraging. The correction on 2026-11-16 does not change those saved results
or their recorded radii. Prepare a new campaign directory for the corrected
convention; code fingerprints reject resuming an old campaign with the new code.
The records for `main-20261115` below remain historical; recovery of the corrected
campaign is recorded separately.

B2 uses `gc2d_data:gc2d-notebooks-data/developements/main_reflected_campaign/<campaign>/<case>`.
The manifest is uploaded last, and complete remote archives are read back with
hash validation. Worker images contain code, never B2 credentials.

### Recovery after a B2 download cap

`rclone copyto` may fail with exit 7 during a download HEAD request even when
uploads and listings remain available. Preserve sanitized stderr: exit 7 alone
does not identify the cause. In this campaign a listing-based GET exposed
`403 download_cap_exceeded`; the account's daily download cap had been reached.
Include full archive read-back in the transfer budget. Raising a cap in the B2
console can take up to ten minutes to take effect; see the
[B2 caps and alerts documentation](https://www.backblaze.com/docs/cloud-storage-create-and-manage-caps-and-alerts).

The campaign-local `recover_b2_uploads.py` stages only the saved pending archives
through hard links, verifies their original manifest SHA256 digests, and copies
to the bucket root with `--immutable --checksum --multi-thread-streams 0`.
B2's native multipart upload avoids the generic multi-thread copier's additional
download HEAD. Remote listing sizes and SHA1 hashes are checked before the
manifests are published and again afterwards. This does not change account caps,
overwrite different objects, resubmit simulations or change the scientific code.

`verify_b2_recovery.py` then downloads those recovered archives and verifies every
file against its original SHA256 and size, resuming completed cases from
`upload-recovery.json`. Until this succeeds, cases remain
`uploaded_pending_readback`, not `complete`. The two missing numerical cases
remain failures. Partial publication requires the explicit option below; saved
cases whose uploads are still unverified cannot be omitted through that option.

### Corrected campaign recovery

On 2026-11-16, the user manually changed the daily B2 spending caps from USD 0.10
to USD 0.30 for storage and from USD 0.30 to USD 0.90 for downloads. The current
screenshot confirms the saved settings. A tiny manifest GET at 14:53 UTC still
returned `403 download_cap_exceeded`. At 15:00 UTC, about seven minutes later,
the rho=0 RK2 manifest downloaded with the correct size and SHA256, confirming
restored download access after the manual change. Full result read-back finished
at 15:34 UTC; `upload-recovery.json` records `readback_complete` with zero
simulation resubmissions. Confirm access with a listing-based GET: a direct rclone read can return
exit 0 with zero bytes under this failure, so neither an exit code nor the settings
screenshot is sufficient.

The corrected campaign's `recover_b2_results.py` refuses to run while collection
is active. Its `upload` operation selects saved pending results, verifies their
local manifests and uploads immutable data before manifests using the bucket-root
copy described above. Its `verify` operation performs full SHA256 read-back,
retains completed checkpoints and skips cases already verified. Neither operation
resubmits numerical work or includes code and campaign-input snapshots.

### Corrected numerical outcome

`main-20261116-direct-rho` finished with 86 complete results, all verified in B2
against their full SHA256 digests. Only rho=0 ImplicitMidpoint and ABBA2Implicit
failed nonlinear convergence and have no result archives. The campaign retains
status `failed` for those two cases; its accepted export is explicitly partial.
The historical `main-20261115` objects and immutable website deployment remain
separate and unchanged.

`direct-energy-validation.json` checks all 86 local archive manifests and samples
21 saved times for all 84 particles in each result. Independent H, analytic
partial_t H and H-H0 comparisons pass with maximum absolute differences
6.94e-17, 1.39e-17 and 6.94e-17 respectively. The report has zero failed checks;
its `partial` status reflects the two unavailable numerical cases.

`zero-radius-comparison.json` compares the six available rho=0 methods with the
historical campaign. Five state arrays are bitwise identical. GaussLegendre4 first
differs by 5.55e-17 at t=24.5 and later separates substantially. Frozen numerical
sources, worker dependencies, field, particles and time grids match; the historical
energy quadrature ran only after integration. Amplification of floating-point
perturbations is consistent with the evidence, but the precise origin of the
initial difference is not established.

The code/input snapshot is verified locally. Uploading `source.tar.gz`,
`campaign.json` and `campaign-provenance.json` was rejected by automatic approval
review because the explicit upload permission covered results. Those three files
have not been uploaded; additional user authorization remains pending. This does
not change the completed verification of the scientific result archives.

## Time and cost per case

Each worker records requested and maximum physical CPU cores, requested/maximum
memory, visible logical CPUs, processor model, software versions, wall time,
process CPU time and process-lifetime peak RSS. RSS can include earlier inputs
in a reused container. Wall time includes decoding, preparation, compilation,
integration and diagnostic reduction; it excludes container startup and transfers.
The case also retains its Modal call ID and Unix start/end timestamps.

The two cost estimates multiply this measured interval by the CPU allocation
and either requested or maximum memory. The [Modal price list](https://modal.com/pricing),
checked 2026-11-15, supplies USD 0.0000131/physical-core-second and
USD 0.00000222/GiB-second. Modal bills the greater of requested and used resources;
see [resource billing](https://modal.com/docs/guide/resources). Estimates cover
only worker compute, exclude credits, taxes, startup, idle time, transfers and
B2/Cloudflare services, and are not invoices. Actual billed cost remains explicitly
unavailable unless reconciled against a separate Modal billing record.

For the corrected campaign, `execution-metrics-partial-86.json` and the
15:35 UTC `resource-table.json` agree on 63,924.3656 seconds of summed worker
time (17.76 hours) and an 8,185.6970-second worker span (136.43 minutes).
The estimated totals are USD 2.242466746 at the requested 4 GiB and
USD 2.810115113 at the 8 GiB limit, both with two physical CPU cores. These cover
only the 86 returned results: costs of the two failed cases are unknown, not zero.
The resource report's `provisional` label reflects missing numerical cases,
although all 86 available cases have completed B2 verification. It is not a bill.

## Poincare-only website

By default, the exporter requires all campaign cases. Explicit partial publication
admits B2-verified results and recorded failed cases without saved archives;
pending uploads, unknown cases and corrupt archives still stop the export.
The exporter selects every tenth scientific sample, including cycle zero.
It writes lossless little-endian float64 arrays
`(cycle, 84, xy)`, with a SHA256 fingerprint for each lazily loaded asset. Display
coordinates are reduced modulo two, not folded into period one. State archives
remain unwrapped. All eight methods remain visible without a method selector.
The compact white desktop layout has RK2, ImplicitMidpoint, RK4 and GaussLegendre4
in the first row, followed by ABBA2Midpoint, ABBA2Implicit, BM4Midpoint and BM4Implicit
in the second. Narrow screens use two or one columns. Two shared start/current-cycle
sliders have one-cycle minus/plus buttons and a visible interval indicator; moving
one endpoint past the other moves both to that cycle. Reset restores the full time
interval and Full view restores the shared unit-square viewport `[0,1] × [0,1]`,
also used on initial load. Returns outside this viewport are clipped; stored
period-two coordinates and the numerical trajectories are unchanged. The website retains
rho/particle/family selection, animation and synchronized zoom/pan. It excludes
intermediate trajectory samples, energy/projection plots and horizontal transport Qx.

Each method panel provides **Télécharger l’image** for a direct PNG download of
the current frame, with no preview, export settings or auxiliary files. The
existing **Copier l’image** control remains available. Both use a 1200-by-1320 PNG
with a white background, the method name, exact rho label, cycle interval and
selected-particle count. The image retains the current particle selection and
colors, viewport and zoom, normalized axes, tick values and scale, accumulated
returns, start circles and current positions. Export margins keep long tick
labels clear of the vertical-axis title. It captures the view at the click
and pauses cycle playback. Both controls are disabled for loading, unavailable or failed
results. If the browser lacks image clipboard support or denies the write, the
panel explicitly reports that copying is unavailable and offers **Télécharger
l’image** for the same PNG; it does not claim clipboard success.

For a partial campaign, an availability notice gives the available/expected counts
and identifies the missing combinations. Missing cases retain their panels with
an explanatory message. A missing result or a failed network request does not
prevent the other panels from loading; load errors remain distinct from numerical
failures. Every downloaded data asset retains its size and SHA256 checks.
The footer and `rhoConvention` metadata distinguish the direct base-cell radius
of schema-2 results from the historical schema-1 conversion. For current results,
`spatialScaleM` supplies the recorded cell width in `rho_m = rho * L`; exports
reject cases with incompatible radius conventions.

Deploy the reference campaign's generated `site/` only to the
`gc2d-main-reflected-rho-div20` Cloudflare Pages project. Preserve the separate
`gc2d-main-reflected`, historical and pilot publications. No single binary asset
exceeds the Pages per-file limit; the complete 5000-cycle asset is 6,721,344 bytes.

After every case is complete and verified in B2, run the CLI's `finalize`
operation with the same directory. It archives `execution-metrics.json` containing
the per-case resource/time/cost records, the sum of worker wall times and the
elapsed span from the first worker start to the last worker finish, then exports
the complete website. Neither aggregate duration includes local transfers.
Campaign inputs and numerical-source snapshots have their own SHA256 fingerprints
and are archived separately; `finalize` does not upload those files.

For the reference campaign's 86-case publication, use:

```bash
.venv/bin/python scripts/run_main_campaign.py finalize notebooks/developements/main_reflected_campaign/main-20261117-rho-div20 --allow-partial
```

The campaign progress remains 86 complete and two failed. Website metadata records
`publicationStatus="partial"`, 86 available cases and 88 expected cases; the two
unavailable entries have reasons and no invented assets. Resource summaries include
only available cases and retain unknown actual billing. Their immutable snapshot
is `execution-metrics-partial-86.json`, leaving `execution-metrics.json` available
for a later complete campaign. `view --allow-partial` uses the same availability
checks but only exports the local website. Neither operation repeats integrations.
Both CLI exports contain returns only. Before publication, regenerate the field
backgrounds with `export_campaign(directory, source=field_path, allow_partial=True)`;
the reference campaign's `visualisation.ipynb` performs that export.

For changes only to viewer controls or presentation, refresh the existing HTML:

```python
from pathlib import Path
from visualization.main_campaign import refresh_campaign_html

refresh_campaign_html(Path(
    "notebooks/developements/main_reflected_campaign/main-20261117-rho-div20/site"
))
```

`refresh_campaign_html(site_path)` reads the recorded radius convention and cell
width from `data.json`, then rewrites only `index.html`. It preserves `data.json`,
`_headers` and all 97 existing binary assets (86 return arrays and 11 field
backgrounds) byte for byte, without reading trajectory archives or resampling
fields. Do not use `view` or `finalize` for this operation: those commands export
returns without a field source and replace the background metadata. Keep the
original publication manifests and receipts as historical evidence, and record
verification of the changed HTML separately before deploying the reference site.

### Historical publication and chapter 4 products of 2026-11-16

The historical 86-case viewer remains available at immutable deployment
[8f6d7942](https://8f6d7942.gc2d-main-reflected.pages.dev/); the
[100-cycle pilot](https://pilot.gc2d-main-reflected.pages.dev/) remains separate.
The corrected campaign's `finalize --allow-partial` completed and generated
`site/` with 86 available cases, two explicit convergence failures, 5000 cycles,
`rhoConvention="base_cell_width"` and the corrected code/source fingerprints.
Deployment [5656a5cb](https://5656a5cb.gc2d-main-reflected.pages.dev/) is published
on branch `main` at [the production alias](https://gc2d-main-reflected.pages.dev/).
Production metadata, HTML and all 86 assets passed complete size/SHA256
comparison with the local export at 15:41:32 UTC; `publication-verification.json`
records `verified` and the deployment URL. At 15:42:49 UTC,
`artifacts-archive-verification.json` confirmed 24 archived derived files totaling
5,583,897 bytes, including their manifest, with complete SHA256 read-back in the
new prefix. These include energy, resource, publication and zero-radius reports
and that campaign's chapter 4 data/figures; code, notebooks and source inputs are excluded.

The five canonical chapter 4 notebook/figure sources were updated and executed
without additional integrations. Their products from that date retain the source archive
hashes and French figure labels; copied thesis products are identical. The third
set contains 19,623 of 40,000 unwrapped returns inside its displayed viewport;
the fourth has 29,784 of 40,000 inside its original zoom. The website uses modulo-two
coordinates, so its third-set viewport membership differs intentionally from the
unwrapped figure. Historical chapter 4 products remain backed up separately.
A partial publication never marks all 88 numerical cases complete.

## Independent historical replica

The saved `main-20261115/site/` is also published at
[gc2d-main-reflected-legacy.pages.dev](https://gc2d-main-reflected-legacy.pages.dev/)
in its own Cloudflare Pages project, with immutable deployment
[c0a67d40](https://c0a67d40.gc2d-main-reflected-legacy.pages.dev/) from 2026-11-17.
It retains the old physical radius `rho*0.06/(2*pi)` metres and the original
5000-cycle, 84-particle results: 86 available cases and the two recorded rho=0
convergence failures. It is separate from the corrected production and pilot.

This is a faithful archive republication. Preserve its original eight-panel
layout and controls as a bounded exception to the current two-panel viewer
standard; do not add potential/vector backgrounds or regenerate its data.
The complete historical local inventory stayed unchanged: 89 files totaling
578,087,941 bytes. All 88 public files (HTML, metadata and 86 data assets), totaling
578,087,841 bytes, passed complete download and SHA256/size comparison at
07:11:13 UTC. The remaining 100 bytes are `_headers`, submitted separately by
Wrangler as configuration. Corrected-production and pilot HTML/metadata hashes
matched their pre-deployment baselines.

The separate `legacy-website-20261117/source-manifest.json` and
`publication-verification.json` under the campaign development directory preserve
the inventory and verification evidence. No integration or B2 write was performed.
Browser control failed with a `timeoutMs` TypeError, so no new visual or interaction
check was completed; verification covers the published bytes and metadata.

## Radius-divided reference campaign

The `main-20261117-rho-div20` campaign uses `rho_i = i/400`, i=0,...,10:
the original requested radii divided by exactly 20. This changes the input radii,
with no conversion inside the dynamics. In particular, the original rho=0.30
case supplies the input value 0.30/20=0.015 for the new case; this does not imply
identical trajectories. The 84 initial positions, eight
methods, 5000/40/10 cycle/step/sample protocol, field, solver controls and
direct-energy evaluation remain the same as the corrected campaign.

The CLI accepts `prepare --rhos` followed by explicit finite non-negative radii;
later operations use the saved campaign record. Canonical decimal keys retain
the historical minimum of two fractional digits and enough further digits to
distinguish every radius. For example, `rho-0.00-RK4` stays unchanged while
`rho-0.0025-RK4` and `rho-0.005-RK4` identify distinct results.

```bash
.venv/bin/python scripts/run_main_campaign.py prepare notebooks/developements/main_reflected_campaign/main-20261117-rho-div20 --rhos 0 0.0025 0.005 0.0075 0.01 0.0125 0.015 0.0175 0.02 0.0225 0.025
.venv/bin/python scripts/run_main_campaign.py run notebooks/developements/main_reflected_campaign/main-20261117-rho-div20 --workers 16
```

The calculation and visualization notebooks share this run directory and the
explicit B2 prefix `developements/main_reflected_campaign/main-20261117-rho-div20`.
Scientific results and derived reports belong there; the numerical source and
input snapshot remains local. The independent Cloudflare project is
`gc2d-main-reflected-rho-div20`; the corrected, historical and pilot sites retain
their own destinations. Execution and publication outcomes are recorded in the
[change results](../../specs/changes/2026-11-17-rho-div20-campaign/results.md).

Before production, all eight methods completed two real-field cycles at
rho=0.015 with the full 84-particle batch and the same fixed-step and output
schedule. Independent evaluation of all saved samples passed, with maximum
absolute differences 5.56e-17 for H, 1.05e-17 for partial_t H and 8.33e-17 for
H-H0, rounded upward. The preflight contains no energy quadrature; it verifies
the execution path and diagnostics, not long-time convergence.

For the new viewer, call `export_campaign(directory, source=field_path)` after
the result checks; pass `allow_partial=True` only when terminal numerical
failures are retained explicitly. The optional source adds a shared left panel
of the actual gyroaveraged Hamiltonian and GC velocity field to the eight-method
grid. It reconstructs no trajectories. The 128-by-128 potential grid and
33-by-33 velocity grid cover `[0,2]²` at 51 forcing phases; H=Phi_rho/B and the
velocity use the existing normalized coordinates and clock without extra factors.
Independent endpoint evaluations must agree before exact display closure is
applied. Each radius keeps its color and arrow scales fixed across phases.

The phase controls are independent of the common start/current-cycle controls.
The initial viewport is `[0,1]²`, and a separate button shows `[0,2]²`.
Hollow circles follow the selected particles' saved positions at the start cycle.
Potential and vector layers can be toggled independently. French labels identify
the phase and normalized units. Field assets contain float64 values and separate
SHA256/size checks, with 7,573,296 bytes per radius at the default grid sizes.
Omitting `source` still permits a return-only export; archived published sites
are not regenerated automatically.

### Radius-divided collection outcome

The `main-20261117-rho-div20` collector finished on 2026-11-17 with 86 complete
results, each verified against its full B2 bytes and SHA256. Only
`rho-0.00-ImplicitMidpoint` and `rho-0.00-ABBA2Implicit` failed nonlinear
convergence and have no saved archives. The campaign retains status `failed`;
publication must show those two unavailable combinations explicitly. The earlier
campaigns and their B2 prefixes remain separate.

The final direct-energy audit verified all 86 available archives with zero
failed comparisons, checking 21 saved times for each of the 84 particles. The
maximum absolute differences for H, partial_t H and H-H0 are 5.56e-17, 1.39e-17
and 9.72e-17, rounded upward. Its `partial` status reflects only the two missing
numerical cases. The final resource report records `complete_with_failures`,
`final_snapshot=true`, 86 B2-verified cases, zero pending cases and no data errors.
The 86 resource records sum to 68815.513930861 worker-seconds (19.12 hours) and
span 6314.1570773124695 seconds (105.24 minutes). Estimated costs are USD
2.414048229 for requested 4 GiB and USD 3.025129992 at the 8 GiB limit, using two
physical cores. These estimates cover returned records only; actual billing and
the two failed calls' costs remain unknown. Summed worker times overlap and the
observed worker span is not elapsed launch-to-publication time.

The final local export passed `verify_publication.py local`: 100 files totaling
661417826 bytes include 86 return assets, 11 field assets, HTML, metadata and
`_headers`. The frozen source manifest binds all files to this campaign.
The independent [production website](https://gc2d-main-reflected-rho-div20.pages.dev/)
is published through immutable deployment
[6115ea3b](https://6115ea3b.gc2d-main-reflected-rho-div20.pages.dev/).
At 10:46:04 UTC, `publication-verification.json` confirmed full size/SHA256
comparisons of 99 public files totaling 661417726 bytes: HTML, metadata,
86 return assets and 11 field assets. `_headers` contributes the remaining
100 local bytes as deployment configuration. The HTML and metadata of the
corrected, historical and pilot websites still match their frozen baselines.

The public in-app-browser checks at 10:47:25 UTC confirmed explicit failure
panels at rho=0, all eight methods at rho=0.0025 and rho=0.0175, independent phase
and cycle controls, start-cycle updates, potential visibility, period/base-cell
views and French labels, with no console errors. `browser-verification.json`
binds this sampled interaction evidence to the same source manifest as the
complete asset checks; it does not claim exhaustive browser testing of every
radius/control combination.

At 10:48:00 UTC, `reports-archive-verification.json` confirmed six immutable files
and 438356 bytes read back with SHA256 below the new prefix's `reports/` path.
The five reports cover direct energy, resources, public files, the source manifest
and browser checks; their manifest was uploaded last. The source/input snapshot,
code and notebooks remain local and excluded from that archive. Numerical
collection, final audits, independent publication and report archival are
complete, while the two convergence failures remain recorded without retries.
