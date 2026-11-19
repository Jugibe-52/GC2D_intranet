# Main reflected campaign

The campaign compares ABBA2Midpoint, ABBA2Implicit, BM4Midpoint, BM4Implicit,
ImplicitMidpoint, RK2 (explicit midpoint), RK4 and GaussLegendre4. This explicitly
chosen fixed-step protocol supersedes the general fourth-order comparison defaults
for this experiment only. No adaptive reference is included.

PHI_2.h5 supplies constant field 0 and variable field 15, without smoothing.
Both loader rescalings are enabled: time is measured in physical forcing cycles,
and space in base-cell widths. Guiding-center dynamics applies B=1.5 exactly once.
Cubic clamped reflected splines give a spatial period of two base widths; see
[spatial extension](../dynamics/spatial-extension.md). Requested rho ranges from
0 to 0.50 in increments of 0.05, in base-cell widths. The campaign passes this
value directly to the gyroaverage: `rho_runtime == rho`. The physical equivalent
`rho_m = rho * spatial_scale_m` is recorded only as metadata; for a base width
of 0.18 metres, rho=0.30 means a radius of 0.054 metres.
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
.venv/bin/python scripts/run_main_campaign.py prepare notebooks/developements/main_reflected_campaign/main-20261116-direct-rho
.venv/bin/python scripts/run_main_campaign.py run notebooks/developements/main_reflected_campaign/main-20261116-direct-rho --workers 16
.venv/bin/python scripts/run_main_campaign.py view notebooks/developements/main_reflected_campaign/main-20261116-direct-rho
```

Use a separate directory with `prepare --cycles 100 --pilot` for the eight-method
rho=0.30 pilot. Preparation fingerprints the numerical source tree and scientific
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
For a partial campaign, an availability notice gives the available/expected counts
and identifies the missing combinations. Missing cases retain their panels with
an explanatory message. A missing result or a failed network request does not
prevent the other panels from loading; load errors remain distinct from numerical
failures. Every downloaded data asset retains its size and SHA256 checks.
The footer and `rhoConvention` metadata distinguish the direct base-cell radius
of schema-2 results from the historical schema-1 conversion. For current results,
`spatialScaleM` supplies the recorded cell width in `rho_m = rho * L`; exports
reject cases with incompatible radius conventions.

Deploy the generated `site/` to the `gc2d-main-reflected` Cloudflare Pages project.
Updating the production alias retains the historical immutable deployment URLs
and the separate pilot alias. No single binary asset
exceeds the Pages per-file limit; the complete 5000-cycle asset is 6,721,344 bytes.

After every case is complete and verified in B2, run the CLI's `finalize`
operation with the same directory. It archives `execution-metrics.json` containing
the per-case resource/time/cost records, the sum of worker wall times and the
elapsed span from the first worker start to the last worker finish, then exports
the complete website. Neither aggregate duration includes local transfers.
Campaign inputs and numerical-source snapshots have their own SHA256 fingerprints
and are archived separately; `finalize` does not upload those files.

For the current 86-case publication, use:

```bash
.venv/bin/python scripts/run_main_campaign.py finalize notebooks/developements/main_reflected_campaign/main-20261116-direct-rho --allow-partial
```

The campaign progress remains 86 complete and two failed. Website metadata records
`publicationStatus="partial"`, 86 available cases and 88 expected cases; the two
unavailable entries have reasons and no invented assets. Resource summaries include
only available cases and retain unknown actual billing. Their immutable snapshot
is `execution-metrics-partial-86.json`, leaving `execution-metrics.json` available
for a later complete campaign. `view --allow-partial` uses the same availability
checks but only exports the local website. Neither operation repeats integrations.


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
and current chapter 4 data/figures; code, notebooks and source inputs are excluded.

The five canonical chapter 4 notebook/figure sources were updated and executed
without additional integrations. Their current products retain the source archive
hashes and French figure labels; copied thesis products are identical. The third
set contains 19,623 of 40,000 unwrapped returns inside its displayed viewport;
the fourth has 29,784 of 40,000 inside its original zoom. The website uses modulo-two
coordinates, so its third-set viewport membership differs intentionally from the
unwrapped figure. Historical chapter 4 products remain backed up separately.
A partial publication never marks all 88 numerical cases complete.
