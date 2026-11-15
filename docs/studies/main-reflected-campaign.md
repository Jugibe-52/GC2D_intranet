# Main reflected campaign

The campaign compares ABBA2Midpoint, ABBA2Implicit, BM4Midpoint, BM4Implicit,
ImplicitMidpoint, RK2 (explicit midpoint), RK4 and GaussLegendre4. This explicitly
chosen fixed-step protocol supersedes the general fourth-order comparison defaults
for this experiment only. No adaptive reference or thesis editing is included.

PHI_2.h5 supplies constant field 0 and variable field 15, without smoothing.
Both loader rescalings are enabled: time is measured in physical forcing cycles,
and space in base-cell widths. Guiding-center dynamics applies B=1.5 exactly once.
Cubic clamped reflected splines give a spatial period of two base widths; see
[spatial extension](../dynamics/spatial-extension.md). Requested rho ranges from
0 to 0.50 in increments of 0.05 and means physical radius rho*0.06/(2*pi) metres.
The exact physical positions, IDs and colors of the existing 84 particles are
read from the sibling thesis's `datos/conjuntos_particulas_rho030.json`. Their
selection is historical, not recomputed for the new reflected field.

Each of 88 cases advances all particles jointly for 5000 cycles, with 40 fixed
steps/cycle. Ten aligned samples/cycle give 50001 state rows `(time, 168)`;
no shadow samples or interpolated outputs are permitted. All numerical execution
and diagnostic reductions use JAX float64 on Modal CPU. Analytic Newton uses
absolute tolerance 1e-12, relative tolerance 1e-11 and at most 40 corrections.

## Scientific records

Archives retain unwrapped coordinates, H and positive analytic partial_t H at
saved times. Composite Simpson quadrature uses every accepted endpoint on the
40-step grid. `energy_integral` and `energy_balance = H-H0-energy_integral`
have shape `(84, saved_times)`. `energy_quadrature_difference` is Simpson minus
trapezoid accumulated on the same accepted endpoints: it measures sensitivity,
not a rigorous error bound. Time-dependent H is not a conserved quantity.

For the two implicit projection methods, `projection_multipliers_sampled` has
shape `(saved_times-1, 2, 84)`, with x/y components and signs preserved.
`projection_particle_norms` and `projection_particle_block_max` have shape
`(saved_times-1, 84)`. The latter includes all four accepted steps between samples.
There is no initial multiplier. Nonlinear iteration/residual counts refer to
the joint particle batch, not independent per-particle Newton counts.

## Execution and recovery

```bash
.venv/bin/modal deploy examples/modal_main_campaign_app.py
.venv/bin/python scripts/run_main_campaign.py prepare notebooks/developements/main_reflected_campaign/main-20261115
.venv/bin/python scripts/run_main_campaign.py run notebooks/developements/main_reflected_campaign/main-20261115 --workers 16
.venv/bin/python scripts/run_main_campaign.py view notebooks/developements/main_reflected_campaign/main-20261115
```

Use a separate directory with `prepare --cycles 100 --pilot` for the eight-method
rho=0.30 pilot. Preparation fingerprints the numerical source tree and scientific
inputs. Reuse the original directory and `run` to recover confirmed Modal receipts;
never delete receipts to force resubmission. An unconfirmed submission requires
inspection. Local result collection is serialized to bound RAM, while up to 16
cases execute remotely. Complete local archives survive upload failures.

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

Deploy the generated `site/` to a separate Cloudflare Pages project. The existing
catalog and previous deployments are not overwritten. No single binary asset
exceeds the Pages per-file limit; the complete 5000-cycle asset is 6,721,344 bytes.

After every case is complete and verified in B2, run the CLI's `finalize`
operation with the same directory. It archives `execution-metrics.json` containing
the per-case resource/time/cost records, the sum of worker wall times and the
elapsed span from the first worker start to the last worker finish, then exports
the complete website. Neither aggregate duration includes local transfers.
The production archive also retains the campaign inputs and a numerical-source
snapshot with SHA256 fingerprints.

For the explicitly accepted 86-case publication, use:

```bash
.venv/bin/python scripts/run_main_campaign.py finalize notebooks/developements/main_reflected_campaign/main-20261115 --allow-partial
```

The campaign progress remains 86 complete and two failed. Website metadata records
`publicationStatus="partial"`, 86 available cases and 88 expected cases; the two
unavailable entries have reasons and no invented assets. Resource summaries include
only available cases and retain unknown actual billing. Their immutable snapshot
is `execution-metrics-partial-86.json`, leaving `execution-metrics.json` available
for a later complete campaign. `view --allow-partial` uses the same availability
checks but only exports the local website. Neither operation repeats integrations.


The accepted 86-case production viewer is published at
[gc2d-main-reflected.pages.dev](https://gc2d-main-reflected.pages.dev/), with 5000
cycles per available result. At rho=0, ImplicitMidpoint and ABBA2Implicit show
recorded nonlinear-convergence failures; the other ten radii contain all eight
methods. The [100-cycle pilot](https://pilot.gc2d-main-reflected.pages.dev/)
remains separate. Deployment `8f6d7942` contains 86 data assets and two public
HTML/metadata files, plus cache headers. The campaign-local publication receipt
records verification evidence; a partial publication does not mark all 88
numerical cases complete.
