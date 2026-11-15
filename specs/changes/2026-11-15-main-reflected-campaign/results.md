# Execution record

On 2026-11-15 the user accepted preparation and execution of the agreed simulation,
B2 storage and Poincare-only Cloudflare viewer, subsequently requesting resource,
time and estimated cost records per rho/method case. This implementation did not
modify the thesis; a later, separately authorized update of section 4.1 is recorded
in the sibling thesis repository.
The [permanent protocol](../../../docs/studies/main-reflected-campaign.md) describes
inputs, sampling, physical units, archives and recovery commands.

Implemented named RK2 explicit midpoint (two stages), ImplicitMidpoint (one stage),
signed projection multiplier statistics, compact JAX energy/projection diagnostics,
recoverable Modal orchestration, B2 read-back verification and the static viewer.
Workers request two physical CPU cores and 4096 MiB RAM, with an 8192 MiB limit;
at most sixteen cases run in parallel. Resource/time/cost metadata are passive.
Estimated USD amounts use dated public tariffs and explicitly exclude actual billing.

## Checks completed

- Strict mypy passed for 322 source files. Source and wheel distributions built successfully.
- The full suite ran 786 tests with three skips. Twelve subcases of one BM4 test
  initially errored because dictionary equality cannot compare the newly recorded
  vector multipliers. The test now compares each statistic elementwise, preserving
  exact equality; all seven tests in that module passed. Numerical code was not changed
  by this repair. The other full-suite tests passed; the whole suite was not rerun.
- Eight campaign tests passed: second-order convergence and NumPy/JAX agreement,
  quadratic conservation for implicit midpoint, exact cubic Simpson integration,
  fourth-order quadrature convergence, signed per-particle block reductions,
  whole-cycle/period-two selection, schedule validation and recovery after local
  polling timeout without a second submission, and refusal to finalize incomplete
  campaigns while preserving unknown actual billing (several checks share one test).
- Both new model theory PDFs compiled. SDD validation passed for 204 records and
  22 requirements. Project memory records the campaign-specific protocol in 296 lines.
- Eight-method, rho=0.30, 84-particle Modal pilots completed at two and 100 cycles.
  Every archive was uploaded to B2 and reloaded with hash verification. The 100-cycle
  pilot measured worker times from 6.25 to 41.91 seconds and process lifetime peak
  RSS around 774–954 MiB. These are pilot measurements, not full-run estimates.
- Browser QA of the 100-cycle viewer confirmed eight panels, family selection and
  start-cycle control. Binary assets contain exactly 101 returns per particle,
  including cycle zero; ten-sample scientific histories are excluded. Full-cycle
  behavior is additionally tested by independent sample selection.

## Production results and archive recovery

`notebooks/developements/main_reflected_campaign/main-20261115` contains the frozen
campaign and 88 distinct confirmed Modal call IDs. A local zero-wait timeout was
initially reported as a case failure by the submitting client; the remote calls
continued. Recovery now catches both SDK and built-in timeout classes and reuses
original receipts. No numerical job was resubmitted. The collector stored results
one at a time to bound local RAM while Modal ran up to 16 cases.

At 17:45 UTC, Modal reported 19 successful calls, 16 running inputs and 53 queued
inputs, with no failed call observed at that time. Local/B2 collection trailed compute
completion; `progress.json` is the authoritative per-case collection record. RK2 at rho=0
completed and passed B2 verification with worker wall time 491.5109 seconds.
The scientific code fingerprint still matches the prepared campaign.

B2 also contains `campaign.json`, `source.tar.gz` and `campaign-provenance.json`
under `gc2d_data:gc2d-notebooks-data/developements/main_reflected_campaign/main-20261115`.
The source snapshot corresponds to the submitted numerical implementation;
subsequent local finalization helpers do not alter that implementation.

The completed Modal computation produced 86 local archives. ImplicitMidpoint and
ABBA2Implicit at rho=0 failed nonlinear convergence and have no complete archive.
Their numerical failures are distinct from the 39 saved cases whose B2 upload
failed. There are no remaining running or queued Modal inputs. The obsolete
collector and waiting finalizer were stopped; they will not publish an incomplete
88-case campaign automatically.

The upload wrapper originally retained only rclone exit code 7. A controlled retry
exposed a failed download HEAD, and a bucket-root GET identified the precise cause:
`403 download_cap_exceeded`. The user increased the daily download cap from
0.10 to 0.30 USD and the storage cap from 0.03 to 0.10 USD. These account changes
were made by the user, not by the recovery scripts.

On 2026-11-15, `recover_b2_uploads.py` recovered all 39 saved archives, comprising
117 data objects and 39 manifests (10.653 GiB including manifests). Local SHA256
digests matched the original manifests; all remote objects matched their size and
SHA1 through bucket listings. Manifests were uploaded after their data. Full
remote SHA256 read-back of all 39 recovered cases finished successfully at
21:08:29 UTC through `verify_b2_recovery.py`. Their cases became complete only
after that check. The campaign-local `upload-recovery.json` records the exact
files, hashes and verification timestamps. No simulations were repeated or
scientific sources changed during recovery.

The final progress record has 86 complete cases and the two numerical failures.
A fresh remote inventory matched all 344 local archive files in path and size:
21,803,491,097 bytes (21.803 GB or 20.306 GiB). SDD validation passed for 209 records
and 22 requirements; `git diff --check` passed. The recovery guidance and its
Spanish reading copy are aligned, and `MEMORY.md` remains at 300 lines.

The 86 measured cases total 18.409 worker hours over an 84.026-minute interval
between their first start and last finish. Requested-memory and maximum-memory
cost scenarios total 2.3248 and 2.9133 USD; these exclude the two numerical failures
and all items outside the measured worker interval. Actual invoice amounts remain
unknown. All records report two physical cores requested/limited, 4096/8192 MiB
memory, Python 3.12.10 and JAX 0.7.1. CPU model is unknown. Process-lifetime peak
RSS ranges from 2.266 to 5.063 GiB and can include earlier inputs in a reused worker.

## Viewer refinement

The user approved the families-style interface on 2026-11-15. The viewer now
uses a compact white layout, two start/current-cycle sliders with minus/plus
buttons, and all eight methods permanently visible without a method selector.
Desktop order is RK2, ImplicitMidpoint, RK4, GaussLegendre4 above ABBA2Midpoint,
ABBA2Implicit, BM4Midpoint, BM4Implicit. Moving either endpoint beyond the other
moves both to that cycle. Reset restores the full interval; Full view restores
the shared spatial view. Existing scientific data and asset fingerprints are unchanged.

Browser checks passed for eight panels in two desktop rows, exactly two sliders,
no method selector, synchronized intervals 1–99, 100–100 and 0–0, disabled buttons
at bounds, playback and reset to 0–100. The narrow-screen layout also remained
usable without horizontal overflow. JavaScript syntax and strict mypy (322 files)
passed. Cloudflare deployment `32ab58b0` published the updated 100-cycle pilot at
https://pilot.gc2d-main-reflected.pages.dev/; the public page loaded and verified
all eight assets. The updated exporter also supplies the pending production viewer.
Project memory retains this campaign-specific layout decision in 296 lines,
with its complete Spanish reading copy.

The subsequently authorized unit-square display uses `[0,1] × [0,1]` on initial
load and after Full view, clipping outside returns while preserving the
period-two assets. Deployment `d51ecae6` updates the same pilot URL. Public-browser
QA confirmed the 0–1 axes for all eight panels, zoom and restoration with Full
view. JavaScript syntax passed; Cloudflare uploaded only the changed HTML and
reused all nine other files. The production exporter uses the same viewport.

## Remaining work

Resolve the two numerical failures before finalizing a complete 88-case campaign;
neither upload recovery
nor partial result availability satisfies that prerequisite. Consolidated
`execution-metrics.json` and the production Poincare export remain pending.
The published Cloudflare site is still the 100-cycle pilot. Actual Modal invoice
charges remain unavailable. No Git commit, push or production deployment is part
of this archive-recovery operation.

## Partial publication implementation

On 2026-11-15 the user authorized publishing the 86 available results with the two absent cases visible. Implementation and validation are in progress; the numerical campaign remains 86/88 complete.
