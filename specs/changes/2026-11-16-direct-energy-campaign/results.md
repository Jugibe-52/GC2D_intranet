# Execution record

On 2026-11-16 the user authorized repeating the campaign, updating the website and chapter 4 notebooks, preserving old Backblaze results and storing new results separately. The user then explicitly accepted direct energy evaluation without quadrature.

## Numerical and storage outcome

The corrected `main-20261116-direct-rho` campaign finished with 86 complete results, all fully verified in its separate B2 prefix. Only rho=0 ImplicitMidpoint and ABBA2Implicit failed nonlinear convergence; neither has a result archive. Campaign status remains `failed` for these two cases, and publication is explicitly partial. Recovery retained the original call IDs and metrics and resubmitted no integrations. `upload-recovery.json` records `readback_complete` at 15:34 UTC for the 19 recovered cases; the collector verified the other 67.

The user manually changed B2 daily spending caps from USD 0.10 to USD 0.30 for storage and from USD 0.30 to USD 0.90 for downloads. The screenshot confirmed saved values. A tiny GET still returned `403 download_cap_exceeded` at 14:53 UTC; at 15:00 UTC the rho=0 RK2 manifest downloaded with the correct size and SHA256. Subsequent full read-back completed successfully. Recovery uses immutable result-only uploads, data before manifests, remote SHA1/size checks and full SHA256 read-back; an exit-0 empty download is rejected.

The final `direct-energy-validation.json` checks manifests and all 86 local results with zero failed checks. It independently reevaluates H and its analytic partial time derivative at 21 distributed saved times for all 84 particles. Maximum absolute differences are 6.94e-17 for H, 1.39e-17 for its derivative and 6.94e-17 for H-H0 (rtol=1e-11, atol=1e-12). Its `partial` status denotes the two unavailable numerical cases, not failed energy checks.

The final zero-radius comparison covers six available methods. Five reproduce historical state arrays bitwise. GaussLegendre4 first differs by 5.55e-17 at t=24.5 and subsequently separates substantially. Frozen numerical sources, worker dependencies, field, particles and time grids match; the old quadrature ran after integration. Amplified floating-point perturbations are consistent with the evidence, but the precise origin remains unresolved.

The final 86-case resource snapshot at 15:35 UTC and `execution-metrics-partial-86.json` agree: summed worker time is 63,924.3656 seconds (17.76 hours), and the first-start/last-finish interval is 8,185.6970 seconds (136.43 minutes). Estimated compute is USD 2.242466746 for requested 4 GiB and USD 2.810115113 for the 8 GiB limit, with two physical CPU cores. Costs of the two failed cases are unknown and excluded; these are not bills. The resource report retains `provisional` because two numerical results are missing, although all 86 available results are B2-verified.

## Publication and retained history

`finalize --allow-partial` completed, archived the partial resource metrics and generated the 5000-cycle website with 86 available/88 expected cases and `rhoConvention="base_cell_width"`. Cloudflare deployment [5656a5cb](https://5656a5cb.gc2d-main-reflected.pages.dev/) is published on branch `main` at the existing [production alias](https://gc2d-main-reflected.pages.dev/). It contains HTML, metadata, 86 data assets and cache headers. Production HTML, metadata and all 86 assets passed full size/SHA256 comparison with the local export at 15:41:32 UTC. `publication-verification.json` records `verified` and the deployment URL. At 15:42:49 UTC, `artifacts-archive-verification.json` confirmed all 24 derived archive files (5,583,897 bytes, including their manifest) through complete SHA256 read-back in the new B2 prefix. This includes the final energy, cost, publication and zero-radius reports and current chapter 4 products, excluding code, notebooks and source inputs.

The updated thesis compiled successfully with `latexmk -pdf main.tex`: `main.pdf` has 66 pages and 9,560,825 bytes. Its log has no errors, LaTeX warnings, overfull or underfull boxes; both edited LaTeX files pass `git diff --check`. Independent review found no discrepancies in the revised text, resource table or cost estimates.

The five canonical chapter 4 notebook/figure sources were updated and executed; `postprocessing/chapter4/current/manifest.json` and `verification.json` record the resulting products and source hashes. The 22 product/source files total 5,953,304 bytes excluding the manifest; thesis copies match, and the historical 28-file backup (14.6 MB) is intact. The three notebooks ran cell by cell through Python with one namespace per notebook and captured/saved outputs; a Jupyter kernel could not start because nbclient encountered socket restrictions. The two Python generators ran directly. AST checks covered all five sources, nbformat validated the three notebooks, and synthetic checks covered extraction and rejection of old metadata. No additional integrations ran. French figures retain unwrapped coordinates: 19,623/40,000 third-set returns are visible, and 29,784/40,000 fourth-set returns lie inside the original zoom. The web uses modulo-two coordinates, which changes third-set viewport membership.

Operational steps are complete; only the separately requested code/input upload authorization remains pending.

The measured HDF5 source matches the historical fingerprint. The historical B2 prefix retains 350 objects (21,804,172,904 bytes), including 86 case manifests; scientific-file sizes match and its provenance receipt passed SHA256 read-back. The [historical deployment 8f6d7942](https://8f6d7942.gc2d-main-reflected.pages.dev/) and [100-cycle pilot](https://pilot.gc2d-main-reflected.pages.dev/) remain separate.

Uploading `source.tar.gz`, `campaign.json` and `campaign-provenance.json` was rejected by automatic approval review because the explicit permission covered results rather than code and particle inputs. These three files remain local and verified; additional user authorization is still pending. No commit or push was made.

## Checks before production

The 15 focused campaign/publication tests passed, including direct H and analytic partial_t H checks for all eight methods, arbitrary integer recording strides, projection reductions and historical-schema reading. Strict mypy passed for all 324 source files; SDD validation passed for 234 records and 22 requirements.

All eight real-field Modal cases at rho=0.30 completed for two cycles in `check-20261116-direct-rho`. They retain 21 samples, schema 2 and rho_runtime=0.30, without energy quadrature. Independent local evaluation of H differed by at most 4.163336342344337e-17. The numerical source fingerprint is `3a20c31ef023ede3977297d2022b71ab24e13abc4ac04b044e0acf1448921e1a`.
