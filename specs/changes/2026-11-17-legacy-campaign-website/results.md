# Historical website publication record

On 2026-11-17 the user authorized creation of an independent Cloudflare website for the old simulation. The four preparation documents were presented before publication. The accepted scope preserves the saved `main-20261115/site/` bytes, historical radius convention and eight-panel presentation in the separate `gc2d-main-reflected-legacy` project.

## Source audit

Independent audit and deployment preflight agree on 89 files totaling 578,087,941 bytes, including 86 binary assets totaling 578,035,584 bytes. The metadata retains 5000 cycles, 84 particles, eight methods, eleven radii and 86 available results out of 88. The unavailable cases are rho=0 ABBA2Implicit and ImplicitMidpoint, with their recorded convergence failures.

The HTML SHA256 is `368b21ab714e56539b8e55e4b6a472de0d540f6bbc16df6b984027487e78ea34`; the metadata SHA256 is `4f807628329d2872e9599f5566a000657ad2e0c2bd6112289d0d2411ad73a72e`. Both match the historical remote deployment. The historical code fingerprint is `17319cd275ef1da7d605d9d1b9d6b38ab5ff0216e78b203b16d1fa140c79d90b`; the measured-field source fingerprint remains `4bbd873cd05992e0a615d66ebbf4d96b38c8b3c36be2ca62932e66af7e6d4ced`.

## Verified publication

The independent Pages project `gc2d-main-reflected-legacy` was created successfully with production branch `main`. The first creation attempt delegated automatically to Workers and failed without deploying. Inspection of the installed CLI confirmed that `pages project create ... --force` opts out of that delegation for project creation; the successful command used that flag without overwriting another project. Subsequent publication commands use the normal Pages deployment path without `--force`.

The historical replica is published at [gc2d-main-reflected-legacy.pages.dev](https://gc2d-main-reflected-legacy.pages.dev/), with immutable deployment [c0a67d40](https://c0a67d40.gc2d-main-reflected-legacy.pages.dev/) on branch `main`. At 2026-11-17 07:11:13 UTC, `publication-verification.json` recorded `verified`: all 88 public files (HTML, metadata and 86 binary assets), totaling 578,087,841 bytes, were downloaded completely and matched their local sizes and SHA256 hashes. The 100-byte `_headers` configuration was submitted separately by Wrangler; it accounts for the difference from the full 89-file source inventory and is not counted as a public file downloaded for hash verification.

The complete local source inventory was repeated without changes. Corrected production and pilot HTML/metadata hashes still match their pre-deployment baselines. New evidence is stored separately under `notebooks/developements/main_reflected_campaign/legacy-website-20261117/` in `source-manifest.json` and `publication-verification.json`; existing campaign receipts remain untouched.

Browser control failed with a `timeoutMs` TypeError, so no visual or interaction check was completed. The successful verification establishes that the published files reproduce the archived site; it does not claim that the browser interface was opened or exercised.

The campaign documentation and its Spanish copy now identify the independent historical address and its original radius convention. Memory records the bounded archival exception: preserve this replica's original eight-panel presentation without adding the newer potential/vector background. No simulation or viewer behavior changed, so current behavior specifications did not need modification.

There were no new integrations, Backblaze writes, numerical changes, Git index changes, commits or pushes. Final SDD and focused whitespace checks passed; `MEMORY.md` remains within its 300-line limit.
