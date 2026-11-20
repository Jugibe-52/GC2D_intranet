# Implemented and published

The user authorized the per-method image-copy control on 2026-11-17. The proposal was presented and implemented within that scope.

Every method panel now offers **Copier l’image**. It renders the same selected particle returns, colors, current zoom and normalized axes at double resolution, with the method title, exact rho, cycle interval and particle count. The resulting PNG is 1200 by 1320 pixels, with a white background and French text. Copying pauses cycle playback and snapshots the selected view before asynchronous encoding. Unavailable/loading panels remain disabled. Unsupported or denied clipboard access produces an explicit download link for the same PNG.

`refresh_campaign_html(site_path)` updates only the HTML from the existing site metadata, preserving its radius convention and all binary assets. The study guide and its Spanish copy document both controls and this refresh procedure. Project memory records the reusable behavior and remains at 300 lines.

## Checks

- All eight focused publication tests passed, including the new HTML refresh test without any scientific archives. Metadata, background/return bytes and headers remain identical, with no numerical loading or field sampling.
- `mypy src/visualization/main_campaign.py`, Node JavaScript syntax checking and `git diff --check` passed.
- The full local publication inventory validated 86 available cases, two explicit failures, 11 phase-resolved backgrounds and all asset hashes. Of the 100 site files, only `index.html` changed; the other 99 files remain byte-identical.
- Local browser checks exercised rho 0.015, family 42, cycles 10–100 and a dragged zoom. A disposable wrapper displayed the actual PNG supplied to the clipboard API: method title, exact rho/cycles, five particles, axes and zoomed ticks were visually checked at 1200 by 1320 pixels. The clipboard write resolved successfully; an independent clipboard read through the browser tool returned no items, so no paste/read-back verification is claimed.
- A disposable denied-permission page showed the correct fallback and kept the unavailable zero-radius methods disabled. The browser download-event wait timed out, but the actual downloaded PNG was found and its signature and 1200-by-1320 dimensions verified. No application console errors were reported.
- The public website shows eight copy controls and successfully completed an RK4 clipboard write with rho 0.015.

## Publication

Wrangler published to the existing project and branch `gc2d-main-reflected-rho-div20` / `main`: [deployment 7157e155](https://7157e155.gc2d-main-reflected-rho-div20.pages.dev/). The stable [reference website](https://gc2d-main-reflected-rho-div20.pages.dev/) serves the update. Wrangler uploaded one changed file and reused 98 existing uploaded files, then applied `_headers`.

The public HTML and metadata, a representative RK4 return asset and its field asset match the local SHA256 values. The immutable deployment HTML also matches. All three historical websites retain their previous HTML and metadata. Verification is recorded separately in `copy-image-source-manifest.json` and `copy-image-publication-verification.json` under the reference run; original publication receipts remain unchanged. This checks representative public assets and every local asset, without claiming a new full public download of all 97 binaries.

No simulations, B2 transfers, changes to thesis files, commits or Git uploads were performed. Temporary preview servers were stopped.
