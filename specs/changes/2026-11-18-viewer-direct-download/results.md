# Implemented and published

On 2026-11-18 the user accepted the one-click current-frame download proposal. The change applies to the main reference viewer.

Each method panel now has **Télécharger l’image**, which directly downloads a 1200 × 1320 PNG with a white background and French text. It preserves the selected viewport, particles, colors, method, radius and cycle interval, pauses playback and captures the frame before asynchronous encoding. Loading and unavailable results stay disabled. The existing clipboard control remains available. The exported plot has wider margins to separate long vertical tick labels from the axis title. No preview, settings or auxiliary user downloads are added.

## Verification

- Eight focused publication tests passed. JavaScript syntax, mypy for the modified module, SDD validation and whitespace checks passed.
- Only `site/index.html` changed; the other 99 site files are byte-identical to the initial inventory.
- A real local browser download used RK4, rho 0.015, cycles 10–100, the five particles in family 42 and a manually zoomed viewport. The downloaded PNG was inspected visually after fixing the tick-label spacing, and its signature and dimensions verified.
- The browser showed disabled download controls while loading and for the two unavailable zero-radius methods. No browser console errors were reported.
- The browser download-event wait timed out, but the actual downloaded file was found and verified; no event delivery is claimed.

## Publication

Wrangler updated the existing `gc2d-main-reflected-rho-div20` project on branch `main`, uploading one changed file, reusing 98 files and applying `_headers`. Deployment: [e14a06e7](https://e14a06e7.gc2d-main-reflected-rho-div20.pages.dev/).

The stable [reference website](https://gc2d-main-reflected-rho-div20.pages.dev/) shows all eight download buttons after a hard reload. An RK4 click on that public URL reported that the download had started; no newly saved public PNG was located, so completed-file verification is limited to the real local download. Direct byte verification with urllib returned HTTP 403, so public verification is based on the browser and deployment receipt, not a new remote hash audit. The local audit is saved as `direct-download-publication-verification.json` under the reference campaign.

No scientific integrations or changes to thesis sources were needed. The existing concurrent repository edits were preserved; no commits or Git uploads were performed.
