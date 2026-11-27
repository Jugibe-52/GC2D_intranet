# Implemented and published

The user authorized the combined 2 × 4 image download on 2026-11-21. Implementation follows the presented scope on the existing reference website.

The global **Télécharger les 8 évolutions (2 × 4)** button creates a 4800 × 2640 PNG with all eight method panels in display order. It preserves rho, cycles, selected particles, viewport and point size. Playback pauses and rendering captures one synchronous snapshot. Missing results and load errors retain labelled panels. Loading, duplicate clicks and encoding errors are handled explicitly. Existing individual downloads remain available.

## Verification

- All 11 JavaScript viewer tests passed, including grid placement/order, unavailable and load-error slots, selection capture before asynchronous encoding, duplicate-click prevention, loading states, object-URL cleanup and encoding failure recovery.
- All eight focused Python publication tests passed. Mypy for the modified module, SDD validation and whitespace checks passed.
- The actual embedded renderer produced an inspected 4800 × 2640 PNG through the native Canvas library using all eight saved rho=0.015 datasets and cycles 0–5000. The image has the intended labels, axes and layout. This is a renderer verification, not a browser-saved download.
- The local browser showed the button enabled at rho=0 with six available cases and disabled during the switch to rho=0.015. Clicking at rho=0 reported that the download started without console errors. The browser download-event API timed out and no saved file was located; completed browser-file verification remains unavailable. Chrome tab creation also failed in the automation layer; browser checks used the in-app browser.
- Only the reference site's `index.html` changed. All other 99 files are byte-identical; all 97 binary assets also match their recorded sizes and SHA256 values. Numerical integrations were not repeated.

## Publication

Wrangler deployed one changed file and reused 98 uploaded files, then applied `_headers`, on the existing project and branch `gc2d-main-reflected-rho-div20` / `main`. The deployment is [08898ae0](https://08898ae0.gc2d-main-reflected-rho-div20.pages.dev/). The [stable website](https://gc2d-main-reflected-rho-div20.pages.dev/) visibly serves the new button. No full remote-asset hash audit is claimed.

The viewer documentation and Spanish reading copies are updated. No thesis files, personal settings, commits or Git uploads were changed by this task.
