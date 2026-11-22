# Implemented and published

The user authorized point-size adjustment and right-button viewport dragging on 2026-11-21. The presented scope applies to the existing main reference website.

**Taille des points** now sets a shared multiplier from 0.25× to 4×, initially 1×. Accumulated and current returns scale together, with hollow start rings kept outside the current marker. PNG exports use the same sizes. Right-button dragging updates the common viewport during the gesture; left-drag zoom and Shift-left-drag pan remain available. Pointer cancellation and capture loss release the gesture. View bounds stay inside the spatial period-two square.

## Verification

- Seven JavaScript tests passed using the actual embedded viewer functions with canvas and pointer-event doubles: live pan, resized canvas coordinates, boundary clamping, release outside the canvas, left zoom, Shift pan, unrelated pointers, cancellation, capture loss, and screen/PNG drawing sizes.
- Eight focused Python publication tests passed. Mypy for the modified module, SDD validation and whitespace checks passed.
- The local browser showed sizes 1×, 4× and 2×, rho 0.015, eight available methods, cycle changes, left-drag zoom and suppression of the right-click context menu. No console errors were reported. One automation key action timed out, but the subsequent UI state confirmed the requested 4× value.
- The browser tool cannot perform native right-button drags; that gesture is verified by the event tests, not claimed as a native browser drag check.
- Only `site/index.html` changed. The other 99 files are byte-identical, including all 97 binary assets, each verified against its recorded SHA256 and size.

## Publication

Wrangler deployed one changed file, reused 98 files and applied `_headers` to the existing `gc2d-main-reflected-rho-div20` project on branch `main`. Deployment: [a98c55f7](https://a98c55f7.gc2d-main-reflected-rho-div20.pages.dev/).

The stable [main website](https://gc2d-main-reflected-rho-div20.pages.dev/) shows the new instructions and size control. Changing it to 2× with cycles 0–100 worked; rho=0 still shows six available results and the two explicit unavailable cases. No console errors were reported. Verification is based on the deployment receipt, public browser and local hashes; no new full remote-asset hash audit is claimed. The detailed inventory is stored in `point-size-pan-verification.json` under the reference campaign.

The publication documentation and its Spanish copy are updated. No scientific integrations, thesis-source edits, personal-configuration changes, commits or Git uploads were needed.
