# Results

Accepted and completed on 2026-11-22: replace determinant-only accumulation by explicit ordered physical Jacobian products, recalculate the study and update thesis section 4.8.

## Implementation and numerical record

For each particle the implementation forms `M_0=I`, `M_(n+1)=A_n @ M_n` and evaluates `abs(det(M_N)-1)` directly. All local float64 physical Jacobians are saved, including copy reconstruction and averaging. Products use 80 decimal digits; accumulated matrices and determinants are preserved as decimal strings. Validation rebuilds the matrices with 120 digits. This controls product rounding and cancellation without removing earlier rounding in the local Jacobians.

The full eight-case run `midpoint-jacobian-product-20261122-rho0015` completed with the accepted groups, radius, steps and horizons. It contains 768000 local 2-by-2 Jacobians and 512 accumulated matrices at the requested horizons. Numerical integration plus matrix products took 631.4779984329871 seconds; validation and transfer are excluded. The previous logarithmic record remains unchanged and readable through its schema-1 validation path.

All forward cycle states, returned states, return vectors, return norms and local signed defects match the previous run exactly. Accumulated area defects differ by at most `5.1858625241124115e-14`. BM4 still has smaller maxima in all 16 evaluations of each group, and the area maxima decrease with step refinement at every horizon. The periodic BM4 maximum at 80 cycles and `h=1/160` is now `4.02668614e-12`, displayed as `4.03e-12` in the thesis.

## Verification

- Nine focused geometry tests pass, including oscillator closed forms, independent finite differences of the full nonautonomous map, noncommuting factors and a shear example exposing float64 determinant cancellation.
- Mypy passes for the two changed source modules. The SDD validator and whitespace checks pass.
- Every saved case reconstructs exactly at 80 digits. The 120-digit repeat gives identical float64 area defects for all 512 samples; the largest scaled matrix-entry discrepancy is `8.274185960644919e-76`.
- Direct float64 matrix products can yield spurious area changes: the largest discrepancy from the decimal products across last-horizon samples is about `2521.7685`. The thesis therefore describes the increased accumulation precision explicitly.
- Both notebooks execute successfully from saved results without further trajectory integration. Both regenerated figures retain the accepted boxes, whiskers and individual points; all 128 boxes, 1024 plotted values and file hashes are checked against the new record. Figure text remains French.
- `latexmk -pdf main.tex` confirms the current 89-page `main.pdf`. Its log has no errors, undefined references or overfull boxes; three earlier underfull warnings remain at chapter-4 lines 576–579. Text extraction confirms the new recurrence, precision description and corrected value. The thesis PDF was not opened or rendered.

## Storage and documentation

All thirteen scientific files were uploaded to `gc2d_data:gc2d-notebooks-data/studies/midpoint_geometry/midpoint-jacobian-product-20261122-rho0015` and downloaded again for SHA256 comparison. The verified receipt is dated `2026-11-22T21:01:18.894524+00:00` and its manifest hash is `80fce4813128b4265e3b5fda232e290fe1141c400ad07ab08a0368222826e2c2`. Local files remain a recovery copy.

The study protocol, current diagnostic specification, paired notebooks, thesis generator, README, chapter text and figure provenance are updated. English Markdown changes have matching Spanish reading copies. The French chapter has no corresponding developed subsection and is unchanged. No integration method, reference-campaign archive, Git commit or public deployment was changed. The accepted work is complete.
