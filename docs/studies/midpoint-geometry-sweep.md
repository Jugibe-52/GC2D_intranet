# Accumulated midpoint defects across step and horizon

The current thesis section 4.8 uses `midpoint-jacobian-product-20261122-rho0015`. It compares ABBA2Midpoint and BM4Midpoint with four fixed steps, `h=1/20,1/40,1/80,1/160`, at horizons of 10, 20, 40 and 80 physical cycles, starting at phase zero. A cycle has normalized duration one. Each group has 16 evaluations: one evaluation is a `(h,T)` combination containing both methods and both diagnostics.

The radius is 0.015 in base-cell widths. The reflected field, magnetic field 1.5, cubic interpolation, selected source mode 15, unsmoothed HDF5 fingerprint and original initial positions come from `main-20261117-rho-div20`. Coordinates remain unwrapped. The two groups retain their IDs and colors: periodic candidates `1,2,60,64,65,72,76,83` and chaotic candidates `12,15,18,21,24,27,30,33`. Only these 16 particles are integrated because they are exactly the groups being compared. These labels do not prove periodicity or chaos. The original 84-particle reference campaign is not rerun or altered.

## Measurements and interpretation

For each method and step, one forward trajectory reaches 80 cycles. Its prefixes provide all four horizons. Each horizon has an independent backward integration, starting at the saved forward state and using times `N*h,(N-1)*h,...,h` with step `-h`. Each physical advance reconstructs and averages its own temporary copies.

For each particle evaluate the complete physical Jacobian `A_n=D Phi(t_n,y_n)` through the canonical analytic composition derivative, including copy reconstruction and averaging. Accumulate the Jacobian with `M_0=I` and `M_(n+1)=A_n @ M_n`. At each horizon evaluate `abs(det(M_N)-1)` directly, without logarithmic sums or determinant products.

Local Jacobians are evaluated in float64. Their entries are converted exactly to decimal arithmetic and matrices are multiplied at 80 decimal digits; validation rebuilds the product at 120 digits. Defects must agree with relative tolerance `1e-20` and absolute tolerance `1e-30`, and matrix entries with scaled error below `1e-40`. This precision controls product rounding and determinant cancellation, but cannot remove earlier rounding in the local Jacobians. Direct and structural local defects remain independent controls; they do not determine the new accumulated defect. Nonfinite states and matrices are rejected.

The return diagnostic is the Euclidean norm of the difference between the final backward state and the original state, per particle, in base-cell widths. It includes accumulated symmetry defects, sensitivity along the trajectory and rounding; it is not the error relative to an exact trajectory. No energy-conservation surrogate or adaptive reference integration is used. At each horizon, group summaries take the maximum and median of the eight particle values. They do not maximize over time or mix local defects with accumulated values. Defects near local-Jacobian rounding error require care, and differences between horizons need not be monotone.

## Reproduction and saved arrays

The paired notebooks [calcul.ipynb](../../notebooks/04_08_defauts_geometriques/calcul.ipynb) and [visualisation.ipynb](../../notebooks/04_08_defauts_geometriques/visualisation.ipynb) select the same run and `studies/midpoint_geometry` prefix. The calculation notebook reuses completed files by default; explicitly enable computation with a fresh run ID to repeat the experiment. The visualization notebook only reads verified saved data and invokes the thesis plotting script.

From the repository root:

```bash
.venv/bin/python -B scripts/run_midpoint_geometry_sweep.py
.venv/bin/python -B scripts/run_midpoint_geometry_sweep.py --archive-only
```

The local directory `outputs/midpoint_geometry/midpoint-jacobian-product-20261122-rho0015` is a recovery copy. The scientific destination is `gc2d_data:gc2d-notebooks-data/studies/midpoint_geometry/midpoint-jacobian-product-20261122-rho0015`. Uploads use the immutable `ArtifactStore`; a verified receipt is created only after downloading and checking every scientific file. Failures preserve recoverable local results. Notebook sources are not synchronized through the result bucket.

Each of eight method/step cases has a separate NPZ. Float64 local Jacobians have axes `(step,particle,2,2)`; accumulated products are saved as decimal strings with axes `(horizon,particle,2,2)` and their determinants as `(horizon,particle)` to preserve their precision. Signed local defects have axes `(step,particle)`. Cycle states use `(cycle,component-major xy)`, and backward states use `(horizon,component-major xy)`. Return vectors have axes `(horizon,particle,coordinate)`. Area and return magnitudes have axes `(horizon,particle)`. Parameters preserve the 16-ID particle order. Forward and backward timings exclude transfer, validation and matrix products, whose timing is recorded separately. They are not independent runtime benchmarks.

The record contains parameters, per-particle and group summaries, validation results, a SHA256/size manifest, and frozen sources and input selections. The HDF5 field is identified by its fingerprint rather than duplicated. The historical local and 100-cycle experiment remains available through [its original protocol](midpoint-geometry.md).

## Checks and figures

Focused tests compare accumulated ABBA2 oscillator area and reversal with closed forms, compare a nonautonomous composed-map Jacobian with independent finite differences, and expose a deliberately wrong reverse clock. A noncommuting-shear test checks factor order. Saved-data validation reconstructs matrix products at 80 and 120 digits, area magnitudes and Euclidean norms from raw arrays; the plotter independently checks group reductions.

The thesis script `TFM_redaccion/notebooks/geometria_midpoint_barrido.py` exports `simplecticidad_midpoint_acumulada_fr` and `simetria_midpoint_acumulada_fr` in PDF, PNG and provenance JSON. Both figures have two group rows and four horizon columns, common logarithmic axes within each figure, and the same method colors. Each panel plots boxes of eight defects against `h`: first and third quartiles, median, minimum–maximum whiskers and eight individual points. Quartiles are linearly interpolated on untransformed defects. Horizontal offsets only improve readability. The generator rejects nonpositive values instead of omitting particles on the logarithmic axis. Printed fonts follow the thesis typography registry at a final width of 160 mm.

## Status and history

The [current change record](../../specs/changes/2026-11-22-midpoint-jacobian-product/results.md) documents the matrix-product run, precision validation and actual B2 archive status.
The previous `midpoint-accumulated-20261122-rho0015` sweep used logarithmic accumulation; its results and checks remain a [historical record](../../specs/changes/2026-11-22-midpoint-accumulated-sweep/results.md). Schema-1 records remain readable and are not relabeled as matrix products.
