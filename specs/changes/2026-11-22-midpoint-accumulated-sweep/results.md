# Result

Authorized and completed on 2026-11-22. The [study runner](../../../src/studies/midpoint_geometry_sweep.py), [diagnostics and archive module](../../../src/diagnostics/studies/midpoint_geometry_sweep.py), [CLI](../../../scripts/run_midpoint_geometry_sweep.py) and [focused tests](../../../tests/test_midpoint_geometry_sweep.py) implement the accepted accumulated-defect sweep without changing either numerical method.

The run `midpoint-accumulated-20261122-rho0015` completed all eight method/step cases in 636.585184327 seconds of recorded numerical work. Four steps and four horizons give 16 evaluations per group, each including ABBA2Midpoint, BM4Midpoint and both diagnostics. Forward prefixes are shared; every horizon has its own backward integration. The 16 selected particles retain the reference initial positions and the two established groups.

## Verification

- Seven geometry tests passed, including three new tests for oscillator closed forms, a nonautonomous composed-map finite-difference Jacobian and reverse clock, and invalid horizons.
- Thirteen package-layout tests passed. Mypy passed for all 334 source files.
- Raw-data verification independently reconstructed signed log sums, area magnitudes, Euclidean norms and group reductions. All values were finite; there were no orientation flips.
- Maximum local determinant-identity residual: `1.4875611393961702e-15`. Maximum difference between direct and structural accumulated log determinants: `8.609477701778272e-14`.
- At `h=1/40`, saved forward states through 80 cycles and the 10-cycle return states exactly match the corresponding historical run. Maximum signed local-defect differences are `6.5052130349130266e-18` for ABBA2 and `7.115076756936123e-19` for BM4. Maximum log-sum differences are `5.551115123125783e-17` and `2.1141942363467336e-18`, respectively.
- Both [calculation](../../../notebooks/04_08_midpoint_geometry/01_compute.ipynb) and [visualization](../../../notebooks/04_08_midpoint_geometry/02_visualize.ipynb) notebooks passed format and code validation, then executed completely against the verified saved data. Outputs were saved without repeating integration.
- The thesis plotter independently checked the per-particle and group data and generated matching French figures. The PNGs were inspected for legibility. The rewritten section distinguishes area deformation from reversal error and reports nonmonotone cases.
- Final thesis compilation with `latexmk -pdf main.tex` succeeded: 86 pages, no errors or undefined references, and only three pre-existing underfull warnings outside section 4.8. The thesis PDF was not opened or rendered.

## Scientific archive

Local recovery copy: `outputs/midpoint_geometry/midpoint-accumulated-20261122-rho0015`.

Verified destination: `gc2d_data:gc2d-notebooks-data/studies/midpoint_geometry/midpoint-accumulated-20261122-rho0015`.

The immutable record contains eight NPZ cases, parameters, summary, checks, frozen sources and input selections, and a manifest. All thirteen scientific files were downloaded again and verified by SHA256 and size. The verified receipt is dated `2026-11-22T19:40:55.605344+00:00`; its manifest SHA256 is `0a27d18ef53330a903e1228610ba562290f8c4c079a86646a4caa0edbcc6fab7`.

The [permanent protocol](../../../docs/studies/midpoint-geometry-sweep.md) documents reproduction and interpretation. BM4 has smaller maxima for both diagnostics at every tested step/horizon in both groups. Area defects decrease with step refinement at every horizon; the chaotic-candidate return at 80 cycles has nonmonotone exceptions for both methods. These data do not measure error against an exact trajectory.

Historical results and unrelated working changes were preserved. No commits, Git push or public deployment were performed.
