# Implementation and checks

- [x] Add explicit coordinate backends and declarative GC/FC JAX subclasses;
  route shared equations through instance coordinates and fixed module attributes.
  Verify that no property or input inspection selects the coordinate backend.
- [x] Adapt cached execution preparation, preserving source data and optional JAX.
- [x] Update focused tests and measure GC/FC NumPy/JAX agreement, including
  packed histories, derivatives and FC flows with positive and negative durations.
  Use float64 and the existing field tolerances (rtol 2e-11, atol 2e-12), allowing
  the existing derivative tolerance (rtol 2e-10, atol 2e-11).
- [x] Update permanent contracts, model architecture, memory and Spanish copies.
- [x] Run focused tests, source type checks, the headless full suite, both smoke
  examples and the package build. Record unavailable hardware and actual results.
- [x] Review the diff and finish with `python scripts/check_sdd.py`.
