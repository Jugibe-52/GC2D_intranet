# Execution results

Accepted on 2026-10-23 through `$sdd prepara y ejecuta`, following the agreed
method-owned fixed step and shortened final-step policy.

## Implemented

Fixed methods inherit `FixedStepMethod` and accept a validated keyword-only
`step` (default 0.1). `SimulationRequest` contains only interval and output times.
DOP853/Radau own their former request bound as `max_step`. Host and JAX share
integer-indexed starts and a shortened final duration; `advance` keeps its name.
Method configuration and independent shadow sampling remain separate.

Supported studies, examples, tests and ten notebook sources are migrated.
Eight development notebooks remain ignored; two experiment notebooks remain
tracked. Historical backup/executed notebooks and saved numerical outputs are
preserved. Model guides and eight extended-family diagrams describe ownership;
shared documentation, Spanish copies and memory are updated (298 memory lines).

## Verification

- Before/after captures cover thirteen method classes, energy off/on and
  NumPy/SciPy/JAX CPU: 52 runs, 670 diagnostic/state arrays. All arrays agree
  exactly (maximum difference 0) for [0, 0.04] with nominal step 0.02 and saved
  times 0, 0.005, 0.02, 0.033, 0.04. Adaptive tolerances retain their defaults.
  The seeded guiding-center field uses A=0.08, M=3, 16x16 samples, interpolation
  order 5, seed 27 and rho=0.05. This measures regression agreement, not physical
  trajectory accuracy.
  CPU/JAX differences on these captures are 0 for physical states,
  8.14e-20 or less for passive momentum and 1.39e-17 or less for physical
  Hamiltonian and generalized energy.
- New shortened-final-step CPU/JAX checks passed for all eleven fixed classes,
  including passive energy and nonlinear iterations, on [0.3, 0.355] with
  step 0.02. State tolerances are rtol=3e-10, atol=3e-11; energy tolerances are
  rtol=3e-9, atol=3e-11. Pure tests cover clipping, configuration reuse, shadow
  independence, invalid values and unrepresentable clocks.
- `MPLBACKEND=Agg JAX_ENABLE_X64=true .venv/bin/python -m unittest discover
  -s tests -v` ran 665 tests in 619.461 s: 661 passed, three GPU tests were skipped,
  and one resume test still used the old request-owned-step distinction. Its
  input was corrected to change the method's step; a final 33-test run covering
  Modal execution, core behavior and the scalar/clipping cases passed. Thus the
  full run plus the focused rerun cover all current cases; the monolithic run
  itself reported that one failure. No known failures remain.
- Another 71-test run covering integration, validation, adaptive and Poincare
  behavior passed. The new eleven-family shortened-step CPU/JAX test also passed
  in the complete suite.
- Strict mypy passed for all 305 source files. Both smoke examples and the source
  and wheel build passed. The new diagram was visually inspected.
- All ten changed notebook sources compile. In-memory checks of cells 2/4/6 of
  both basic orbit notebooks passed with M=2, 16x16 fields, interval [0, 0.005],
  step 0.002 and seven samples, producing durations 0.002, 0.002, 0.001.
  A development HDF5 notebook passed over [0, 0.025] with step 0.1. Reduced
  settings and outputs were never saved into the canonical notebooks.

Initial migration test failures were corrected; earlier failed logs remain in
`/tmp/gc2d-method-step/` alongside numerical captures and verification logs.
JAX GPU checks are unavailable on this machine. Remote job serialization was
checked locally; no cloud run or deployment was performed. Final diff review,
Spanish-copy/ignore checks and `.venv/bin/python scripts/check_sdd.py` passed
(89 records, 22 requirements). The SDD checker validates document structure,
not numerical correctness. The accepted implementation is complete.
