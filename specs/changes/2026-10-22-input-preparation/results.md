# Execution results

The user authorized preparation and execution using the assistant's judgment on
2026-10-22. The four preparation documents were presented together before code
changes. Their scope was accepted through that combined request.

## Implemented

Private input preparation now separates ownership and validation from assignment
in initial conditions, physical histories, diagnostics, GC/FC parameters and
accuracy records. Gauss--Legendre and SDIRK share Newton configuration preparation.
Common comparison and Poincare configuration use named local input helpers.
Six visualization modules prepare the ten identified plotting paths before
creating figures. The change affects 16 source files, including one new private
module. Public signatures, exports, numerical stepping and archive schemas remain
unchanged.

The integration architecture, current state/result descriptions and their Spanish
reading copies are aligned. MEMORY.md records the verified preparation pattern
and contains 292 lines. All nine Spanish reading copies remain ignored by Git.

## Checks

Checks use Python 3.12.3 and `.venv/bin/python` from the repository root.

- The focused contract, state, ownership, initial-condition, comparison and
  Poincare selection passed 86 tests in 27.565 seconds. The separate new plotting
  module passed seven tests in 2.105 seconds, covering all ten plotting functions,
  later invalid entries, existing figures and valid rendered curves.
- Loading the original visualization modules from commit `5a77ff6` in memory
  makes the new plotting regressions fail in 14 cases, with no test errors.
  Loading current modules passes them. A separate comparison of valid figures
  found exact agreement for all ten plots: curves, styles, labels, annotations,
  limits and scales.
- An in-memory comparison with `5a77ff6` checked 694 constructor cases across
  physical dynamics, both shared Newton methods, Poincare/common comparison
  configurations, initial values, histories and accuracy records. Accepted
  values, normalized types, array dtype/read-only flags and exception types and
  messages matched. Inputs included booleans, NumPy scalars, strings, non-finite
  values and malformed shapes.
- `python -m mypy src` passed for 259 source files. Both supported examples
  passed: RK4 used 20 steps; projected ABBA used four steps with one Newton
  correction per step.
- `python -m build` built the source and wheel distributions successfully.
- `MPLBACKEND=Agg python -m unittest discover -s tests -v` completed 641 tests
  in 441.862 seconds: 638 passed, three GPU-only tests skipped, no failures or
  errors. Available CPU and JAX checks passed; this host has no JAX GPU.
- An AST comparison confirmed unchanged public function/method signatures and
  explicit exports in all 15 modified existing source modules. No tracked
  notebook changed. `git diff --check` passed.
- Final `python scripts/check_sdd.py` passed (58 records, 19 requirements).
  All prepared implementation and verification steps are complete.

## Scope and limitations

No notebook, saved scientific data, numerical formula or public input policy was
changed. Long notebook campaigns, deployment and synchronization were not run.
Checkpoint assertion policies and other separately identified behavioral changes
remain outside the prepared scope. No commit or push was made.
