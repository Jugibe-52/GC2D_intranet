# Results

Accepted on 2026-10-26 through the explicit request to prepare and execute the
corrected composition proposal, preserving request.py.

## Implemented

Restored request.py byte for byte from the pre-relocation version, including its
validators and defaults. InitialValueProblem now contains a validated request;
omission creates the existing uniform default. with_request returns a problem
sharing physical owners without modifying the original.

simulate and local/Modal executors take problem and method, with optional keyword
execution settings and observers. Numerical methods retain their separate input
arguments. Payload version 3 transports the request only within the problem and
rejects earlier versions. Migrated supported calls, packaging, current contracts,
model/API documentation and Spanish copies. Memory records the new ownership and
payload version while remaining at 300 lines. The earlier relocation record is
retained as history with a link to its replacement.

Nine notebooks were updated, including two tracked experiment notebooks and seven
ignored development notebooks. Stored outputs and execution counts were preserved;
backup notebooks were not changed. Distributions, physical equations, algorithms,
method controls and numerical archive schemas are unchanged.

## Verification on 2026-10-26

Commands use .venv/bin/python and MPLBACKEND=Agg for integrations.

- `-m unittest discover -s tests -v`: 717 tests in 638.685 seconds. The original
  process reported 12 subtest errors in two tests of test_extensible_architecture:
  it had loaded the old custom executor signature before its correction. There
  were no other failures; 3 GPU tests were skipped because no JAX GPU is available.
- After correcting that test executor, `-m unittest discover -s tests -p
  test_extensible_architecture.py -v` passed all 13 tests, covering both affected
  tests and their subcases. The full suite was not repeated; unchanged passing
  checks were reused. No failing checks remain unaddressed.
- Focused checks also passed: test_problem_request.py (5), test_execution.py (13),
  test_core.py (11), test_execution_modal.py (20), test_package_layout.py (13) and
  test_sdd.py (35). They cover nonzero initial times, request ownership, physical
  reuse, custom execution, isolated worker transport and the restored modules.
- `-m mypy src`: passed for 303 source files.
- examples/gc_orbit.py and examples/projected_abba.py completed with 20 and 4 steps.
- Executed all 10 migrated notebook simulation expressions in memory using a
  small seeded potential, one particle, interval (0.2, 0.22), three output times
  and representative RK4/BM4/SDIRK methods. Verified complete output, requested
  times and initial values. No complete notebook or canonical study was rerun.
  The first validation attempt omitted the temporary MAX_STEP binding; after
  adding it to the validation environment, every expression passed.
- `-m build --outdir /tmp/gc2d-compose-build`: wheel and source archive built;
  both were inspected for problem.py and request.py.
- Audited supported Python callers for retired simulate arguments; only the
  deliberate rejection test retains one. Checked notebook output preservation,
  Spanish-copy presence, ignore status and `git diff --check`.
- `scripts/check_sdd.py`: passed, 139 records and 22 requirements.

## Limits and compatibility

All planned implementation and local checks are complete. GPU behavior and an
actual remote deployment were not exercised. Reduced notebook checks cover the
migrated calls, not every pre-existing notebook configuration (one historical
calculation notebook still configures the previously removed coupling_frequency).
Older configured problems and Modal receipts must be recreated; remote clients
and workers must use matching revisions. No deployment, commit or push was made.
