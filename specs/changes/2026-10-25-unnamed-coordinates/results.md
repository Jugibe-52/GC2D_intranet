# Results

On 2026-10-25 the user accepted separate GC/FC coordinate classes with respective
JAX subclasses and requested SDD preparation and execution. The four revised
documents were presented together before implementing this final hierarchy.
A later explicit `sincronizar-codigo` invocation separately authorizes Git/B2
synchronization; publication is outside the SDD operation itself.

## Implemented

`PhysicalCoordinates` is a stateless abstract base. `GCCoordinates()` fixes
`(x, y)` and `FCCoordinates()` fixes `(x, y, vx, vy)`, with read-only dimensions
2 and 4. There are no dimension constructor fields, names or role-index tables.
`split()` returns ordinary tuples. `JaxGCCoordinates` and `JaxFCCoordinates`
inherit their respective conventions and only select `jnp` as array namespace.
`CoordinateValues` and `JaxPhysicalCoordinates` are removed without aliases.

Dynamics, contracts, snapshots, method guards, exports and consumers use these
classes. `StateLayout.snapshot()` explicitly supplies an immutable NumPy
convention; compatibility does not infer a model from dimension. Custom systems
retain explicit coordinate implementations. GC velocity access preserves the
previous exceptions: `ValueError` on coordinates/initial data, `TypeError` on
completed results. GC map rejection retains the position-coordinate explanation.

Permanent guides, affected model architecture, current contracts and memory are
updated with Spanish copies. No notebooks were edited or executed. No numerical
formulas, units, integration algorithms or archive schemas changed. The separate
shared-fixed-methods proposal remains preparation only.

## Checks

All commands use `.venv/bin/python` (Python 3.11.10) from the repository root.

- Focused suite: 81 tests passed in 25.526 seconds across `test_core`,
  `test_states`, `test_state_ownership`, `test_physical_workspaces`,
  `test_extensible_architecture`, `test_function_methods`,
  `test_shared_array_dynamics` and `test_package_layout`. This includes JAX
  tracing, backend agreement, immutable snapshots and generic component axes.
- `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v` completed
  708 tests in 735.834 seconds: 703 passed, three GPU tests were skipped, and two
  error-contract regressions were reported in `test_abba` and
  `test_initial_solvers_formulations`. The process had imported the earlier code
  before the corrections; its exit status was 1, not a full-suite success.
- Both regressions were corrected. Repeating the complete affected modules with
  `MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_abba tests.test_initial_solvers_formulations tests.test_states tests.test_physical_workspaces -q`
  passed all 47 tests in 23.057 seconds, including external-snapshot validation.
  No known failures remain. The entire suite was not repeated after these
  error-path corrections; the other successful checks are reused unchanged.
- `python -m mypy src` passed for all 306 source files, including the corrections.
- Both examples passed: `examples/gc_orbit.py` completed 20 steps;
  `examples/projected_abba.py` completed four with one Newton correction each.
- `python -m build --outdir /tmp/gc2d-fixed-coordinates-build-final` produced
  the source archive and wheel after the corrections.
- All 23 changed permanent Markdown documents have ignored Spanish copies with
  matching code blocks and coordinate identifiers. Change documents also have
  Spanish copies. Memory remains within its limit at 299 lines.
- Source, examples, scripts and notebooks (including ignored development files)
  contain no retired coordinate identifiers. Tests mention removed fields only
  to verify their absence. `git diff --check` passed.

Logs for this execution use `/tmp/gc2d-fixed-coordinates-*`. The final wheel's
291 bundled Python sources match the working sources exactly. Final
`python scripts/check_sdd.py` passed: 118 records and 22 requirements. The
accepted implementation is complete, with the validation limits stated above.

## Numerical and archive comparison

The original pre-change capture is `/tmp/gc2d-unnamed-before`; the final hierarchy
capture is `/tmp/gc2d-fixed-coordinates-after`. Both use
`JAX_ENABLE_X64=1 .venv/bin/python scripts/capture_physical_state_baseline.py <output> --backend scipy`.
All 789 arrays across 54 cases and every recorded scalar diagnostic are exactly
equal: zero observed difference. The later corrections affect error paths only.

The script fixes a random potential (seed 43, amplitude .05, two modes,
16-by-16 grid), two particles, GC rho .1/B 1.5, FC rho .2/eta .1, interval
[.15,.255], step/max_step .02 and eight irregular output times. It covers
classical, adaptive and projected methods, energy observation and solver variants.
This is a NumPy/SciPy before/after comparison; JAX agreement is tested by the
suites, not a separate before/after numerical capture.

GC and FC archives created before any edit, with three times and two particles,
load with identical physical arrays, initial data and position/velocity views,
and the correct fixed NumPy class. Existing numerical archives need no migration.
Old configured-object pickles must be recreated; current objects serialize.
GPU execution is unavailable on this machine and is not claimed.

## Superseded implementation evidence

The first execution removed names and `CoordinateValues` but introduced positive
`state_dimension` and validated immutable position/velocity indices. The user
rejected those configurable fields, leading to the final hierarchy above. Its
initial working tree was clean; no commit or push occurred during that execution.

That interim version passed 70 focused tests in 16.509 seconds, mypy for 306
files, both examples (20 and four steps), and the source/wheel build under
`/tmp/gc2d-unnamed-build`. Its `/tmp/gc2d-unnamed-after` numerical capture exactly
matched the original 54 cases/789 arrays, and pre-change archives loaded correctly.
Its 19 changed permanent documents and five change documents had Spanish copies;
source/notebook scans and whitespace checks passed. Its full suite was interrupted
when the user revised the design and never established a full-suite success.
These facts preserve the earlier evidence, not validation of the final hierarchy.
