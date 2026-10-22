# Execution results

Accepted on 2026-10-22: relocate existing pure contracts and reorganize all
existing contracts by source domain, with consumer and documentation migration.
The user requested preparation and execution together for the agreed scope.

## Implemented

Moved 37 Protocol interfaces and two TypedDict declarations from implementation
packages. All 40 Protocol and two TypedDict declarations in `src` now live under
`contracts`. Existing records and aliases follow the same domain hierarchy;
method observations and study results have their own family/domain modules.
See the [complete map](../../../docs/simulation/contracts.md).

Updated canonical imports, worker packaging, current specifications, architecture
text, eight active extended-family diagrams, and Spanish reading copies.
All seven public facade export lists are unchanged. Comparison-only type imports
no longer load implementations. Cloud bundles contain the complete contracts
package, while the transfer process imports only standard-library worker contracts.
Project memory records this ownership rule and remains at 300 lines.

## Verification

- Compared syntax trees: all 77 relocated class/function definitions are unchanged.
  Scanned all source declarations: zero Protocol/TypedDict definitions remain outside contracts.
- `.venv/bin/python -m mypy src`: passed for 305 source files.
- `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`: 654 tests,
  successful in 459.800 seconds; three GPU checks skipped because no GPU is available.
- After final import and bundle adjustments, reran 29 focused tests covering
  package imports, record persistence, validation, and contract ownership: all passed.
  The cloud archive test checks both a standard-library-only transfer process
  and a scientific process using the complete bundled package.
- Both `examples/gc_orbit.py` and `examples/projected_abba.py` passed: 20 RK4 steps
  and four ABBA steps respectively, with one Newton correction per ABBA step.
- Ran cells 2, 4, 6, and 8 of the two basic orbit notebooks in memory, including
  animation: grid 12 by 12, maximum wave number 2, interval `[0, 0.02]`, three
  saved samples and frames. Both passed; original notebook hashes are unchanged.
- `.venv/bin/python -m build --outdir /tmp/gc2d-domain-contracts-dist`: source and
  wheel distributions built. All 48 contract Python files are packaged, all 47
  contract submodules import from the wheel, and removed module paths are absent.
- Regenerated the eight active extended-family diagrams and visually checked the
  shared diagram. `git diff --check` and the SDD document checker passed.

An initial layout assertion incorrectly required `contracts.execution` to be
absent; it now checks the new domain package without restoring retired options.
The first import scan also touched 125 generated frozen-cache files. Those exact
mechanical changes were restored from their recorded source commit; all 365
cached source files match their recorded hashes. Frozen snapshots and notebook
content remain unchanged. Historical specification edits only repair source
metadata; their acceptance, prose, and results remain intact.

## Limits

GPU behavior was not exercised, and cloud checks were local without submitting
remote jobs. Numerical archive schemas are unchanged; Python pickles tied to
removed module paths require regeneration. No implementation work remains.
