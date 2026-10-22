# Execution record

Accepted on 2026-10-22 through the user's request to prepare and execute the
discussed process-based organization, including the project `AGENTS.md` rule.

## Implemented

Relocated diagnostic implementations into `execution`, `methods`, `formulations`,
`states`, `studies` and `io`, with adaptive and extended method subpackages.
All 63 existing public exports still resolve to their canonical implementations.
Former flat module routes are removed, and internal consumers import defining
modules. Numerical processes remain independent of diagnostic implementations.

Updated consumers, documentation, current specifications and the project
`AGENTS.md`, including Spanish reading copies. Older SDD acceptance and evidence
are retained; their checked source references point to the relocated files.
Memory references were corrected within the existing 300-line limit.

Migrated source references in 121 working notebooks: 119 ignored development
notebooks and two tracked experiment notebooks. Stored outputs and metadata
were preserved, and their Git tracking status is unchanged. The 75 historical
`.from-b2-*` conflict copies retain their original module references.

## Verification

Checks ran in the existing `.venv` from the repository root:

- `python -m unittest discover -s tests -p 'test_diagnostic*.py' -v`: 15 passed.
- `python -m unittest discover -s tests -p 'test_package_layout.py' -v`: 11 passed,
  including canonical export identity, removed paths and import independence.
- `python -m mypy src`: no issues in 269 source files.
- `MPLBACKEND=Agg python -m unittest discover -s tests -v`: 649 tests in
  466.535 seconds, successful with three GPU-dependent tests skipped.
- Both `examples/gc_orbit.py` and `examples/projected_abba.py` completed,
  reporting 20 and four steps respectively.
- `python -m build`: source and wheel distributions built successfully.
  The wheel contains all 50 diagnostic Python modules and no retired module
  implementations; all modules and 63 public exports import from the wheel.
- Compared parsed code before and after relocation: computational statements
  in the 42 original diagnostic files and 56 modified existing source consumers
  are unchanged after excluding imports and updating module references.
- Executed the 171 diagnostic import statements in the 121 working notebooks.
  Verified 206 diagnostic imports across source, scripts, examples and tests
  use current paths. No long notebook integrations were run.
- Spanish copies exist and remain ignored; memory remains at 300 lines.
  `git diff --check` passes.
- Final `python scripts/check_sdd.py`: 74 records and 22 requirements pass.

## Limits

GPU execution was unavailable. Notebook verification covered imports; the
numerical and persistence paths were exercised by the test suite and examples.
One working notebook emitted an existing invalid-escape `SyntaxWarning` while
its cells were parsed; its diagnostic imports succeeded. Saved numerical data,
formulas, units, observation timing and archive schemas were not changed.
No implementation work remains within the accepted scope.
