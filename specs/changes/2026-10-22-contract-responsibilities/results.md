# Execution results

On 2026-10-22 the user requested preparation and execution of the ownership
reorganization discussed in the conversation. The four preparation documents
were presented before implementation; that scope was accepted.

## Implemented

`problem` and `execution.options` now own the two moved classes. Their
implementations match the originals exactly; `simulation.InitialValueProblem`
and the new `execution.ExecutionOptions` export point to the canonical classes.
The old contract modules are removed. Packaging includes `problem.py`.

Comparison and recurrence calculations now live in `diagnostics.comparison`
and `diagnostics.bm4_recurrence`. Callers use functions; shared records retain
validation, independent arrays and archive schemas. The BM4 new-work guard
belongs to `studies._validation`; historical nonzero rates remain readable.
`contracts.arrays` retains its implementation and documents the shared NumPy/JAX
agreement, without a generic utilities module or device policy.

Migrated source consumers, four Modal source bundles and 44 ignored local
notebooks. Notebook edits were confined to source; outputs and backup/historical
copies were preserved. Guides, model API references, current specifications and
Spanish copies were updated. Seven old SDD source metadata entries now point to
the moved files, with notes preserving their historical scope and evidence.
Memory records the confirmed owners at exactly 300 lines; existing execution
knowledge was reflowed without dropping content.

## Observed checks

All commands used the repository `.venv/bin/python` on this machine.

- Preparation: `scripts/check_sdd.py` passed with 67 records and 19 requirements.
- `-m mypy src`: passed, 261 source files.
- `MPLBACKEND=Agg ... -m unittest discover -s tests -v`: 645 tests in 570.543 s,
  no failures; three GPU tests skipped because no JAX GPU was available. CPU
  JAX, mixed arrays, import order and numerical regressions passed.
- Focused diagnostic tests: four passed, covering known RMS values on a
  nonuniform grid, signed energy reductions, periodic wrapping, immutability
  and record imports without diagnostics/study orchestration.
- Package layout tests: eight passed, including alternate import orders without
  optional backends and serialization of the moved classes.
- Following the final Modal bundle update, all 19 tests in
  `test_execution_modal.py` passed. A new test copies exactly each example's
  declared source modules into a disposable folder and runs a serialized job
  without the checkout. No remote endpoint was contacted or deployed.
- Following the historical-coupling assertion update, all three tests in
  `test_study_result_contracts.py` passed. CSV, NPZ and HDF5 loading remains
  independent of studies; a recurrence archive retains coupling_frequency=0.3
  and supports diagnostic reading while new coupled calculations are rejected.
- Both supported smoke examples passed: RK4 ran 20 steps; projected ABBA ran four
  steps with one Newton correction per step.
- `-m build` produced source and wheel distributions. The wheel contains the new
  owners, excludes the retired modules and imports successfully in isolation.
- Notebook validation parsed 166 code cells and executed 46 migrated imports
  across 44 notebooks. Seven actual migrated diagnostic expressions ran against
  a two-step five-method comparison and a three-sample recurrence fixture, with
  finite outputs. No full study or canonical output was executed/replaced.
- `git diff --check` passed. All 44 edited notebooks remain ignored by Git;
  every edited canonical Markdown document has its Spanish reading copy.

An intermediate SDD run reported seven missing historical source paths after the
moves. Only their metadata references and explanatory notes were updated; the
subsequent checker passed (68 records, 19 requirements).

## Limits

GPU execution was unavailable. Deployment examples were checked locally only;
a remote worker must use the same revision as its client before new submissions.
Removed Python module imports and old pickles containing those paths have no
aliases; numerical archive schemas remain unchanged.

Notebook checks covered the changed imports and diagnostic expressions rather
than complete historical workflows. Some local notebooks already contain retired
method names such as ABBA4ImplicitSingleProjection; those unrelated historical
uses were not rewritten. Parsing also reported existing invalid-escape warnings
in mathematical strings. No commits, pushes, deployments or transfers were made.
