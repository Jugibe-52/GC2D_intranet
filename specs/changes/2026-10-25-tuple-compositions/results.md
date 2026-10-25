# Execution results

The user accepted preparation and execution of tuple recipes, consecutive time
translation merging and code simplification on 2026-10-25. The four preparation
documents were presented before implementation. On the same date, the user
explicitly authorized declaring fixed tuples directly on method classes and
removing their dataclass fields.

## Implemented behavior

Composition is now the public tuple[Operation, ...] type alias. All four recipes
are tuple literals on the method classes in abba.py and bm4.py, without recipe
fields. ABBA2Midpoint shares ABBA2Implicit.recipe; BM4Implicit shares
BM4Midpoint.recipe. ABBA2 is `(A(.5), B(.5), T(1.), B(.5), A(.5))`.
Method preparation validates the tuple once. Preset names remain ABBA2/ABBA4/
ABBA6/BM4; generic methods accept composition_name, defaulting to "custom".
The old constructor and factories are removed, with migration in the operation
guide. No notebook used those constructors.

Consecutive T operations are combined, giving 5/15/35/30 elementary operations.
Spatial shears and passive energy quadrature remain separate. The numerical
trace no longer contains grouped MapStage records. ABBA/BM4 observers derive
public groups from retained shears without replay; projection equations,
solver controls, public event shapes and source counts are preserved.

The example, current simulation contract, shared model guide, seven theoretical
sources/PDFs, the BM4 implementation reference and both ABBA2 diagrams were updated. Markdown Spanish copies are
aligned and ignored. Memory records the new convention in 300 lines.

## Numerical comparison

Before source edits, a temporary reference captured 5,832 arrays/scalars across
six methods: physical histories, passive momentum and public observation fields.
The fixture is tests.test_extended_family.problem(3), using Potential.random
with A=0.08, M=3, nx=ny=16, seed=27, interpolation_order=5, rho=0.05;
x spans [1,1.2], y spans [1.2,1.4]. The interval is [0.3,0.5], step 0.02,
with 11 equally spaced outputs and the existing default solvers.

| Method | Max physical discrepancy | Max momentum discrepancy | Max event-field discrepancy |
|---|---:|---:|---:|
| ABBA2Implicit | 0 | 0 | 0 |
| ABBA4Implicit | 0 | 0 | 5.55e-17 |
| ABBA6Implicit | 0 | 8.67e-19 | 1.67e-16 |
| ABBA2Midpoint | 0 | 0 | 0 |
| BM4Implicit | 0 | 8.67e-19 | 1.11e-16 |
| BM4Midpoint | 0 | 8.67e-19 | 1.11e-16 |

These are measured differences, not solver tolerances or a long-time accuracy
claim. The comparison required identical field names/shapes and absolute
agreement within 5e-13. Independent split/merged-clock tests use both h=0.15
and h=-0.15, comparing states, tangents and quadrature sources within 3e-15
(and time to 14 decimal places). Existing rotation tests retain second-order
convergence for center/end time placement; generic NumPy/JAX agreement passes
at absolute tolerance 3e-12.

## Checks

- Final focused run: 26 tests passed, covering the extended family, function
  palindromes, shared ABBA, implicit BM4 and the fixed-recipe subclass.
- All six fixed recipes are outside dataclasses.fields and retain the same
  shared tuple through construction, dataclasses.replace and pickle. The
  six-method reference comparison was repeated after moving the declarations
  to their classes, with the discrepancies reported above.
- All 717 discovered tests have confirmed final outcomes across the headless
  runs: 714 passed and three GPU checks were skipped because no GPU is available.
  The original `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`
  run ended with SIGTERM before its summary; interrupted blocks were resumed,
  and individual test identifiers were reconciled with current discovery.
  This is combined coverage, not one uninterrupted full-suite pass.
- During validation, two outdated test setups were corrected: missing
  output_times and an old BM4.operations reference. The subprocess test could
  not spawn from the temporary stdin runner; its normal module run passed all
  three tests (`.venv/bin/python -m unittest tests.test_potential_snapshot -v`).
- Source typing passed for all 304 files with `.venv/bin/python -m mypy src`.
  gc_orbit.py, projected_abba.py and palindromic_time.py completed, and
  `.venv/bin/python -m build` produced source and wheel distributions.
- Seven theoretical PDFs compiled twice without warnings or overflowing boxes.
  Their first two pages and both updated ABBA2 diagrams were visually inspected.
  After updating recipe source locations, the three affected PDFs were compiled
  twice again and their changed pages inspected. Text is legible and unclipped.
- All ten changed or new Markdown originals have ignored Spanish copies with
  matching code blocks and resolving local links. Memory remains at 300 lines.
  `git diff --check` and `.venv/bin/python scripts/check_sdd.py` pass;
  SDD validation reports 124 records and 22 requirements.

No work remains in this change. No long notebook, remote run or saved scientific
archive was regenerated. GPU execution was not tested.
