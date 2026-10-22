# Results

On 2026-10-22 the user requested preparation and execution of the agreed visible
result-type convention and explicitly selected all of `src/`. The four preparation
documents were presented before implementation.

## Implemented

Reviewed 259 Python source files and changed 217. The tree contains 3,870 additional
annotated assignments: 2,668 with values and 1,202 declarations before assignments.
This includes the small supporting type aliases. Single receiving names are typed
inline; unpacked names are declared before use; reused names keep one declaration
per scope. Parameters, existing annotations, discarded targets and dataclass
fields retain their established handling.

The preparation path now names `jax.Device`, and the array namespace helper names
`ModuleType`. `BackendArray` describes NumPy/JAX arrays without claiming a dtype
or shape; `StepKernel` keeps numerical adapter annotations short. Concrete project
records, host/device arrays, figures, axes and paths replace uninformative inferred
types where their operations establish the result. Dynamic contracts can still
contain `Any`; this is not a complete strict-typing migration.

The convention is documented in `docs/development/typing.md`, the execution guide
and the current simulation specification, with Spanish reading copies. Repository
memory records the decision and contains 299 lines.

## Checks

- `python -m mypy --cache-dir=/dev/null src`: passes, 259 source files.
- A temporary scope-aware syntax audit found no unannotated receiving names in
  the agreed call-assignment/unpacking scope. It also checked 1,493 receiving
  names that remain ordinary assignments because they already have declarations.
- A temporary syntax-tree comparison found identical numerical statements and
  control flow in all 259 files after removing annotations, type-only constructs
  and imports. New runtime imports were reviewed separately: standard-library
  `ModuleType`, an existing dynamics protocol, and JAX types inside the adapter
  that already imports JAX. No numerical expressions were changed.
- `MPLBACKEND=Agg python examples/gc_orbit.py`: passes, 20 steps.
- `MPLBACKEND=Agg python examples/projected_abba.py`: passes, 4 steps and one
  Newton correction per step.
- `python -m build --no-isolation` initially failed because this virtual environment
  lacks setuptools. The normal `python -m build` succeeded using its temporary
  isolated environment and produced both source and wheel distributions.
- `git diff --check`: passes. All nine new or updated Spanish reading copies
  are ignored by Git. No notebooks or stored study data were regenerated.

The full headless suite passed: 641 tests in 559.351 seconds, with three GPU tests
skipped because no GPU is available. CPU/JAX comparisons, optional-backend imports,
serialization, studies and visualization checks passed. It was run with
`MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v`
and a temporary `MPLCONFIGDIR`. GPU numerical equivalence remains untested on this
machine. `python scripts/check_sdd.py` passes: 63 records and 19 requirements.
The accepted implementation and checks are complete.
