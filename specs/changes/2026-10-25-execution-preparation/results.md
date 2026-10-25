# Results

On 2026-10-25 the user accepted the four-helper refactoring proposed in the
conversation and requested SDD preparation and execution together. The four
preparation documents were presented before implementation.

## Implemented

Four private preparation helpers now live beside `Execution`. Its `run` method
occupies 40 lines instead of 84, including signature, comments and blank lines.
The lifecycle, worker selection and recording/finalization boundary remain
visible. Descriptive local names and typed returns clarify the preparation data.
The execution guide and current simulation specification include the boundaries,
with corresponding local Spanish copies.

A new compiled-step regression confirms that preparation runs while the job is
active, propagates failure before result allocation, finishes only the consumed
run, rejects its reuse and leaves the method configuration reusable.

## Checks on 2026-10-25

Commands used the existing `.venv/bin/python` (Python 3.11.10).

- Before editing, `python -m unittest discover -s tests -p 'test_execution.py' -q`: 13 tests passed.
- `PYTHONPATH=src:tests MPLBACKEND=Agg python -m unittest test_execution test_compiled_step_contract test_energy_observation test_function_methods -q`: 36 tests passed, including the new failure regression.
- `python -m mypy src`: passed for 304 source files.
- `MPLBACKEND=Agg python -m unittest discover -s tests -v`: completed 701 tests in 712.274 seconds, with no failures and three GPU tests skipped because no GPU was available.
- `python examples/gc_orbit.py` and `python examples/projected_abba.py`: completed with 20 and 4 steps respectively; the projected example reported one Newton correction per step.
- `python -m build`: built the source distribution and wheel.
- Direct comparison against `src/execution/execution.py` at commit `ac9badc`: all 16 cases matched exactly in physical states, saved times, initial snapshots, layouts and every diagnostic. CPU callback events also matched. The matrix used RK4, ABBA2Implicit, GaussLegendre4 and DOP853, SciPy/JAX CPU, and energy observation disabled/enabled. It used a two-particle GC problem with `Potential.random(A=.01, M=2, nx=12, ny=12, seed=19)`, initial array `[.2, .4, .3, .5]`, interval `(0., .2)`, output times `[0., .03, .1, .2]`, step/max_step `.1` and JAX 64-bit arithmetic. CPU callbacks requested quadrature in both energy modes. The original coordinator was loaded in memory from Git for comparison.
- Preparation document validation: `python scripts/check_sdd.py` passed with 103 records and 22 requirements.
- Document validation including the results record: `python scripts/check_sdd.py` passed with 104 records and 22 requirements. `git diff --check` passed.

The planned refactoring and checks are complete. GPU execution and a live remote
Modal job were not exercised; local Modal contract tests passed in the complete
suite. No numerical, public API or saved-data migration is required.
