# Results

On 2026-10-26 the user accepted the proposed execution simplification with full
typing and requested SDD preparation and execution together. The four preparation
documents were presented before implementation.

The working tree was initially clean.

## Implemented

Backend preparation now resolves the device and mode directly and binds hybrid
dynamics without `prepare_environment`. The coordinator calls `prepare_step`
directly inside its lifecycle guard; that function owns fixed-JAX observation
restrictions and checks the prepared run type without redispatching its mode.
Observer and numerical preparation read the request from the problem.
All source annotations and optional imports are retained. No context class,
numerical formula, cache key or public API was introduced or changed.

Two direct-preparation regressions cover callbacks, progress, iteration observers,
adaptive methods and unsupported subclass overrides. The execution guide and
current simulation specification describe the new boundaries. The memory skill
recorded the confirmed ownership decision in `MEMORY.md`, still 300 lines.
All affected Markdown documents have updated local Spanish reading copies.

## Checks

Commands use the existing `.venv/bin/python`, Python 3.12.3 and JAX 0.7.1 on CPU.

- `python -m mypy src`: passed for 303 source files.
- `PYTHONPATH=src:tests MPLBACKEND=Agg python -m unittest test_execution test_compiled_step_contract test_energy_observation test_jax_rk4 -q`: 40 tests in 47.869 seconds, no failures, one GPU test skipped.
- `MPLBACKEND=Agg python -m unittest discover -s tests -v`: 723 tests in 692.512 seconds, no failures, three GPU tests skipped. This includes fixed and hybrid JAX, NumPy/SciPy, import isolation, lifecycle failures, cache reuse and local Modal contracts.
- `python examples/gc_orbit.py`: completed with 20 steps.
- `python examples/projected_abba.py`: completed with four steps and one Newton correction per step.
- `python -m build`: built the source distribution and wheel.
- Preparation SDD validation passed with 148 records and 22 requirements; validation including results passed with 149 records and 22 requirements. `git diff --check` passed.

The agreed implementation and checks are complete. GPU execution and a live
remote job were not exercised; only a JAX CPU device is available. No numerical
archive migration is needed.
