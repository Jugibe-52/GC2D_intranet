# Results

Accepted on 2026-10-27 through the user's request to prepare and execute the
discussed separation of fixed-JAX compilation from `run_steps`.

`Execution.run` now calls `prepare_step` and `prepare_integration` before
allocating results. Preparation completes whole-loop lowering/compilation and
returns `PreparedIntegration`; the worker executes and records it. Executables
are cached separately from fresh input values, with a bound of 16 signatures.

The 12 compiled-step contract tests passed, including separation, changed input
shapes, fresh initial values, bounded reuse and compilation-failure lifecycle.
Source type checking passed for all 303 files. Six pre-change GC/FC simulations
match exactly in physical states, times and every diagnostic; the maximum
physical-state discrepancy is 0.0. Both supported examples passed. The complete
test suite passed: 726 tests in 677.494 seconds, with three skips and no failures.

The first package build without isolation failed because this virtual environment
has no setuptools. The standard isolated build subsequently succeeded. Updated
guides, specifications, model descriptions and memory have Spanish reading copies;
memory remains at 300 lines. The TFM subsection was updated and its root PDF
compiled successfully. SDD validation passed with 154 records and 22 requirements;
`git diff --check` is clean in the code repository. All planned work is complete.
Changes remain local, without commits, publication or synchronization.

## Reproducing the checks

Use `.venv/bin/python` (Python 3.12.3, JAX 0.7.1) on the detected CPU with
`jax_enable_x64=True` for numerical JAX checks. Commands used:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_compiled_step_contract.py' -v
PYTHONPATH=tests .venv/bin/python -m unittest test_compiled_step_contract.CompiledStepContractTests.test_executable_reuse_keeps_fresh_values_and_specializes_shapes -v
.venv/bin/python -m mypy src
MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v
MPLBACKEND=Agg .venv/bin/python examples/gc_orbit.py
MPLBACKEND=Agg .venv/bin/python examples/projected_abba.py
.venv/bin/python -m build
.venv/bin/python scripts/check_sdd.py
```

The extra reuse check passed after extending it to change the interval and
shadow-sampling times without changing their array shapes. It verifies that the
shared executable consumes the new values as well as the new initial state.

The direct baseline used pre-change commit
`36eb4f0f751ee3e618883bf02f19be662634447d`. A synthetic potential used
`Potential.random(A=.1, M=2, nx=12, ny=12, seed=74)`, `rho=.1`, and FC `eta=-.4`.
All six cases used step `.1`, interval `(.3, .53)` and samples
`[.3, .301, .4, .47, .5, .53]`, including independent shadow samples and a shortened
last step. GC started at `[.2, 1.1, .3, .7]`; FC appended velocities
`[.01, -.02, .03, .01]`. Cases were GC RK4 with energy off/on, GC GaussLegendre4
and ABBA4Implicit with energy, and FC RK4 and GaussLegendre4 with energy.
Default solver controls were unchanged. The local comparison script is
`/tmp/gc2d-compile-baseline.py` (`save` before editing, `compare` afterwards);
its records and logs are temporary validation artifacts.

No GPU is available, and no remote deployment or long notebook run was performed.
The public Modal API and result archive formats are unchanged. Compilation is
process-local and existing public entry points perform it in the worker.
