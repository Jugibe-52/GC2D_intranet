# Results

On 2026-11-06 the user authorized preparation and execution of the Hamiltonian naming and derivative change discussed in the conversation. The four preparation documents were presented before implementation.

## Implemented

GC dynamics and study results now expose `hamiltonian_potential`. Drift evaluates the existing Hamiltonian splines directly as `(-H_y, H_x)`; Hamiltonian values, particle Jacobians and time derivatives share that object, including JAX preparation. `Potential.electric_field` and full-cyclotron dynamics retain their behavior. The ABBA diagnostic keeps its historical negative-gradient values and field name.

Current source, tests, one supported experiment notebook and eight active local development notebooks were migrated. Archived notebook copies and saved outputs were preserved. Current documentation and Spanish reading copies were updated; the memory decision stays in the existing entry and MEMORY.md remains at 300 lines. The scoped TFM paragraph describes the new calls, and its root PDF was recompiled successfully without warnings.

## Checks

- Before/after comparison: 42 arrays are exactly equal (maximum absolute difference 0). The synthetic field used `Potential.random(A=0.08, M=3, nx=16, ny=18, seed=27)`, B = 1, 1.7, -2.3 and rho = 0, 0.3. Checks covered drift, H, particle Jacobians, time derivatives, three-time histories and three-particle RK4 trajectories over [0, 0.05] with step 0.01.
- `.venv/bin/python -m unittest discover -s tests -p 'test_gc_hamiltonian.py' -v`: all 3 tests passed, including finite-difference derivative checks, prepared-spline reuse and compiled JAX histories.
- `.venv/bin/python -m mypy src`: passed for 305 source files.
- `examples/gc_orbit.py` and `examples/projected_abba.py`: passed, with 20 RK4 and 4 ABBA steps.
- The supported BM4 notebook was checked in memory with a 16-by-16 grid, M=3, interval [0, 0.03], step 0.01 and 4 saved states. Integration and the last animation frame completed; canonical notebook outputs were not regenerated.
- `latexmk -pdf main.tex` in TFM_redaccion: succeeded; main.pdf is current.
- `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/gc-hamiltonian-matplotlib .venv/bin/python -m unittest discover -s tests -v`: 741 tests ran in 679.872 seconds: 738 passed and 3 GPU tests were skipped because no GPU was available.
- Final SDD validation: passed (169 records, 22 requirements). `git diff --check` passed. All 12 affected English documents have ignored Spanish reading copies, and current notebook sources contain no old attribute references. No changes were made to `src/potential` or full-cyclotron dynamics.

## Limitations

`.venv/bin/python -m build --no-isolation` could not start the build because the existing environment lacks `setuptools` (`setuptools.build_meta` cannot be imported). No dependency was installed. Older pickles containing the renamed configured objects must be recreated, as documented in the API migration guide. Existing numerical archives and the pre-existing HDF5 working-tree change were preserved.
