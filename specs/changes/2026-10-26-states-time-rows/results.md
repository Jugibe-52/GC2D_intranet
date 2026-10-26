# Implemented and checked

On 2026-10-26 the user authorized preparation and execution of time rows in States and migration of its consumers.

States owns contiguous physical rows `(samples, state_size)`: `states[i]` and `values[i]` correspond to `t[i]`. Construction, streaming/bulk recording, selection and completed-result pickling follow that contract. Component accessors retain `(particles, samples)` for vectorized physics. CPU energy diagnostics and compiled recording convert explicitly at their boundaries; integration formulas and tolerances are unchanged.

General solution archives now use schema 2. The reader converts schema 1 columns explicitly, even for square arrays, without rewriting existing files. Supported older States pickle payloads also decode their original orientation. Specialized study reference arrays and archive schemas retain their documented component-first conventions. Studies, plots, examples, notebook generators and 16 active notebooks were adapted. Historical executed notebooks, backup copies and existing experiment artifacts were preserved.

## Verification

- Full discovery (`MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`) ran 717 tests in 632 seconds, with 3 GPU skips. During the initial migration it reported 6 failures and 3 errors across 8 cases. All 8 were corrected and passed in a final 25-test rerun that also includes the 17 row/lifecycle tests. This is a consolidated result, not a second clean full-suite run.
- Additional focused runs passed: 70 execution, persistence, geometry and trajectory tests; 51 selection, study and plotting tests; 17 ownership/study tests; and 5 energy/JAX tests. Final coverage includes CPU/JAX sparse sampling and particle independence, GC/FC histories, one-sample and square archives, version disagreement, read-only ownership and old pickle restoration. Temporary logs are in `/tmp/gc2d-time-rows-*.log` on this machine.
- `.venv/bin/python -m mypy src`: success for 303 source files. `git diff --check`: passed.
- Syntax checked all 16 migrated notebooks. Executed import/setup and calculation/plot cells 1, 3 and 4 of `notebooks/developements/bm4_simple_inspection.ipynb` in memory, with time span `(0, 0.2)` and 3 samples: shape `(3, 2)` and initial-row equality passed. No long notebook was executed; no result was published. Existing invalid-escape warnings in another notebook and the expected headless plotting warning did not prevent verification.
- `.venv/bin/python -m build --outdir /tmp/gc2d-time-rows-dist`: source and wheel distributions built successfully. The first attempt without isolation could not find setuptools; the standard isolated build passed without changing project dependencies.
- `.venv/bin/python scripts/check_sdd.py`: passed (144 records, 22 requirements). Updated current specifications, architecture/API guides and their Spanish copies. Updated the existing memory entry and Spanish copy; `MEMORY.md` remains 300 lines.

GPU execution could not be checked on this machine. No implementation work remains within the accepted scope. No TFM files were changed, so no thesis recompilation was needed; no commits or uploads were made.
