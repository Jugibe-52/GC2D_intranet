# Execution results

Accepted on 2026-10-24: implement the presented physical-state and generic-workspace proposal, including coordinate ownership, separate passive momentum, method-owned maps/projections, consumers, documentation and numerical compatibility. The starting working tree was clean at `efe2436`; its method-owned step behavior is the baseline. No commit or push was made.

## Implemented behavior

Every accepted state now contains one physical vector `(d*N,)`, scalar time and optional separate physical momentum `(N,)`. Controllers, step records, shadow samples, adaptive dense output and compiled JAX carry these explicitly. Initial momentum is zero; supplied nonzero momentum persists. Double copies exist only during composition/projection and never accumulate across steps.

`SingleFormulation(physical_size)` and `DoubleFormulation(physical_size)` retain only size and prepare/split complete temporary copies, including shifted candidates and FC velocity blocks. They preserve NumPy/JAX without importing numerical methods or physics. GC/FC map implementations belong to `methods.extended.maps` and `methods.splitting`; projection equations/contracts and projected diagnostics belong to their method domains. Retired specialized layouts, state containers and wrappers have no compatibility aliases.

Dynamics declare immutable component names and position/velocity roles. Initial conditions and results own those structural descriptions and arrays independently of live physics. Problems and output validation compare order and roles as well as dimensions; an undeclared external system requires explicit coordinates. Generic external three/five-component cases, complete blocks and sample axes are covered. Existing GC/FC archive schemas and historical metadata remain readable; old configured-object pickles must be recreated.

Methods evaluate saved H once; `states.energy` combines supplied H with physical kappa. GC split work retains its summed-momentum convention and divides by two on projection. Solver equations, controls, physical-only error criteria, stage reuse, observation and failure behavior are unchanged.

## Numerical comparison

The reproducible capture procedure is [capture_physical_state_baseline.py](../../../scripts/capture_physical_state_baseline.py). Run it on the baseline and final sources, placing the corresponding `src` first on `PYTHONPATH`, using the same environment:

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/capture_physical_state_baseline.py /tmp/physical-scipy --backend scipy
JAX_ENABLE_X64=1 .venv/bin/python scripts/capture_physical_state_baseline.py /tmp/physical-jax --backend jax
```

Each backend covers 54 cases and stores 789 arrays plus scalar diagnostics/controls: the exact field, GC/FC inputs, nonzero initial time, irregular saved times and shortened final step prescribed in [steps.md](steps.md); energy off/on, classical/adaptive/projected methods, Newton/Broyden and supported reduced/simultaneous projections. JSON records retain actual method and solver settings. The environment was Python 3.11.10, NumPy 1.26.4, SciPy 1.17.1 and JAX 0.7.1, with JAX CPU float64.

| Comparison | Measured result |
|---|---|
| Before/after NumPy/SciPy | All 789 arrays bitwise equal |
| Before/after JAX CPU | All 789 arrays bitwise equal |
| Direct/adjoint GC/FC maps | All 48 captured arrays bitwise equal |
| CPU/JAX maximum physical trajectory difference | `5.551115123125783e-16` |
| CPU/JAX maximum H, H+kappa or kappa difference | `8.326672684688674e-17` |
| CPU/JAX maximum accepted-grid difference | `4.7548927017970755e-09`, FC Radau; identical discrepancy before the refactor |

Direct/adjoint comparisons include signed substeps `+/-0.02`, perturbed copies, nonzero momentum, GC coupling disabled/enabled at 0.3, tracking disabled/enabled and reversals. Independent convergence/reversal tests remain active. Physical comparisons pass `rtol=3e-10, atol=3e-11`; energy comparisons pass `rtol=3e-9, atol=3e-11`. These are cross-backend agreement limits, not bounds on physical trajectory error.

Within each backend, accepted grids, energy histories and numerical work are unchanged. Only the intended accepted/workspace dimension metadata changes: 40 cases change `accepted_internal_state_dimension` and 27 change `base_splitting_state_dimension`. Explicit workspace/time/momentum sizes are added. Runtime timing metadata is not expected to be identical. The CPU/JAX HBVM FC vector-field evaluation difference of 64 evaluations per step was already present in the baseline; it is not a new discrepancy. Capture data remain under `/tmp/gc2d_physical_{before,after}_{scipy,jax}.{npz,json}` and `/tmp/gc2d_split_{before,after}.npz` for this session.

## Consumers and documentation

Audited 251 local notebooks, including ignored development material. Migrated source cells in 21 working notebooks (2 tracked, 19 ignored); compiled 153 code cells. Ran reduced in-memory subsets of `single_particle_orbit`, `single_gc_bm4_orbit` and ignored `bm4_simple_inspection`: 16x16 random fields, two modes, interval `[0,0.021]`, five samples. The measured-field development example used a deterministic synthetic field solely for this API check. Original notebook outputs and settings were not executed or overwritten. The 102 archival/backup notebooks, including `from-b2` and stored executed results, retain their provenance.

Updated public examples, study/diagnostic consumers, current specifications, guides and Spanish reading copies. Rebuilt 15 canonical theory PDFs with two LaTeX passes, with no overfull boxes or undefined references; inspected rendered affected pages. Regenerated eight compact extended-family diagrams and reviewed the shared layout. Historical detailed/global diagrams were not regenerated. Repository memory records the confirmed contract (291 English lines, 300 Spanish lines).

The SDD checker resolves the two retired coordinate-source paths only in historical change records, against existing current destinations. Current specifications still reject retired or missing paths; 35 checker tests verify this and the existing rules. Old change records were preserved.

## Verification and limits

- Final focused run: 52 tests passed, including isolated archive loading with dynamics and potential construction/loading prohibited, new snapshots, signed maps, palindromic convergence and SDD history validation.
- Full headless suite: 664 tests in 703.264 s: 661 passed and 3 GPU-only skips. After a final ownership review, adaptive step results were detached from the live SciPy state; 23 adaptive/ownership tests passed in 1.951 s and the repeated 54-case CPU capture again matched all 789 baseline arrays exactly.
- Source mypy: passed, 304 source files.
- `examples/gc_orbit.py` and `examples/projected_abba.py`: passed (20 and 4 steps respectively).
- Package build: wheel and source archive passed; wheel inspection confirms current modules and no retired formulation modules.
- `git diff --check`: passed. Final SDD validation: 94 records and 22 requirements passed.

The first broad test run exposed stale dimension/import fixtures and a missing local `pandas` test dependency. Fixtures were migrated and pandas 2.x was installed in the project environment before the final run. Earlier package failures also exposed cache-only retired directories; those generated caches were removed. No numerical mismatch was concealed by changing tolerances.

GPU hardware is unavailable, so its three conditional checks are skipped. No remote/cloud execution, long scientific notebooks, deployment or performance claim is included. Numerical archives preserve compatibility; direct callers of retired layout/wrapper APIs must migrate as described in the [API guide](../../../docs/simulation/api-migration.md).

Reproduce the repository checks from its root:

```bash
MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v
MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_physical_workspaces tests.test_adaptive_integration tests.test_state_ownership
.venv/bin/python -m mypy
MPLBACKEND=Agg .venv/bin/python examples/gc_orbit.py
MPLBACKEND=Agg .venv/bin/python examples/projected_abba.py
.venv/bin/python -m build --outdir /tmp/gc2d-physical-state-build
.venv/bin/python scripts/check_sdd.py
```

The accepted scope is complete. Final local logs are `/tmp/gc2d_physical_suite2.log`, `/tmp/gc2d_adaptive_ownership_final.log`, `/tmp/gc2d_physical_final_focused.log`, `/tmp/gc2d_physical_mypy_final.log` and `/tmp/gc2d_physical_build_final.log`. The final CPU recapture is `/tmp/gc2d_physical_final_scipy.{npz,json}`. These temporary files are session evidence, not required runtime inputs.
