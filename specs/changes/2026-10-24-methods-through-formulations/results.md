# Execution results

The user accepted the revised proposal and requested execution on 2026-10-24.
The function-based methods, doubled workspace helper, solver-owned nonlinear
configuration and observer-owned passive energy are implemented and verified.
The accepted scope is complete, with the existing notebook and hardware
limitations described below.

## Implemented behavior

Execution binds `f` and optional derivatives. The standalone FC splitting
helper receives exact-flow callables explicitly. Methods retain no problem or
dynamics object and work with ordinary callables and independent initial conditions. `SingleFormulation` is removed; only doubled methods use
`DoubleFormulation`. Solvers own algebraic tolerances, iteration limits,
derivative policy, backtracking and iteration callbacks. Adaptive integration
accuracy controls remain with the adaptive algorithm.

`simulate(..., observers=(EnergyObserver(),))` enables passive energy. Execution
creates fresh observer state, binds H and its time derivative, and preserves
original-source event identity. Physical steps emit optional quadrature data or
an accepted interpolant; they have no energy momentum input or result. Interior
samples and diagnostic map replay do not change accepted observer state. The
compiled JAX loop uses a separate functional accumulator, including supplied
nonzero initial momentum. Local Modal jobs serialize the new configuration.
See the [migration guide](../../../docs/simulation/api-migration.md) for usage.
Archive schemas and numerical labels are retained; descriptive formulation,
derivative and observer-owner metadata reflects the new responsibilities.

## Numerical comparisons

The actual pre-change source was frozen in `/tmp/gc2d_functions_original_src`.
The main matrix uses `scripts/capture_physical_state_baseline.py`: seed 43,
two particles, GC and FC, interval `[0.15, 0.255]`, step 0.02 and eight output
times, including interior samples and a clipped last step. Each backend has
54 cases and 789 arrays. Supplements add solver policies, analytic/finite
differences, projections, signed maps, coupling and event snapshots.

| Before/after comparison | Bitwise equal common arrays | Largest energy difference |
| --- | ---: | ---: |
| SciPy main | 771 / 789 | 1.734723475976807e-18 |
| JAX CPU main | 769 / 789 | 1.734723475976807e-18 |
| SciPy supplement | 3591 / 3605 | 2.168404344971009e-19 |
| JAX CPU supplement | 453 / 467 | 8.673617379884035e-19 |

All physical trajectories, accepted grids and numerical work counts match their
own pre-change backend exactly. Common step, stage, Newton and replay snapshots
and all 48 signed spatial-map arrays also match exactly. The SciPy supplement
adds 972 quadrature arrays to events intentionally. Removed packed map momentum
is compared through its physical slice and the separate energy observer.
Energy differences come from floating-point regrouping of quadrature weights.
GC keeps signed half weights and the previous reduction order, including its
overflow behavior. All 76 checked values across 13 effective default profiles
are unchanged; configured tolerances are not physical-error estimates.

Cross-backend maxima remain the same as before: physical values
5.551115123125783e-16 and energy 8.326672684688674e-17. These satisfy the established
trajectory bounds `(rtol=3e-10, atol=3e-11)` and energy bounds
`(rtol=3e-9, atol=3e-11)`. The existing FC Radau accepted-grid difference
4.7548927017970755e-9 and FC HBVM field-count difference 64 also remain unchanged.
They compare different backends, not a before/after regression.

## Simplification

Counts include every production `.py` under `src`, including new contracts and
observers; lines include comments/blanks and statements count `ast.stmt` nodes.
There are 304 files both before and after.

| Package | Lines before → after | Statements before → after |
| --- | ---: | ---: |
| contracts | 1748 → 1762 | 1068 → 1071 |
| diagnostics | 8069 → 8116 | 3860 → 3928 |
| dynamics | 433 → 448 | 260 → 273 |
| execution | 1136 → 1217 | 724 → 783 |
| formulations | 78 → 71 | 50 → 47 |
| methods | 4948 → 3713 | 2723 → 2102 |
| solvers | 616 → 702 | 360 → 418 |
| simulation | 222 → 226 | 61 → 65 |
| states | 644 → 644 | 396 → 396 |
| studies | 21205 → 21073 | 10724 → 10758 |
| potential | 1301 → 1301 | 638 → 638 |
| visualization | 10715 → 10715 | 5597 → 5597 |
| problem.py + request.py | 195 → 195 | 99 → 99 |
| **Total** | **51310 → 50183** | **26560 → 26175** |

Net reduction: **1127 lines and 385 statements**. Methods alone lose 1235 lines
and 621 statements. The 53 mirrored method-control declarations are removed,
as are `SolverOptions`, `configured_solver`, method energy helpers and the
single-copy formulation. The small growth in execution, diagnostics, solvers and
contracts implements their new responsibilities. Study statement growth comes
from constructing explicit solver configurations instead of spreading old
method keywords. No compatibility tree replaces the deleted layers.

## Verification and documentation

The final command `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`
completed **700 tests in 618.903 seconds: 697 passed, 3 GPU skips, no failures**.
The earlier complete run exposed four stale test calls (an executor double and
three observer subcases); these were migrated without weakening assertions.
The final log is `/tmp/gc2d_functions_full_tests_verified.log`.

Focused checks pass for method families, dense and particle derivatives,
convergence, energy, reversibility, events, sampling, ownership, lazy imports,
compiled boundaries and local Modal workers. All 304 source files pass
`.venv/bin/python -m mypy src`. Both required examples run successfully, and
`.venv/bin/python -m build` produces the source archive and wheel.

The notebook audit covers 147 current notebooks and 668 parseable code cells;
104 historical notebooks are preserved. Seven ignored development notebooks
needed direct API migration. Tracked experiment notebooks use migrated studies.
Four reduced, in-memory smoke runs pass: BM4 inspection, generalized energy,
single-particle RK4 and single-particle BM4. Canonical scientific parameters and
stored results were not replaced with reduced validation outputs.

Current guides, specifications, project guidance and memory are updated, with
Spanish reading copies. Model documentation includes 16 reviewed diagrams and
18 rebuilt theoretical PDFs. All 40 changed Markdown originals have Spanish
reading copies with matching code blocks, links and checklist states. Memory
remains at 290 lines. `git diff --check` is clean, and
`.venv/bin/python scripts/check_sdd.py` passes (99 records, 22 requirements).
This last validator checks document structure, not numerical correctness.

GPU-specific numerical checks cannot run because no JAX GPU is available. Modal
serialization and workers were checked locally; no paid remote job was run.
Six local development notebooks retain pre-existing coupled-BM4 incompatibilities:
`compare_bm4_with_sdirk4_two_particles.ipynb`,
`study_bm4_mu_vs_displacement_2p.ipynb`,
`recurrences/study_near_center_particle_recurrence_bm4.ipynb`,
`recurrences/study_trajectory_10_recurrence_bm4_tighter_newton.ipynb`,
`recurrences/study_trajectory_10_recurrence_bm4.ipynb` and
`bm4_h5_simple_inspection/calculation.ipynb`, all under
`notebooks/developements/`. The old `BM4Implicit1` and coupling options had
already been retired by commits `771bfd2` and `5a77ff6`; their untouched B2
copies contain the same configurations. Preserving their nonzero coupling
requires separate scientific migration. The obsolete zero-coupling keyword in
`recurrences/study_trajectory_10_four_cycle_segments_bm4.ipynb` was removed
without changing its physics, and its isolated constructor was checked.

Local detailed evidence is in `/tmp/gc2d_functions_*`, including the SciPy/JAX
comparison JSON, default-profile comparison, code inventories, notebook audit
and smoke logs. These temporary captures are development evidence, not new
versioned numerical archives. No commit or push was performed.
