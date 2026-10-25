# Shared fixed-step algorithms

Accepted on 2026-10-25: implement [the proposal](proposal.md), preserving the adaptive implementations. Implementation and verification are complete.

Fixed methods now use one `advance`, stage, composition and residual implementation for NumPy and JAX. Potential and dynamics own cached `_as_jax()` preparation; optional `JaxNewtonSolver` and `JaxBroydenSolver` inherit ordinary controls. Execution prepares and compiles the common step. The four alternate method-owned JAX modules and fixed `jax_mode` declarations are removed. Compact array loops avoid expanding long compositions; analytic particle solves, including HBVM, retain small independent systems. Reduced 2N and simultaneous 6N projection equations remain independent of execution choice.

## Numerical comparisons

A frozen copy of source commit `f859efefe3447e31bf56119e59b9a06b95a6b128` supplied 128 GC/FC reference runs. The problems come from `JaxMethodTests.setUpClass`: three particles, potential seed 27, float64, GC rho=0.05 and FC rho=0.3/eta=-0.4. The comparison uses step 0.05, interval [0.3, 0.423], saved times [0.3, 0.305, 0.35, 0.371, 0.423], every supported fixed family, both projections and solvers, analytic/finite differences, energy off/on, and DOP853/Radau with max_step=0.05.

- All 64 NumPy runs are exactly equal in trajectories, energy, times, diagnostics and callback counts, including after preserving HBVM particle solves.
- Across 64 JAX runs, the largest trajectory change is 1.74e-18 and energy changes are zero. Times and iteration counts are identical. Residual norms differ by at most 1.43e-17, consistent with floating-point reassociation.
- Another 28 generic-recipe runs are exactly equal numerically, including CPU observation contents. They cover `T(.5), [C(.5)], A(.5), B(.5)` reflected into a palindrome, optional coupling 0.7, both projections and solvers, two particles, seed 81, step 0.04 and interval [0.2, 0.253]. All ten forced nonconvergence cases retain their RuntimeError behavior.

Comparison limits were rtol=3e-10/atol=3e-11 for trajectories and rtol=3e-9/atol=3e-11 for energy, with exact checks for unchanged arithmetic. These are measured comparisons, not physical accuracy guarantees. Independent convergence, conservation, projection and tangent tests remain in the suite.

The deliberate diagnostic difference is JAX finite-difference batching: global-coordinate probes now match the common NumPy derivative target. Metadata reports `vectorized_global_columns`. With three particles, HBVM field calls per step change from 29 to 61 for GC differences and from 45 to 109 for FC differences; residual and correction counts stay equal. These dense differences require O(N²) matrix storage. Analytic declared particle blocks retain small independent solves; full Broyden remains dense.

## Checks and costs

Focused solver, component, execution, shared-algorithm and numerical suites passed. Final focused reruns include 8 shared-algorithm/solver tests, 7 function-method tests and 6 HBVM tests. They cover compiled rejection of invalid initial Broyden data even with zero residual. Type checking passed for 304 source files; both public examples and source/wheel builds passed. The SDD validator passed (119 records, 22 requirements), including its 35-test suite. The full suite passed: 715 tests in 1043.409 s, with three GPU skips. This includes archive round trips, local Modal workers, studies and visualization contracts. The commands are recorded in [steps](steps.md).

For the same three-particle GC case with energy enabled, complete JAX CPU calls measured:

| Method | First call | Median of three warmed calls |
|---|---:|---:|
| RK4 | 4.645 s | 2.070 ms |
| ABBA4Implicit | 8.531 s | 8.492 ms |
| HBVM42 | 9.170 s | 3.508 ms |

These synchronized end-to-end timings include preparation, transfers and result construction. The problem is reused across methods. Hardware was Intel i3-1215U, Python 3.11.10, NumPy 1.26.4 and JAX 0.7.1; other validation was running, so these are illustrative costs, not a controlled speedup comparison. No GPU is available; GPU equivalence remains unverified.

Initial broad checks were interrupted to replace slow expanded composition compilation with compact common loops. Intermediate typing/import issues were corrected. Final source review preserved analytic HBVM particle solves and strengthened Broyden input validity; their focused tests and affected baselines were rerun. Full-suite execution started before those final local refinements; the final focused reruns validate the affected code.

DOP853/Radau implementation remains byte-identical (SHA256 `eac0395308f8a5393108f88cbed5190cd08bc7dae4399bcf9208cabd7ad316a1`). Production Python across methods, solvers, potential, dynamics, contracts and execution decreased from 9,287 to 9,134 lines. Guides, current contracts, 13 fixed-model diagrams and memory are updated, with Spanish copies; adaptive diagrams and mathematical theory are unchanged. No direct notebook consumers of the retired private modules were found. Local Modal worker/serialization checks passed; no remote jobs were launched. Historical SDD source references use the existing move table without altering old acceptance or evidence.
