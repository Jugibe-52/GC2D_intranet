# Execution results

The user accepted the common palindromic architecture scope and authorized preparation and execution on 2026-10-21. The four documents were presented before implementation. The separate completed uncoupled-BM4 record was preserved.

## Implemented behavior

All six ABBA/BM4 presets inherit PalindromicMethod and share their implicit or midpoint advance. Immutable A/B/T/C operations and Composition.from_half define the actual chronological execution on NumPy and JAX. Only T moves the clock; coefficients multiply h directly. Generic PalindromicImplicit and PalindromicMidpoint accept experimental recipes, including explicit C with an independent HarmonicCoupling. BM4 presets contain no C.

Exact particle tangents and passive energy use the retained elementary trace. Existing grouped observations, solver controls and projection placement remain supported. The guide and runnable example explain center/end time placement and recipe-only subclasses. All affected model guides, compact diagrams, seven theoretical PDFs, current contracts and Spanish reading copies were updated. Project memory records the architecture and scalar configuration metadata convention (283 lines).

## Numerical evidence

Six pre-change three-particle trajectories and passive momentum histories were saved before implementation. For Potential.random(A=0.08, M=3, nx=16, ny=16, seed=27), rho=0.05, t in [0.3, 0.34] and maximum step 0.02, the maximum absolute discrepancy was exactly 0.0 for every named method, in both spatial and momentum histories.

The new tests independently replay the end-time recipe, verify signed reversal with and without C to absolute tolerance 3e-15, and compare exact C-containing particle tangents with centered differences (relative tolerance 2e-8, absolute 2e-9). Seven generic configurations check NumPy/JAX physical states and, when enabled, passive momentum to absolute tolerance 3e-12.

On the exact nonautonomous rotation f(t,z)=(1+t+t²)Jz, z(0)=(1,0.2), t=0.8, the maximum component errors were:

| Projection and T placement | h=0.1 | h=0.05 | h=0.025 |
|---|---:|---:|---:|
| Implicit, center | 3.10761e-3 | 7.70275e-4 | 1.92154e-4 |
| Implicit, ends | 1.05155e-3 | 2.60657e-4 | 6.50236e-5 |
| Midpoint, center | 3.10661e-3 | 7.70252e-4 | 1.92154e-4 |
| Midpoint, ends | 1.05109e-3 | 2.60646e-4 | 6.50234e-5 |

Both spatial error and passive energy balance decrease by factors between 3.5 and 4.5 under halving, consistent with second order. These are measured errors, independent of the nonlinear stopping tolerances; they do not establish arbitrary reordered recipes as fourth- or sixth-order methods.

## Checks and corrections

Focused ABBA tests (103), BM4 tests (21), core family tests (6), lifecycle tests (7) and block-study tests (4) passed. The first complete 628-test run reported two integration errors: a test patched the former projection owner, and block-study collection interpreted array-valued recipe coefficients as a step history. The patch target now follows the common base and coefficients are scalar JSON metadata. Both affected modules passed after correction. A new C call-count check was also corrected to inspect one explicit advance, avoiding a scheduler roundoff remainder step.

The final complete headless run finished successfully: 628 tests in 504.135 seconds, with three GPU tests skipped and no failures or errors. Typing passed for all 258 source files. The source and wheel distributions built successfully. Both standard smoke examples and the new palindrome example ran successfully. A reduced in-memory execution of the area notebook used eight circle points, t in [0, 0.04] and steps 0.02/0.01; it completed without modifying the canonical notebook. All seven theoretical PDFs compiled without final warnings or overflowing boxes and were visually reviewed after correcting obsolete page breaks. SDD structure validation passed (53 records, 19 requirements), and git diff --check passed during execution. The user requested stopping tests just after the complete run finished; no additional checks were launched. These closing documentation updates were not revalidated afterward.

GPU execution was unavailable; no long notebook, remote campaign or saved scientific result was regenerated. JAX accepts the generic built-in recipe configurations, while arbitrary subclass overrides remain explicitly unsupported. Arithmetic midpoint projection does not guarantee reversibility or exact symplecticity.
