# Execution results

Accepted on 2026-11-05: the user approved candidate-only `residual(x)` and
`jacobian(x)` for both projections and the corresponding TFM explanation, then
explicitly requested preparation and execution. The four documents were
presented before implementation.

## Implemented

`ProjectionProblem` binds each step and provides the complete residual and its
Jacobian. Formulation helpers are now `assemble_residual`. Newton's adapter
reuses the matching composition trace in an immutable local context; analytic
particle blocks and dense/difference routes retain their previous operations.
Broyden, guesses, stopping rules, reconstruction and counters are preserved.
The general solver API is unchanged.

Both TFM formulations and their connection to solvers are updated. Root
`TFM_redaccion/main.pdf` compiled successfully with `latexmk -pdf main.tex`
(no warnings); the changed pages were visually inspected.
The guides, model architecture, current specification and Spanish copies are
updated. The memory records the interface within its existing 300 lines.
Existing unrelated TFM edits were preserved. This execution created no commits
or uploads. A concurrent operation recorded the chapter changes in `8b0ce0f`;
further unrelated TFM edits and their rebuilt PDF were preserved.

## Checks

- Six new interface tests pass, including nonlinear/cross-particle derivatives,
  both formulations, ABBA2/ABBA4/BM4 recipes, candidate changes, initial roots,
  composition reuse and JAX JIT/autodifferentiation. The largest discrepancy
  against independent residual differences was **1.704e-10** (probe size 1e-6).
- Full suite: `python -m unittest discover -s tests -v` completed successfully:
  **738 tests in 713.646 seconds, 3 skipped** because no GPU is available.
  The other 735 tests passed; there were no failures or errors.
- `python -m mypy src`: passed for 305 files. Both repository smoke examples pass.
- Source and wheel builds pass using `python -m build --no-isolation` with the
  already installed bundled setuptools 84.2.1/wheel 0.48.0 on PYTHONPATH.
  The initial attempt found no setuptools in `.venv`; no dependency was installed.
  Import and residual/Jacobian checks from the extracted wheel pass without
  importing JAX on the CPU path.
- SDD validation passes (164 records, 22 requirements); `git diff --check` passes.
  Every changed Markdown document has its ignored Spanish reading copy.

Commands used `.venv/bin/python`, with `MPLBACKEND=Agg` and `JAX_ENABLE_X64=true`
for numerical tests. An initial focused run exposed a missing coordinates
argument in the new test fixture; it was corrected before the six passing tests.

## Numerical regression

All **180 arrays from 30 integrations match exactly**: maximum absolute
difference **0.0**, including states, iteration/evaluation counts, residual
norms, multiplier norms and passive energy. The baseline was recorded from
clean code at `05099e3`; the comparison script is
`/tmp/gc2d-projection-baseline.py` (`record` before changes, `compare` afterwards).

The cases combine scipy/JAX, ABBA2/ABBA4 with both formulations and reduced BM4,
with analytic Newton, finite-difference Newton and Broyden (30 maximum
iterations and existing default tolerances). Inputs are
`Potential.random(A=.08, M=2, nx=12, ny=12, seed=53)`,
`GuidingCenterDynamics(potential, rho=.05)`, two particles at x=[.2,.55],
y=[.3,.65], fixed step .02, interval [.15,.23] and output times
[.15,.16,.17,.19,.21,.23], with EnergyObserver enabled.
This measures preservation of the previous implementation, not trajectory
accuracy against an exact physical solution. GPU behavior is not verified here.
