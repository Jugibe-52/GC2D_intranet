# Execution results

The user accepted the hierarchy presented in the conversation and requested preparation and execution on 2026-11-07. The four preparation documents were presented before implementation. Scope: migrate the four fixed classical methods to tableau-driven Runge–Kutta inheritance and preserve supported behavior.

## Implemented

`methods.runge_kutta` and matching contracts replace the old namespaces without aliases. Immutable, hashable `ButcherTableau(A,B,C)` validates finite coefficients, shapes and consistency. Three reusable algorithms implement explicit, sequential diagonal and coupled stages. Euler/RK4, SDIRK4 and Gauss–Legendre inherit those calculations and declare their tables and metadata. Gauss keeps its specialized two-stage observation. Generic families support NumPy/JAX, with coefficients in their compiled configuration keys and recorded metadata.

Newton controls, particle/dense Jacobians, finite differences, iteration callbacks, silent observation replay, passive quadrature, main-step clipping and independent shadow samples are preserved. Imports, four model guides/diagrams, current specifications and Spanish reading copies were migrated. Two active ignored notebooks only needed Markdown source-link changes; no numerical cells or stored outputs were changed. Historical notebook copies and study archives were retained. Euler and RK4 theory PDFs were rebuilt and all five pages visually inspected, with no overfull boxes. Memory records the inheritance and cache contracts in 299 lines.

## Checks completed

- A pre-change baseline captured 702 arrays for the four methods, with and without analytic particle Jacobians. Three particles start at x=(0.2,-0.7,1.2), y=(0.5,0.6,-0.1), t=0.3. The field is (-y + 0.1 sin(x) + t², x + 0.2 cos(y) - t), with 12 steps of 0.08 and a final step of 0.03. Comparing physical states, stage states/times/weights and nonlinear statistics at absolute tolerance 2e-12 gave a maximum discrepancy of 2.220446049250313e-16 and identical nonlinear work counts.
- 45 focused existing tests passed (Euler, Gauss, SDIRK, Newton observations, package layout and prepared constructor contracts).
- All 8 new tests in `tests/test_runge_kutta.py` passed, including independent linear algebra for three coupled stages, mixed explicit/DIRK stages with different diagonals, second-order convergence, immutable serialization, shared-callable cache separation and complete generic CPU/JAX runs with energy, nonzero initial time, irregular output and a shortened final step.
- `python -m mypy src` passed for 310 source files. Both `examples/gc_orbit.py` and `examples/projected_abba.py` completed with the headless plotting backend.
- `python -m build` produced the source distribution and wheel. Wheel inspection found the new family and no retired classical package.
- Spanish reading copies and the two development notebooks remain ignored. No tracking rules, commits or remote data were changed.

## Final validation

`MPLBACKEND=Agg python -m unittest discover -s tests -v` completed 754 tests in 795.474 seconds: OK, with three skips. The final eight-test Runge–Kutta module was also run separately after adding the complete-job/cache checks, and all eight passed. The 12 Poincare automation tests passed after updating the family name in its validation message. These commands used the repository `.venv/bin/python` (Python 3.11.10, NumPy 1.26.4, JAX 0.7.1).

The final build and source typing passed. `python scripts/check_sdd.py` passed with 179 records and 22 requirements; `git diff --check` passed. All planned steps are complete.

## Limits

JAX GPU checks require a GPU; this machine reports none. Numerical equivalence permits roundoff changes rather than bitwise identity. Configured-object pickles referencing retired module paths must be recreated; completed solution formats stay unchanged. No long notebook or production study was run, and this change makes no performance claim.
