# Execution results

Accepted on 2026-10-21: implement the initial-condition, reusable-solver and
projection-class scope presented in this conversation and the four preparation
documents. The user explicitly requested preparation and execution together.

Implementation and verification are complete; no planned work remains.

## Implemented behavior

InitialConditions owns physical input and layout; InitialValueProblem accepts it
alongside existing arrays. Shared NewtonSolver/BroydenSolver configurations live
in solvers. ReducedProjection and SimultaneousProjection own the 2N/6N equations;
methods provide their composition and derivatives. CPU and JAX use the same
projection operations, while Gauss, SDIRK and HBVM reuse Newton. Existing controls,
physical scales, sampling and saved-result formats are preserved.

Examples, model documents, current contracts and Spanish reading copies are
updated. The extended-model theory PDF was rebuilt and visually checked. A new
BM4 component diagram describes the implemented architecture; historical diagrams
remain intact. The memory records these responsibilities in 258 English lines.
See the [usage guide](../../../docs/simulation/initial-solvers-formulations.md).

## Verification

Environment: Python 3.12.3, NumPy 1.26.4, SciPy 1.17.1 and JAX 0.7.1, on CPU.

- `MPLBACKEND=Agg JAX_ENABLE_X64=true python -m unittest discover -s tests -v`:
  616 tests in 509.365 seconds, no failures and three GPU skips. This includes
  12 new tests covering ownership, solvers, formulations, JAX and serialized jobs.
- `python -m mypy src`: passed for 256 source files.
- `python examples/gc_orbit.py` and `python examples/projected_abba.py`: passed.
- `python -m build`: source and wheel distributions built; imports and new
  constructors passed using an extracted wheel. Public CPU imports do not load JAX.
- Two notebooks were checked in memory with cells 2, 4 and 6 and reduced parameters:
  `single_particle_orbit.ipynb` (10 RK4 steps) and `single_gc_bm4_orbit.ipynb`
  (2 BM4Midpoint steps), both under `notebooks/experiments/basics/`. Both used
  16x16 fields, maximum wave 2, interval [0, 0.02] and seven samples. Canonical
  notebooks were not rewritten or executed over their original long intervals.
- Solver/formulation imports have no upward dependencies on methods or execution.
  Changed Markdown documents have their ignored Spanish reading copies.
- `python scripts/check_sdd.py`: passed (43 records, 19 requirements).
  `git diff --check`: passed. SDD validates structure, not scientific correctness.

Commands used the repository `.venv/bin/python`; numerical runs used
`MPLBACKEND=Agg` and JAX checks enabled 64-bit precision.

## Numerical comparison

The final implementation reproduced all 90 saved arrays from 16 pre-change
integrations exactly: maximum absolute difference **0.0**, including physical
states, nonlinear iteration/evaluation counts, residual norms and passive energy.
The reference numerical source was at commit `b25117d51f40` before this change.

The cases used both scipy and JAX execution with RK4, GaussLegendre4, SDIRK4,
ABBA4Implicit reduced/simultaneous/Broyden, BM4Implicit and HBVM42, all tracking
energy. Input was Potential.random(A=0.05, M=2, nx=16, ny=16, seed=43), guiding-center
rho=0.1 and B=1.5, packed state [0.25, 0.65, 0.35, 0.7], interval [0.15, 0.25],
maximum step 0.02 and eight samples including off-grid times. This is measured
regression agreement for these cases, not a bound on physical trajectory error.

## Corrections and limits

The first complete run had 614 tests, three GPU skips and one failure: extracting
Newton changed HBVM's Jacobian-capability check order. Restored the initial
residual evaluation before that check and reused it in Newton; all nine validation
contract tests then passed. The final suite includes two additional tests.

The baseline capture first used an invalid backend label, corrected to scipy
before recording. A non-isolated build lacked setuptools; the normal isolated
build succeeded without changing project dependencies. An extracted-wheel smoke
call first used positional GC components, then passed with the documented keywords.

GPU execution is not verified on this machine. Serialized jobs were executed
locally; no remote job was submitted. Existing unrelated SDD edits were retained.
No commits, pushes or personal-configuration changes were made.
