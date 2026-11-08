# Execution results

The user accepted and authorized the two-base separation on 2026-11-08 with
the request to prepare and execute the scope discussed in the conversation.
The four preparation documents were presented before implementation.

IntegrationMethod now binds only the field and common numerical resources.
ImplicitIntegrationMethod adds the optional derivative and solver contract.
Runge--Kutta, projected palindromes and HBVM inherit that specialization;
execution, JAX and historical energy replay supply derivatives only there.
Explicit direct preparation rejects the jacobian keyword. Equations, solver
defaults, simulate calls and completed-result schemas are unchanged.

## Checks

- The 89 focused preparation/execution tests and 26 state/Modal tests passed.
  They cover two-argument external providers, derivative-free explicit runs,
  implicit extensions, independent repeated bindings, solver strategies,
  observations and CPU/JAX agreement. The first focused attempt exposed stale
  explicit Jacobian arguments in tests; those callers were migrated and rerun.
- Strict mypy passed for all 313 source files (`.venv/bin/python -m mypy src`).
- A before/after comparison was exactly equal for all ten named built-in methods,
  including recorded step sizes and available nonlinear-work arrays. It used
  Potential.random(A=0.05, M=2, nx=12, ny=12, seed=27), B=1.5, default rho,
  initial x=(0.3,0.8), y=(0.4,0.9), interval [0.1,0.23], step 0.03 and saved
  times [0.1,0.117,0.16,0.23]. This checks preservation, not absolute accuracy.
- Both examples passed: `examples/gc_orbit.py` took 20 steps;
  `examples/projected_abba.py` took four with one Newton correction per step.
- `.venv/bin/python -m build` produced source and wheel distributions. The wheel
  contains the canonical implicit base and its public export.
- Current simulation contracts, affected model/API guides and all 27 modified
  or new Markdown originals have updated, ignored Spanish reading copies.
  Both affected theory PDFs compiled twice without overflow or unresolved
  references. MEMORY.md records the contract and remains at 288 lines.
- The complete headless suite passed: 751 tests in 835.061 seconds, with three
  GPU checks skipped. Reproduce with
  `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`.
- Final `.venv/bin/python scripts/check_sdd.py` passed: 194 records and 22
  requirements. `git diff --check` passed. Implementation and validation are complete.

## Limits

No JAX GPU is available; CPU JAX is exercised. Modal checks use local
serialization and mocked workers, with no remote deployment. Notebook inspection
found no direct method.new_run calls requiring migration in active sources;
full notebooks and scientific campaigns were not executed. Old prepared-run
pickles should be recreated; completed solution archives keep their format.
