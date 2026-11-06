# Implemented behavior

Accepted on 2026-11-06: execute the temporal separation and deterministic auto selection presented in the conversation and in proposal.md, retaining all three differentiation modes.

Numerical preparation no longer receives or stores a SimulationRequest. Execution passes the request directly to planning and recording, and execution._scipy prepares live SciPy intervals and schedules output interpolation. Shared adaptive step details live in contracts. Gauss, SDIRK, HBVM and projected methods select derivatives without initial probes. Invalid supplied derivatives fail when used; explicit finite differences remain available. Broyden retains its projection-provided initial matrix and residual-only behavior. HBVM retains its first-residual validation boundary.

Direct preparation consumers, current simulation contracts, architecture/migration documentation and their Spanish copies were updated. No direct preparation calls were found in notebook code cells. Repository memory was updated in place and remains at 300 lines. Chapter 3 was aligned with the implemented behavior. Pre-existing edits in both repositories were preserved; no commit or push was performed.

# Checks completed

- `python -m mypy src`: passed for 306 source files.
- The final targeted regression run passed 104 tests covering preparation, execution, adaptive sampling, Jacobians, projection, state ownership and external execution interfaces. The six new temporal-preparation tests also preserve Broyden without field derivatives.
- Before/after snapshots compare 168 arrays from 48 short simulations: 12 methods, sparse/dense output and energy tracking disabled/enabled. All final arrays match exactly (maximum absolute discrepancy 0), including trajectories, available step times, Newton iterations, SciPy work counters and energy momentum. The case uses Potential.random(A=0.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5), rho=0.05, initial state [1.0, 1.2], interval [0.3, 0.355], nominal step/max_step 0.02, sparse times [0.3, 0.355] or dense times [0.3, 0.301, 0.32, 0.347, 0.355]. Constructors retain their default solver tolerances. The snapshot comparison is a regression check, not an independent accuracy reference.
- `examples/gc_orbit.py` and `examples/projected_abba.py` passed with 20 and 4 steps, respectively.
- Package sdist and wheel built successfully. The environment lacked setuptools; an isolated build fetched it into temporary storage. The wheel includes execution/_scipy.py. The final rebuild after moving the shared record also passed using the cached build dependency; the wheel was checked against the final controller and shared-record sources.
- `latexmk -pdf main.tex` in TFM_redaccion succeeded: main.pdf has 57 pages. The log contains no compilation errors or warnings. No PDF viewer or rendering was used.
- Specification validation passed for 174 records and 22 requirements. Spanish copies are ignored by Git; scoped whitespace checks passed.

# Final validation and limits

Complete `python -m unittest discover -s tests -v` ran 746 tests in 735.091 seconds: 741 passed, three GPU tests were skipped because no GPU is available, and two tests still asserted the old preparation behavior. The ABBA test expected the removed initial Jacobian evaluation; the independent-formulation test still passed a third argument to new_run. Both were migrated and their entire modules passed afterwards. The subsequent 104-test targeted run passed, including those two modules. After the final shared-record relocation, a further 29 adaptive, preparation and package/import tests passed. The complete discovery was not repeated after those corrections; no failure remains in the checks rerun against the corrected code. CPU/JAX comparisons in the full run passed.

The full-run log is /tmp/gc2d-temporal-suite.log; successful targeted logs are /tmp/gc2d-temporal-final-regressions.log and /tmp/gc2d-temporal-contract-tests.log. Build and compilation logs are /tmp/gc2d-temporal-final-build.log and /tmp/gc2d-temporal-latexmk.log. Temporary snapshots and the comparison script are under /tmp/gc2d-temporal-*. These are local validation artifacts, not permanent study data.

All implementation and documentation steps are complete. Existing saved result formats and configured numerical controls remain supported; direct method preparation callers must adopt the documented signature change. Prepared objects serialized with the old runtime structure must be recreated.
