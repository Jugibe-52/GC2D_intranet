# Execution results

The user accepted and authorized method simplification, adaptive-runtime removal and endpoint energy observation on 2026-11-07. Implementation and validation are complete. Existing uncommitted palindromic and Runge--Kutta work was preserved.

IntegrationMethod consumes temporary InitialConditions; execution owns initial snapshots and progress. Supplied Jacobians are particle blocks, while numerical differences remain dense. EnergyObserver has no parameters or momentum accumulator: it records H, positive analytic partial_t H and energy differences at endpoints and saved samples. It owns snapshots even when physical functions reuse buffers. GC differentiates normalized Fourier time factors through dt=1; FC retains its physical scaling. Historical stage diagnostics and independent SciPy references preserve their archive meanings.

## Checks

- Focused checks passed: 92 corrected-contract tests, 54 preparation/energy tests and six final endpoint tests. The first full run exposed stale expectations and a historical-study metadata merge issue; both were corrected and rechecked. The final complete rerun (`MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`) passed: 746 tests in 857.824 seconds, with three GPU checks skipped.
- Strict mypy passed for 312 source files. Both smoke examples passed: RK4 took 20 steps; projected ABBA took four with one Newton correction each. Source/wheel builds passed; the wheel contains the shared stage diagnostic and no retired adaptive/classical/extended modules.
- Before/after trajectories were exactly equal for RK4, GaussLegendre4, SDIRK4, ABBA4Implicit and BM4Implicit. The check used Potential.random(A=0.05, M=2, nx=12, ny=12), B=1.5, two particles with packed initial state [0.3,0.8,0.4,0.9], interval [0.1,0.23], step 0.03 and saved times [0.1,0.117,0.16,0.23]. CPU/JAX float64 checks for RK4 and ABBA4 gave zero state discrepancy and at most 1.735e-18 in partial_t H.
- Cubic time dependence verifies analytic endpoint derivatives rather than secants, without internal-stage evaluations. Sparse outputs preserve accepted observations. Conservation, order, projection and historical generalized-energy tests remain covered.
- Current contracts, model guides, diagrams and Spanish copies were updated; all 14 affected theory PDFs compiled. SDD validation passed: 189 records, 22 requirements. Confirmed contracts were recorded in MEMORY.md (288 lines).

## Limits

GPU hardware is unavailable. Modal checks use local serialization and mocked workers, without remote deployment. Four local development notebooks had progress calls migrated; only reduced BM4/SDIRK paths ran in memory. Full campaigns and canonical outputs were not regenerated. Three notebooks retain pre-existing retired BM4Implicit1/nonzero-coupling configurations: their scientific setup needs a separate migration before full execution.
