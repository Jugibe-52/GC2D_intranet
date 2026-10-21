# Execution results

The user accepted removal of harmonic coupling from both BM4 variants and authorized SDD preparation and execution on 2026-10-21. The accepted work is complete.

BM4Implicit and BM4Midpoint no longer accept coupling_frequency and never invoke C, including as an identity operation. CPU and JAX retain the signed coefficients, endpoint clock, projections, nonlinear controls and passive energy normalization. HarmonicCoupling remains available independently through formulations; its packed spatial state has shape (4N,) and component order (u_x, u_y, v_x, v_y).

Current study entry points reject nonzero legacy rates before integration. Historical automation manifests remain readable with their original metadata but cannot be resumed with the current methods. Stored results and frozen historical bundles were not relabeled or recalculated. Three supported notebooks, model guides, diagrams, theoretical PDFs and the current simulation contract were updated, with local Spanish Markdown copies. The decision is recorded in MEMORY.md (267 lines).

## Checks and observed results

- Focused BM4 modules: 21 tests passed. Both methods match an independent zero-coupling replay within 3e-15 absolute tolerance for three particles; a failing replacement for C confirms that neither method calls it. Independent C tests cover analytic rotation, fixed mean, reversal, composition, input ownership and invalid data. Existing convergence, Jacobian, nonautonomous time and passive energy checks also pass.
- The full headless suite ran 619 tests in 502.826 seconds: 611 passed, five failed because two study fixtures still supplied pi/8, and three GPU checks were skipped because no GPU was available. The fixtures were changed to zero; all nine tests in the two affected modules then passed (four local-JAX and five Modal-comparison tests). The full suite was not repeated after those fixture corrections.
- After preserving historical automation reads, all 12 automation tests passed, including rejection of new coupled work and coupled restarts. The independent DOP853/Radau reference discrepancy was 1.008e-13; the largest error across all four tested methods was 1.012e-10, and coordinate-rescaling disagreement was 5.829e-15, below the 1e-8 acceptance limits.
- Source typing passed for 256 files. Both gc_orbit.py and projected_abba.py smoke examples completed. Source and wheel distributions built successfully.
- Reduced in-memory checks passed for single_gc_bm4_orbit, generalized_energy and gc_area_and_projected_symplecticity_2. They used a 16-by-16 random field, max wave number 3, span 0.04 and steps 0.02/0.01; the area case used eight boundary points. The initial area check rejected a temporary path outside the notebook tree; the disposable validation layout was corrected and the rerun passed. Reduced settings and generated results were not saved to the canonical notebooks.
- Four theoretical PDFs compiled with two LaTeX passes each, without reported warnings or overflowing boxes. Final pages and updated diagrams were rendered and visually checked. The SDD checker passed (48 change records, 19 requirements); git diff --check passed.

Commands used include .venv/bin/python -m unittest discover -s tests -v with MPLBACKEND=Agg, focused discovery patterns test_bm4*.py, test_local_jax_star.py, test_modal_cpu_comparison.py and test_poincare_automation.py, .venv/bin/python -m mypy src, .venv/bin/python -m build, and .venv/bin/python scripts/check_sdd.py.

## Limits

GPU execution, remote Modal jobs and long notebook campaigns were not run. The Modal tests use local test doubles. Historical development notebooks and frozen bundles were not mass-migrated. The general palindromic A/B/T inheritance redesign remains outside this accepted change. No implementation work remains for the accepted scope.
