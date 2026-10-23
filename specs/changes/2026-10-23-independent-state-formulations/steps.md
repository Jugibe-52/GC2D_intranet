# Implementation and verification steps

- [x] Capture short deterministic baselines before changing code. Use
  `Potential.random(A=0.05, M=2, nx=16, ny=16, seed=43)`, GC rho=0.1 and B=1.5,
  FC rho=0.2 and eta=0.1, GC state `[0.25, 0.65, 0.35, 0.7]` and FC state
  `[0.25, 0.65, 0.35, 0.7, 0.1, -0.2, 0.2, 0.1]`. Use the existing normalized
  coordinates, interval `[0.15, 0.25]`, maximum step 0.02 and eight saved times,
  including off-grid samples. Cover the existing GC/FC method combinations,
  reduced/simultaneous projections, Newton/Broyden and energy on/off on CPU and
  available JAX with 64-bit precision. Record actual solver settings.
- [x] Introduce `SingleFormulation` and `DoubleFormulation`, migrate the two old
  state-class names and define explicit construction/extraction operations.
  Verify GC/FC and external-layout dimensions for N=1 and N=3, with and without
  energy and sample axes, on NumPy and available JAX. Check input preservation,
  independent copies where mutation is allowed, and rejection of invalid sizes.
- [x] Move initial-state construction and passive-momentum initialization into
  each method's run setup. Pass energy capabilities explicitly. Reuse copy
  operations in specialized maps and keep formulation imports independent of
  methods/execution. Preserve supported specialized split interfaces.
- [x] Make doubled step preparation explicit in methods and projection helpers.
  Check two consecutive steps, perturbed candidate states and shortened shadow
  steps. Verify diagonal versus multiplier-shifted input, fixed internal size,
  one projection at the established boundary and no reset within a composition.
  Preserve solver defaults, failure behavior, residual scaling, stage times and
  reuse of the converged trace for energy and observations.
- [x] Compare the new results with the captured baseline, including physical
  trajectories, passive energy, accepted grids and nonlinear work. Expect exact
  equality for representation round trips and unchanged arithmetic paths;
  investigate and report every numerical discrepancy. Keep existing independent
  convergence/reversal tests and CPU/JAX limits: trajectory rtol=3e-10,
  atol=3e-11; energy rtol=3e-9, atol=3e-11. These regression limits and requested
  residual tolerances do not bound physical trajectory error. Report measured
  maximum differences and any unavailable backend.
- [x] Migrate affected imports and usage examples; update current specifications,
  per-model ownership documentation and Spanish reading copies. Audit notebooks
  and saved-data consumers, running only small representative notebook subsets
  in memory or disposable copies. Verify existing archive round trips without
  rewriting scientific results. Keep historical records intact.
- [x] Run focused tests, then the full headless unittest suite, source mypy,
  `examples/gc_orbit.py`, `examples/projected_abba.py` and the package build in
  the project environment. Check the final diff and run
  `python scripts/check_sdd.py`. Record acceptance, actual implementation,
  measurements and remaining limits in `results.md` during execution; update
  memory only with confirmed lessons.

All execution items are complete. See [results.md](results.md) for the measured
comparisons, checks and backend limitations.
