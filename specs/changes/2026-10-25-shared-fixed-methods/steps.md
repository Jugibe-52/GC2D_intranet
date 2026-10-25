# Implementation and verification

Execution accepted on 2026-10-25. Checked items below record completed work.

- [x] Record a fresh baseline before source edits. Use small deterministic GC
  and supported FC problems with multiple particles, float64 JAX CPU, fixed
  steps, a clipped endpoint and irregular saved times. Include every fixed
  family, generic recipes, both supported projections, Newton/Broyden, analytic
  and finite-difference derivatives, energy off/on and CPU observations.
  Capture trajectories, accepted/saved times, energy, multipliers, work counters
  and failures. Also capture DOP853/Radau CPU and hybrid behavior and the hash
  of their implementation file.
- [x] Add potential/dynamics `_as_jax()` preparation and verify lazy imports,
  reused samples/splines/effective fields, idempotent reuse, unchanged originals,
  precision checks and isolated caches. Preserve explicit JAX callers and
  external CPU dynamics without requiring conversion from every protocol.
- [x] Introduce optional JAX Newton/Broyden classes with common solve interfaces.
  Check reused configurations, cached first residuals, zero-correction solutions,
  caller norms, damping, derivative policies, dense coupled systems, declared
  particle blocks and full Broyden updates. Preserve defaults, counts, errors
  and callback restrictions; test compiled failures without host callbacks.
- [x] Share fixed-step algorithms and projection equations across array types.
  First prove explicit Euler/RK4, then the implicit classical/HBVM families and
  palindromic methods. Replace runtime NumPy coercions, mutation, Python scalar
  extraction and mutable observation counters. Keep matrix assembly and
  equation-specific derivatives with the methods; delegate linear solves and
  nonlinear iteration. Verify 2N/6N residuals and Jacobians against independent
  differences and preserve HBVM's norm and deferred derivative checks.
- [x] Prepare the common numerical step in execution and compile it generically.
  Preserve scheduling, device placement, bounded reuse, quadrature, independent
  shadow sampling and one prepare/finalize boundary. Keep the adaptive file
  untouched and verify its CPU/hybrid regressions. Remove replaced fixed JAX
  modules and `jax_mode` dependencies after migrating consumers; retain no
  aliases or alternate algorithm copies.
- [x] Audit fixed algorithms for JAX imports, JAX-specific names, device choices,
  backend branches and forced conversions of runtime arrays. Demonstrate that
  NumPy and compiled JAX call the same numerical functions. NumPy-only runs must
  work without importing JAX. Compare production code across all affected
  packages to expose relocated duplication or unnecessary new layers.
- [x] Compare against the baseline and independent accuracy tests, investigating
  every changed numerical decision or diagnostic. Preserve existing stricter
  tests; general CPU/JAX trajectory limits are rtol=3e-10, atol=3e-11 and energy
  limits rtol=3e-9, atol=3e-11, as in `test_jax_methods`. Report measured maxima,
  settings and counts; require exact equality where arithmetic is unchanged.
  These are comparison limits, not physical error guarantees. Exercise available
  GPU cases and report unavailable hardware explicitly; do not replace device
  execution with a hidden host fallback. Record representative compilation and
  warmed execution costs without promising a speedup.
- [x] Check affected public examples, archive round trips and local Modal jobs;
  run only reduced notebook checks if relevant callers changed. Update permanent
  guides, fixed-model diagrams, current contracts and confirmed memory, with
  Spanish copies. Run focused suites, then
  `MPLBACKEND=Agg python -m unittest discover -s tests -v`,
  `python -m mypy src`, `python examples/gc_orbit.py`,
  `python examples/projected_abba.py` and `python -m build`, using the existing
  environment. During execution create `results.md` with acceptance, actual
  checks and limitations; finish with `python scripts/check_sdd.py` and review
  the complete diff.

A change to equations, solver semantics, supported behavior or DOP853/Radau
requires revised review. Private representation choices within the proposal
do not require a separate approval.
