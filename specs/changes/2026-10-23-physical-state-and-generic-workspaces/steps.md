# Implementation and verification steps

- [x] Capture the actual working-tree baseline before implementation, including
  the accepted method-owned step behavior. Use
  `Potential.random(A=0.05, M=2, nx=16, ny=16, seed=43)`, GC rho=0.1/B=1.5 and
  FC rho=0.2/eta=0.1; GC state `[0.25, 0.65, 0.35, 0.7]` and FC state
  `[0.25, 0.65, 0.35, 0.7, 0.1, -0.2, 0.2, 0.1]`. Use interval [0.15, 0.255],
  fixed method `step=0.02` or adaptive `max_step=0.02`, and saved times
  `[0.15, 0.153, 0.17, 0.181, 0.21, 0.23, 0.25, 0.255]`. Cover supported GC/FC
  method combinations, energy off/on, Newton/Broyden, reduced/simultaneous
  projection and available JAX CPU with 64-bit precision. Record actual solver
  settings, physical results, energy, accepted grids and numerical work. Capture
  direct/adjoint split behavior separately, including coupling and momentum.
- [x] Integrate physical coordinate declarations into dynamics and migrate
  generic packing, initial conditions and results. Check GC, FC and an external
  model with a nonstandard component count, N=1/N=3 and sample axes. Validate
  order/role compatibility even for equal-size vectors; reject incomplete blocks.
  Verify independent data, existing position/velocity access, NumPy/JAX tracing
  and archive interpretation without constructing dynamics or loading a field.
- [x] Simplify the two formulations to size-checked temporary copy operations.
  Remove per-model layout/state containers and migrate specialized maps,
  projections and contracts to their owners. Check that importing formulations
  does not import dynamics, methods, execution, solvers or optional JAX, and
  that no representation branches on GC/FC classes or component counts 2/4.
  Verify shifted candidates remain distinct until the method's projection.
- [x] Migrate persistent snapshots, step records, all methods, controllers and
  host/JAX collection to physical Z plus separate time/momentum. Verify initial
  zero momentum, consecutive steps, nonzero supplied momentum, perturbed
  candidates, negative direct/adjoint substeps, clipped final steps and independent
  shortened samples. Check fixed accepted size, no accumulated duplicate copies,
  immutable inputs, isolated repeated runs and no workspace aliasing with output.
  Preserve one projection, accepted-stage energy reuse, physical-only adaptive
  error control and observer/nonlinear failure behavior.
- [x] Compare physical trajectories, time/momentum histories, H+kappa, accepted
  grids and work against baseline. Expect exact agreement where arithmetic is
  unchanged; investigate and report every discrepancy. Deliberate dimension
  metadata changes must match the new actual physical/workspace sizes. Retain
  independent convergence/reversal checks and CPU/JAX trajectory limits
  rtol=3e-10, atol=3e-11; energy rtol=3e-9, atol=3e-11. Report measured maxima and
  unavailable backends; these limits do not bound physical trajectory error.
- [x] Migrate public imports and consumers; check existing numerical archive
  fixtures, new round trips, standalone loading, diagnostic reconstruction and
  local process/job serialization. Audit tracked and ignored notebooks, execute
  minimal representative subsets in memory, and update permanent guides,
  affected model artifacts, current specifications and Spanish reading copies.
  Check that retired paths/classes have no live compatibility aliases.
- [x] Run focused ownership, formulation, split-map, lifecycle, JAX, persistence,
  package and method-step tests, then the full headless unittest suite, source
  mypy, `examples/gc_orbit.py`, `examples/projected_abba.py` and the package build.
  Review the final diff, update confirmed memory, and run
  `.venv/bin/python scripts/check_sdd.py`. During execution create `results.md`
  with acceptance, actual changes, measurements and any remaining limitations;
  update this checklist only from completed work.

Preparation checks document structure only. None of the implementation or
numerical validation above is completed by creating this proposal. There are no
unresolved scientific choices within this scope; concrete helper placement and
typing details must follow the responsibilities and compatibility stated here.
