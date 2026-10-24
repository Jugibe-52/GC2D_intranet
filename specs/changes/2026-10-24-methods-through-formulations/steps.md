# Implementation and verification steps

- [x] Capture a fresh baseline from the actual tree. Reuse the deterministic
  field, states, interval and sampling grid in the
  [physical-state checks](../2026-10-23-physical-state-and-generic-workspaces/steps.md).
  Cover supported GC/FC methods, multiple particles, energy off/on, analytic and
  finite differences, Newton/Broyden, both projections, negative map durations,
  explicit coupling, clipped endpoints and interior samples. Include available
  float64 JAX CPU and hybrid DOP853/Radau. Save effective solver policies,
  physical/energy histories, grids, counters and event snapshots. Inventory
  production Python lines/statements and duplicate controls across all affected
  packages, including contracts, diagnostics and execution.
- [x] Introduce function-based preparation and explicit derivative/flow contracts;
  bind physical capabilities and backends in execution. Migrate all method
  families and direct callers. Remove retained dynamics/problem resources and
  `SingleFormulation`; keep standalone `DoubleFormulation` for duplicated work.
  Verify plain callables without hidden source access, arbitrary valid layouts,
  existing GC/FC restrictions, input ownership and isolated repeated runs.
- [x] Make the solver the only owner of algebraic controls and iteration
  observation. Preserve family defaults and reconstruct each migrated caller's
  effective configuration. Remove mirrored method fields, legacy precedence and
  redundant solver options. Share derivative helpers while distinguishing field,
  map and residual targets. Test dense coupled systems and independent blocks,
  zero-correction convergence, HBVM's deferred Jacobian/norm/damping, Broyden
  initialization, unsupported configurations and nonconvergence propagation.
- [x] Extract passive energy into the shared observer path. Expose method-owned
  quadrature data or accepted interpolants through small contracts; move observer
  registration to execution and remove energy inputs/results from numerical
  steps. Verify zero and supplied nonzero initial observer momentum, one update
  per accepted step, signed weights, GC normalization, intermediate output,
  repeated sample queries, rejected trials and diagnostic map callbacks. Match
  existing H, momentum and H+momentum histories; confirm tracking on/off produces
  identical physical trajectories and accepted grids. Preserve event identity,
  nested stages, observer failure handling and fresh state across runs.
- [x] Migrate compiled and hybrid execution to function-based physical kernels,
  solver-derived options and a separate functional energy accumulator. Check
  lazy imports, precision, cache reuse, no host callbacks in compiled loops,
  original-source provenance, supported combinations and disabled-observer cost.
  Ensure shadow samples never mutate accepted physical or observer state and
  adaptive error control remains purely physical. Validate existing worker jobs
  locally, including energy configuration serialization and unsupported callbacks.
- [x] Compare with the baseline, expecting exact equality when arithmetic is
  unchanged. Investigate differences in trajectories, energy, grids, work counts
  and observations; separate unchanged numerical work from deliberately relocated
  diagnostic accounting. Retain independent convergence, energy and reversal
  checks. Use established CPU/JAX trajectory bounds rtol=3e-10, atol=3e-11 and
  energy bounds rtol=3e-9, atol=3e-11. Report measured maxima and unavailable
  backends; comparison limits and solver tolerances do not bound physical error.
- [x] Migrate public consumers, diagnostics, checkpoint studies, examples and
  actual notebook users; remove old APIs/helpers without compatibility trees.
  Verify old archives, new round trips and reduced notebook runs. Update affected
  guides, model artifacts, current specifications, project guidance and memory,
  with Spanish reading copies. Audit imports and runtime dependencies, then
  report removed duplication and complete before/after production-code counts,
  including new observers/contracts. Explain growth and remove unnecessary layers.
- [x] Run focused method, solver, observation, sampling, JAX and ownership tests,
  then headless unittest discovery, source mypy, `examples/gc_orbit.py`,
  `examples/projected_abba.py` and package build. During execution create
  `results.md` with acceptance, actual measurements and limitations; update this
  checklist from completed work. Finish with
  `.venv/bin/python scripts/check_sdd.py` and review the complete diff.

No material scope choice remains unresolved. Concrete callable types and compact
quadrature records can be refined within these boundaries; they must not recreate
a universal formulation or change numerical semantics. The implementation and verification steps are complete.
See [results.md](results.md) for measurements and limitations.
