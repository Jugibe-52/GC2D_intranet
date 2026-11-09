# Work and checks

- [x] Inspect the current potential, gyroaverage, GC equations, recording and
  plotting paths; identify the boundary discontinuity and missing upper samples.
- [x] Prepare and present the four documents together; run
  `.venv/bin/python scripts/check_sdd.py` (198 records, 22 requirements passed).
  Check that all four Spanish reading copies are ignored by Git.
- [ ] Resolve the requested boundary behaviour and reconcile all four proposal
  documents, including the concrete public API and sample representation.
  The user has requested execution; do not request duplicate approval of an
  unchanged, fully specified scope.
- [ ] Implement the selected scalar/domain conventions and preserve legacy
  archive interpretation. Create `results.md` when implementation begins.
- [ ] Implement only the selected dynamic/boundary policy, with matching
  Jacobians and NumPy/JAX behaviour. Keep saved physical states unwrapped.
- [ ] Migrate the affected visualizations and arrows using one shared fold.
- [ ] Check negative coordinates, shifted origins, multiple cells, all four
  orientations, exact edges and corners, and unchanged source arrays. Check
  scalar/derivative and NumPy/JAX agreement at float64 tolerance `1e-10`.
- [ ] If physical mirror extension is selected, verify analytic cosine-mode
  gyroaverages against `J0(rho*|k|)` at grid samples, including distinct averaged
  endpoint values; target absolute error `1e-12` for well-scaled test fields.
- [ ] Test the selected boundary behaviour, raw-state persistence, archive
  mismatch rejection and displayed velocities against transformed trajectories.
- [ ] Update permanent documentation and Spanish copies, run focused tests,
  mypy, the headless suite, smoke examples, build and the final SDD check.
  Record actual results and any mathematical limitations without claiming
  global symplecticity from interior-cell tests.
