# Affected parts

`src/formulations/state.py` and its public exports gain the two canonical names
and explicit representation operations. Remove the state representations'
ownership of initial values and time. Preserve structural layouts in
`formulations.layout`; external layouts remain supported without inheriting a
built-in GC or FC class. Shared interfaces stay in the matching `contracts`
domain, and importing representations must not import methods or execution.

Classical, adaptive, HBVM and extended methods must explicitly construct their
initial internal state. Extended methods and their projection/midpoint helpers
prepare copies at each step or residual evaluation. `GCDoubledMaps`, FC split
helpers and direct/adjoint contracts need an ownership audit so their reusable
operations share representation logic without creating a second step lifecycle.
Projection equations and solver implementations keep their current mathematics
and public access routes; a broad package relocation is outside this proposal.

Energy tracking currently obtains dynamics through the state formulation's
problem. Pass the required Hamiltonian operations explicitly from the method
instead. Retain the passive quadrature, accepted-stage reuse and physical-only
error control. Host/JAX export and diagnostics must continue to see physical
samples, independent auxiliary arrays and the same work counters. Preserve the
controller's accepted snapshots and independent shortened sampling steps.

Update the affected ownership, formulation, solver, palindromic, CPU/JAX and
package-layout tests, especially `test_state_formulations.py`,
`test_state_ownership.py`, `test_initial_solvers_formulations.py`,
`test_palindromic_operations.py` and `test_jax_methods.py`. Add checks for FC
double representations; these are representation checks, not FC ABBA/BM4 runs.

Migrate callers of the two renamed state classes, including supported imports
through `simulation`. Keep method constructors, specialized split-map names,
projection selectors, archive schemas and stable diagnostic labels unchanged.
Saved trajectories require no conversion or regeneration. Audit examples,
studies and notebooks for direct uses; edit only affected consumers and validate
notebooks with small disposable runs. Ignored development notebooks remain
ignored, and historical results keep their recorded meaning.

Update the simulation integration, API migration, initial/solver/formulation
and dynamics protocol guides; the affected per-model architecture documents;
and `specs/current/simulation.md`, `states.md` and `results.md` where ownership
changes their contracts. Update current model diagrams or theory sources only
where they describe changed ownership, and regenerate affected artifacts.
Maintain Spanish Markdown reading copies. Keep completed change records as
history and record confirmed implementation lessons in memory during execution.

The main risk is changing numerical work while moving state preparation:
resetting a candidate during composition, projecting twice, accepting stale
copies, redoing energy quadrature, or changing solver scaling. Regression
comparisons must cover these paths as well as array dimensions.
