# Affected parts

Direct changes affect states, problem construction, shared solvers, projection
formulations, implicit methods, JAX adapters, public exports and package discovery.
Migrate imports of moved private modules in source, tests, scripts and notebooks;
keep canonical implementations without compatibility module trees. Preserve old
public method names, array construction and numerical controls. Update examples
and relevant simulation/model documents and current contracts, with Spanish copies.

Check initial ownership, external layouts, pickle/remote job round trips, reusable
solvers, both projection formulations, numerical work counts, failure handling,
CPU/JAX agreement and independent sampling. Capture short deterministic results
before editing and compare afterwards. Existing stored results retain their format;
no trajectory archive regeneration or long notebook execution is required.

Existing uncommitted SDD workflow changes belong to the user and are preserved.
No commits, pushes, deployments, remote calculations, personal settings or TFM
edits are included. The optional problems package and a separate Gauss-stage
formulation class are not needed for this scope.
