# Affected code and evidence

Update potential splines, evaluation, JAX, gyroaveraging and GC scaling together.
Retain base Grid semantics and reproduce explicitly selected periodic results.
Snapshots, solution files, fingerprints and study reconstructions preserve
extension and upper edges. Historical HDF5/rho/radial studies retain their
recorded periodic model; new generic and Poincare automation configurations
expose the choice without changing historical archive identities.
Reflected solution archives use schema 3 so older readers reject them instead
of silently reconstructing a periodic field. Undefined comparison ratios for
stationary corner trajectories remain NaN instead of raising division errors.

Particle animations, Poincare views and distance diagnostics use shared folding
and the correct physical period. Raw states are untouched by display changes.

Tests cover boundaries/corners, crossing velocities, Jacobians, shifted origins,
negative coordinates, temporal modes, cosine-mode spectral gyroaveraging,
CPU/JAX derivatives, persistence and legacy reconstruction. Update relevant
potential/dynamics documentation, model links, current contracts and Spanish
reading copies. Review notebook uses with reduced checks without overwriting
old results. Preserve unrelated working-tree edits. TFM_redaccion is read-only.
Keep existing quintic notebook experiments explicitly periodic; their saved
outputs and metadata remain intact.
