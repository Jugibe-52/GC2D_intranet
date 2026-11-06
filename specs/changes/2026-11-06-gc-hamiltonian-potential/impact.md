# Affected parts

Update GC dynamics, JAX preparation, study result accessors, visualization consumers and their tests. Migrate the supported experiment notebook and active local development notebook sources that use the renamed object; leave archived copies and saved outputs intact. Retain archive schemas, metadata keys and field fingerprints. Old pickles containing configured dynamics or renamed result objects must be recreated; completed numerical data are not recomputed.

Update the README, shared dynamics documentation, API migration guide and current simulation specification, with their Spanish reading copies. Keep historical change records intact. Record the durable naming decision in the existing memory entry. Align only the recently edited paragraph in chapter 3 of the sibling TFM with the new implementation and recompile its root PDF.

The pre-existing change in `data/potential/V1/PHI_2.h5` is outside this task and will be preserved. Checks should use small synthetic fields where possible. No data regeneration, dependency installation, commits or synchronization are planned.
