# Affected parts

Update fixed-method configuration, adaptive bounds, the request API and
host/JAX schedules. Migrate studies, examples, tests and affected notebook source
cells while retaining their numerical settings. Keep development notebooks
ignored and avoid full expensive runs.

Update current simulation contracts, migration guidance and affected model
architecture documentation. Preserve previous change records and unrelated
working-tree edits. Existing saved data retain their schemas and recorded grids;
never relabel old uniformly rescaled trajectories as the new policy.

Checks cover final-step clipping, observations, independent sampling, reuse,
energy and implicit work, adaptive equivalence and backend agreement.
