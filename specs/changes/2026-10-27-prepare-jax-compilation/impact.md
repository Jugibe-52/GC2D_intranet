# Affected parts

Change the private fixed-JAX driver and its call site in `Execution.run`.
`PreparedStep` and numerical algorithms retain their contracts. Extend the
compiled-step tests to verify compilation before execution, cache reuse and
specialization, fresh data, and lifecycle behavior on failure. Existing backend
comparisons cover physical states, energy, nonlinear solves and shadow sampling.
Capture small pre-change runs for a direct numerical regression comparison.

Update the execution and JAX guides, the simulation specification and relevant
model architecture descriptions, with Spanish reading copies. Update the
chapter 3 subsection in the sibling TFM repository and recompile its root PDF.
Record the confirmed preparation boundary in repository memory.

Existing notebook and Modal callers use the same public API and need no changes.
Compiled executables remain process-local; remote workers prepare them locally.
No notebook runs, deployment, data migration, commits or synchronization are
required. CPU is available locally; GPU behavior can only be tested if a device
is detected.
