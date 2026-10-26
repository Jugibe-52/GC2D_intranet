# Proposal

Merge `prepare_environment` into `_prepare_backend`. Resolve the JAX device and
execution mode there, prepare the dynamics, and bind the hybrid adapter when
needed. Keep optional JAX imports lazy.

Let `Execution.run` call `prepare_step` directly for device-resident methods.
Move fixed-JAX callback, progress and iteration-observer restrictions into
`prepare_step`; retain its exact-type guard for independently prepared runs.
Remove `_prepare_compiled_step` and the repeated execution-mode dispatch there.
Keep the non-null device check and preparation inside `try/finally`, before
result allocation. Store the same device reference in `PreparedStep`.

Keep observer and numerical-run preparation separate, but read the request from
their existing problem argument. Preserve all annotations, including declarations
before unpacking, without introducing a new context class or changing public APIs.
