# Separate preparation from the lifecycle

Extract four private functions in `src/execution/execution.py`:

- `_prepare_backend`: prepare dynamics and the selected device, with execution metadata.
- `_prepare_observers`: validate and bind the energy observer and preserve callback order and identity.
- `_prepare_numerical_run`: resolve the Jacobian and quadrature requirements, prepare and validate a fresh numerical run, and describe its observation settings.
- `_prepare_compiled_step`: enforce fixed JAX restrictions and prepare its compiled step when selected.

Keep argument validation, status transitions, `States.prepare`, worker selection
and `States.finalize` visible in `Execution.run`. Compiled-step preparation stays
inside the existing `try/finally`, so failures still consume the prepared run.
All per-run data remains local, and helpers return explicit typed results.
Use descriptive names such as `prepared_dynamics`, `numerical_run`,
`execution_options` and `step_diagnostics`.

Preserve signatures, lazy imports, error order and messages, metadata precedence,
observer behavior, numerical algorithms and sampling. Keep the same single
recording/finalization boundary described in the
[execution guide](../../../docs/simulation/execution.md). No open decisions remain.
