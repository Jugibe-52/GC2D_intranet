# Process ownership

Keep `diagnostics` as the public facade and move its implementations into:

- `execution/`: run and reference progress.
- `methods/`: complete-step iteration summaries, shared Jacobians and
  symplecticity; `adaptive/` owns dense-trajectory observation and `extended/`
  owns ABBA/BM4-specific observers and energy reconstruction.
- `formulations/projection/`: doubled-to-physical projection diagnostics.
- `states/`: geometry of physical state arrays.
- `studies/`: comparison reductions, recurrence analysis and study-specific
  archive readers/writers.
- `io/`: common paths, storage transports, complete-result archives and buffered
  diagnostic output.

Only create packages with actual implementations. In particular, existing
iteration summaries observe complete method steps, so they belong to
`methods/implicit_iterations.py`; a future observer of individual algebraic
iterations would belong to `solvers/`. Keep shared validation at package level.

Retain existing top-level public names and explicit exports. Migrate consumers
to canonical defining modules and remove former implementation paths without
aliases or import hooks. For example, physical polygon area moves from
`diagnostics.geometry` to `diagnostics.states.geometry`. Observation contracts
remain in `contracts`; numerical processes do not import diagnostic observers.

Document this ownership rule in the project `AGENTS.md`, the diagnostic guide
and current specifications. The user's request to prepare and execute accepts
the organization discussed in the conversation and includes this instruction
update. No numerical behavior or personal configuration changes are included.
