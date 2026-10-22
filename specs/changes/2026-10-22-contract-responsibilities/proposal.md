# Proposed ownership

Move `InitialValueProblem` to `problem`, beside `request`, preserving the public
`simulation.InitialValueProblem` export. Move `ExecutionOptions` to
`execution.options`, also exporting it from `execution`. Migrate supported
consumers and retire the old modules without compatibility aliases.

Move RMS and maximum-error calculations from `AdaptiveReference` and
`EnergyAccuracySeries` to `diagnostics.comparison`. Move the periodic recurrence
calculation from `ParallelBM4RecurrenceResult` to `diagnostics.bm4_recurrence`.
For example, `reference.final_rms_floor` becomes `final_rms_floor(reference)`.
Keep calculation names, formulas, reductions, validation and array ownership;
update callers to the function interface. Records retain their constructors and
archive schemas, without importing diagnostics or study runners.

Move only `require_uncoupled_bm4` to `studies._validation`. Historical records
may still contain nonzero coupling; new calculations continue to reject it.
Keep scalar validators needed by archive records in their current owner.

Retain `contracts.arrays` as the NumPy/JAX compatibility boundary: NumPy-only
values select NumPy; any JAX array or tracer, including nested lists or tuples,
selects JAX. Do not add a common module or change device selection. Keep
protocols, step/observer records and other shared study records.

See [the integration architecture](../../../docs/simulation/integration-architecture.md)
for the surrounding responsibilities. No scientific redesign is proposed.
