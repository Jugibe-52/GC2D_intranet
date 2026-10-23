# Affected parts

This preparation describes a new change on top of the current working tree,
including the completed independent-formulation and method-step changes. Their
source edits and records must remain intact until deliberately migrated during
execution. Preparation writes only these four documents and Spanish copies.

In `src/dynamics/`, integrate coordinate declarations and share packing helpers;
move the physical-coordinate contracts from `contracts/formulations/` to the
matching dynamics domain. Migrate `problem.py`, `states/initial.py`, generators,
history accessors and geometry consumers. Preserve independent inputs, component
views, sample axes, serialization and external structural coordinate support.
`formulations/layout.py` and its per-model classes are retired after migration.

Reduce `formulations/state.py` to generic copy operations. Move the actual GC
maps to the extended-method implementation, FC splitting to a method-owned
splitting module, and coupling/projection equations to the corresponding method
modules. Adapt map contracts, stage observations and diagnostics; remove
`_GCExtendedState`, `_FCExtendedState` and the specialized formulation wrappers.
Audit `_PreparedFC`, map configuration snapshots and `initial_internal_state` so
the same ownership problem is not retained under a new name. Preserve direct/
adjoint reversal, coupling coefficients and existing supported capabilities.

All classical, adaptive, HBVM and extended methods must return a physical
accepted vector and explicit auxiliary momentum. Update `IntegrationState`,
`StepInfo`, `StepResult`, controller contracts, the host worker and adaptive
dense sampling together. Keep stage work transient and observations independent.
Energy assembly moves out of state formulations without adding H evaluations,
replaying maps or resetting momentum at step or sampling boundaries.

Update `methods/_jax_dispatch.py`, compiled-step contracts and
`execution/_jax_fixed.py`: particle count and accepted buffers currently depend
on copies and tracking. The new device carry and saved arrays contain physical
values and separate auxiliaries; temporary copies remain in numerical kernels.
Remove dispatch that creates a GC/FC layout from dimension alone. Keep optional
imports, immutable preparation, precision guards, synchronization and isolation.

`diagnostics/io/persistence.py` currently selects layouts by concrete class.
Read/write the existing GC/FC schema labels using coordinate descriptions,
without recreating retired classes or requiring a potential. Audit CSV/HDF5
study readers, archive fingerprints, dimension checks and historical diagnostic
labels. Old numerical results require no rewriting or reintegration. Old Python
pickles naming retired classes are not portable archives; document regeneration
of configured objects while preserving new-object process serialization.

Migrate supported examples, studies, visualizers, public exports, package-boundary
tests and notebook sources that use removed classes or packed internal states.
Inspect ignored development notebooks as well as tracked experiments; keep their
tracking status and use small disposable executions. Full scientific campaigns
and regeneration of historical figures/results are outside this change.

Update the dynamics/initial-state, execution, integration, JAX and API migration
guides; affected per-model architecture and theory where ownership descriptions
change; and `specs/current/simulation.md`, `states.md`, `results.md` and diagnostic
contracts where dimensions change. Regenerate affected model artifacts, maintain
Spanish copies and update confirmed memory during execution. Preserve previous
change records and their evidence; no global diagram or bulk history rewrite.
