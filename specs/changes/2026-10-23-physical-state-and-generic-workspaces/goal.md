# Physical state and generic workspaces

Keep one flat physical state between integration steps, independently of the
single or double formulation. Dynamics defines physical coordinates; the
formulation constructs temporary copies; the numerical method initializes,
advances and projects them. Preserve component-major storage and the current
physical equations and algorithms.

The [previous change](../2026-10-23-independent-state-formulations/results.md)
made the two state representations generic but retained equal accepted copies,
GC/FC layout classes and specialized map/state containers in `formulations`.
Consequently physical interpretation, workspace packing and numerical operations
still overlap. This proposal completes that separation rather than merely
moving the old layout classes to another package.

Remove `GCStateLayout`, `FCStateLayout`, `_GCExtendedState` and
`_FCExtendedState`. Integrate the physical coordinate declarations into their
dynamics and share packing operations. Keep `SingleFormulation` and
`DoubleFormulation` independent of GC, FC, physical parameters and energy policy.

This is a new proposal after the completed change. Its earlier results remain
historical evidence, not validation of this new representation. The completed
[method-owned step change](../2026-10-23-method-step/results.md) is also part of
the starting behavior and must be preserved.
