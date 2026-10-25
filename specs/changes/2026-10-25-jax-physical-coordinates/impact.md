# Affected boundaries

Change coordinate operations, the shared GC/FC methods, dynamics exports and
execution's JAX preparation. The new JAX classes are exposed through their
canonical modules, without eager re-exports from the base dynamics package. Keep numerical kernels, formulation workspaces,
physical state storage and potential evaluation algorithms unchanged.

Update tests that previously expected the base coordinate class to preserve JAX
arrays, and cover both explicit coordinate classes, inherited dynamics methods,
direct JIT/vmap/automatic differentiation, FC split flows, preparation identity,
source immutability and existing spline reuse. Retain GC/FC integration and
hybrid adaptive coverage, optional dependency checks and unsupported-subclass errors.

Update the dynamics and JAX guides, migration notes, affected model architecture
documents, current specifications and project memory, with Spanish reading
copies. Existing simulation notebooks and saved numerical archives need no
migration. Direct users of base coordinates under JIT must choose
`JaxPhysicalCoordinates`; direct JAX dynamics calls use the new subclasses.
