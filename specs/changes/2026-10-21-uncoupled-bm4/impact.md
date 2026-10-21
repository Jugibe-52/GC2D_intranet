# Affected components

Both BM4 classes, JAX adapters, shared formulation exports, their diagnostics, study constructors, portable calculation entry points, and supported runnable notebooks are affected. Retain stored-result readers and immutable historical bundles; reject attempted coupled restarts with current methods.

Update the BM4 and shared extended-model documentation, theoretical PDFs, current simulation contract, and local Spanish reading copies. Tests cover an independent uncoupled replay, convergence, signed reversal, Jacobians, tracking and observer behavior, CPU/JAX agreement, and standalone C.

No changes to physical dynamics, gyro-radius, units, particle ordering, BM4 weights, or nonlinear tolerances are intended. Doubled spatial arrays remain component-major (4N); physical output is 2N. C leaves the copy mean unchanged and rotates their difference with angle 2*frequency*duration.
