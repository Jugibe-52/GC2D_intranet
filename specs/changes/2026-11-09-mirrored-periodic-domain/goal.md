# Continuous reflected Hamiltonian with a selectable extension

Implement the agreed reflected Hamiltonian on the plane with continuous velocity
and spatial Jacobian, retaining the direct periodic model. New potentials use
spatial_extension="reflected"; callers can select "periodic". Old untagged
results retain their periodic meaning.

The 9 November 2026 preparation identified a boundary ambiguity. The
15 November agreement resolves it: zero normal derivatives, reflected scalar
Hamiltonian, canonical rotation, consistent trajectory display, and adapted
gyroaveraging. The user requested preparation and execution, explicitly excluding
TFM_redaccion. Only GC2D_codigo is changed; physical states stay unwrapped.
