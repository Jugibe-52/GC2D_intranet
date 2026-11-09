# Affected parts

Physical mirror extension affects `potential.grid`, `_evaluation`, spline
construction, `Potential.evaluate`, `gyroaverage`, and the JAX evaluation
kernel. The distinction between base-cell width and full spatial period must
remain explicit. Sample arrays keep the documented `(nx, ny)` axis order;
any additional endpoint or doubled-grid storage must be recorded accurately.

Alternating GC rotation also changes `GuidingCenterDynamics.vector_field` and
its particle Jacobians. Boundary termination would affect CPU/JAX execution,
sampling and result completion; an interior-cell Jacobian is not a boundary
crossing rule. Fixed-rotation or display-only choices have different impact.

Visualization changes reach shared folding helpers, Poincare automation and
rho viewers, and GC particle animations. Existing animation arrows sometimes
rotate the electric field directly, so those paths need the same convention as
the selected dynamics. Raw `States` storage remains unwrapped. Physical
minimum-image distances use the full repeat length, not the displayed width.

Potential snapshots, pickles, HDF5 sidecars and study reconstruction must
preserve the selected conventions. Old untagged archives keep their original
meaning; input fingerprints and resume checks distinguish changed dynamics.
Historical saved-only viewers remain tied to their recorded conventions.

Extend potential, JAX, GC Hamiltonian/Jacobian, persistence and viewer tests.
Update affected dynamics documentation, current contracts and model
architecture documents only when the corresponding behaviour is implemented.
Provide Spanish reading copies. Review notebook call sites; use reduced,
disposable checks rather than executing long studies or rewriting old results.
