# Mirrored spatial representation

Represent neighbouring potential cells by reflection, so the original
orientation repeats after two cell widths in each direction. Keep integrated
and saved physical states unwrapped; apply any display folding only to copies.
The potential, displayed trajectories and arrows must use explicitly recorded
coordinate conventions.

The user requested preparation and execution on 9 November 2026 on branch
`codex/mirrored-periodic-domain`. The desired interior-cell reflections are
clear. The behaviour at cell boundaries is not yet selected: reflecting both
the scalar potential and the GC velocity generally prevents smooth crossings.
Execution depends on resolving that physical choice, rather than assigning an
arbitrary value to a discontinuous vector field.
