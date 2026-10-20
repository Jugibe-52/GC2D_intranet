+++
schema_version = 1
kind = "spec"
id = "states"
sources = ["src/states/state.py", "src/states/layout.py", "src/states/generators.py", "src/diagnostics/geometry.py"]
documents = ["docs/simulation/api-migration.md", "docs/dynamics/protocols.md"]

[[requirements]]
id = "STA-001"
tests = ["tests/test_states.py", "tests/test_state_ownership.py", "tests/test_core.py"]

[[requirements]]
id = "STA-002"
tests = ["tests/test_states.py", "tests/test_core.py", "tests/test_input_validation_contracts.py"]
+++

# Physical states and external generators

## STA-001: Owned instantaneous states and reusable layouts

`states.State` stores one finite nonempty NumPy vector and an independent structural layout snapshot. GC uses `[x_1,...,x_N,y_1,...,y_N]`; FC appends `vx` and `vy` blocks. Units belong to the dynamics. Construction rejects histories and incomplete blocks; no unset state or mutating setter is provided. Input changes and edits to returned values, components or layouts cannot change the snapshot. `initial_state` exposes the same independent data for the existing problem-source protocol.

Use `State.from_components(*components, layout=...)`, `.values`, `.split()`, `.positions()`, `.velocities()` and `.particle_count`. Velocity access requires named velocity components and fails for GC. `GCState` and `FCState` remain named component tuples, not storage subclasses. Structural external sources and layouts do not require inheritance from State.

Canonical layout classes live in `states.layout` and are reexported by `states`. They handle packed shape `(d*N,*samples)`, component shape `(N,*samples)` and explicit blocks `(d,N,*samples)` without changing NumPy/JAX backends. Existing split behavior and particle ordering are preserved. Numerical methods continue to operate on arrays. The removed `initial_conditions` package and `contracts.state_layout` module have no aliases.

## STA-002: Independent geometry preparation and diagnostics

`states.generators` exposes `circle`, `square`, `radial_star` and `ranked_radial_star`, returning ordinary State objects with GC layouts. Preserve the previous generator parameters, defaults, validation, particle ordering and endpoint/center conventions. Geometric coordinates use the caller's units without conversion.

`diagnostics.geometry.calculate_area(values, layout=..., period=None)` evaluates the signed polygon area of at least three ordered GC vertices, closing the final edge implicitly. Histories preserve their trailing sample axes. An optional positive finite period applies the existing minimum-image reconstruction; this is not a general area rule for winding contours on a torus. Geometry labels are study metadata, with `polygon` as the unspecified label, rather than State attributes. Saved numerical archives retain their schema and remain readable, including absent initial snapshots.
