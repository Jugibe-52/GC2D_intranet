+++
schema_version = 1
kind = "spec"
id = "states"
sources = ["src/execution/state.py", "src/states/history.py", "src/formulations/layout.py", "src/states/generators.py", "src/diagnostics/geometry.py"]
documents = ["docs/simulation/api-migration.md", "docs/dynamics/protocols.md"]

[[requirements]]
id = "STA-001"
tests = ["tests/test_states.py", "tests/test_state_ownership.py", "tests/test_core.py"]

[[requirements]]
id = "STA-002"
tests = ["tests/test_states.py", "tests/test_core.py", "tests/test_input_validation_contracts.py"]

[[requirements]]
id = "STA-003"
tests = ["tests/test_states.py", "tests/test_method_integration.py", "tests/test_adaptive_integration.py", "tests/test_jax_rk4.py"]
+++

# Physical states, integration state and generators

## STA-001: Owned physical history and reusable layouts

`states.States` owns physical samples of shape `(d*N,T)`, their times `(T,)`, a structural layout snapshot and an optional independent initial array. The recording lifecycle is `prepare -> record/record_step -> finalize`; only recorded columns are exposed. Recording validates time order, shape and finite values. Finalization requires all requested columns, seals the same object and rejects later recording. Direct array construction produces a completed result, including a single sample. Inputs and returned layout or initial-array copies cannot mutate stored data.

Use `.times`, `.values`, `.components()`, `.split()`, `.positions()`, `.velocities()`, `.particle_count` and `.select(indices)`. Velocity access requires named velocity components and fails for GC. Selection retains original times, layout and initial snapshot; whole-run diagnostics remain attached on their own axes. `GCState` and `FCState` are named component tuples, not container subclasses.

Canonical layouts belong to `formulations.layout` and share the `StateLayout` protocol from `contracts.configuration`. They handle packed `(d*N,*samples)`, component `(N,*samples)` and block `(d,N,*samples)` shapes while preserving NumPy/JAX backends, ordering and units. GC orders x then y; FC appends vx then vy. Layouts require no numerical-method dependency. Retired `State`, `Solution`, `IntegrationCollector`, `states.layout` and `contracts.request` have no compatibility aliases.

## STA-002: Independent initial geometry and diagnostics

`states.generators` exposes `circle`, `square`, `radial_star` and `ranked_radial_star`, returning packed NumPy physical arrays. Generator parameters, defaults, validation, particle ordering, endpoint/center conventions and coordinate units remain unchanged. Arrays enter `InitialValueProblem` directly with an optional explicit layout; generators own no time.

`diagnostics.geometry.calculate_area(values, layout=..., period=None)` evaluates signed polygon area for at least three ordered GC vertices, implicitly closing the contour. Histories retain trailing sample axes. A positive finite period enables minimum-image reconstruction, not a general rule for winding torus contours. Geometry labels belong in study metadata, defaulting to `polygon`. `save_solution` and `load_solution` retain their names and archive schemas and preserve absent initial snapshots when loading legacy archives.

## STA-003: Current internal state and independent sampling

`execution.state.IntegrationState(time, values)` owns a finite scalar time and a finite nonempty one-dimensional internal snapshot. A controller manages successive accepted states; the class has no stepping or storage algorithm. Internal coordinates may contain duplicates and passive auxiliaries. Only formulation-extracted physical values enter `States`; auxiliary sample data feed energy diagnostics at finalization.

Fixed off-grid samples use independent shortened maps from the same accepted interval start, never a prior shadow or the next main state. Endpoints reuse accepted states. Adaptive sampling uses accepted dense output. Sample times and main-step statistics retain separate axes. JAX carries equivalent functional device arrays inside compiled loops and constructs completed `States` at the host boundary. Saved histories, observers and serialized results must not alias mutable solver work arrays.
