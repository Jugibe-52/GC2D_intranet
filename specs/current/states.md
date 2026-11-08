+++
schema_version = 1
kind = "spec"
id = "states"
sources = ["src/contracts/methods/quadrature.py", "src/states/initial.py", "src/execution/state.py", "src/states/history.py", "src/dynamics/coordinates.py", "src/dynamics/jax_coordinates.py", "src/states/generators.py", "src/diagnostics/states/geometry.py"]
documents = ["docs/simulation/initial-solvers-formulations.md", "docs/simulation/api-migration.md", "docs/dynamics/protocols.md"]

[[requirements]]
id = "STA-001"
tests = ["tests/test_initial_solvers_formulations.py", "tests/test_states.py", "tests/test_time_row_states.py", "tests/test_state_ownership.py", "tests/test_physical_workspaces.py", "tests/test_shared_array_dynamics.py", "tests/test_core.py"]

[[requirements]]
id = "STA-002"
tests = ["tests/test_states.py", "tests/test_core.py", "tests/test_input_validation_contracts.py"]

[[requirements]]
id = "STA-003"
tests = ["tests/test_states.py", "tests/test_method_integration.py", "tests/test_adaptive_integration.py", "tests/test_jax_rk4.py"]
+++

# Physical states, integration state and generators

## STA-001: Owned physical history and reusable layouts

`states.States` owns physical samples of shape `(T,d*N)`, their times `(T,)`, a structural layout snapshot and an optional independent initial array. The recording lifecycle is `prepare -> record/record_step -> finalize`; only recorded rows are exposed. Recording validates time order, shape and finite values. Finalization requires all requested rows, seals the same object and rejects later recording. Direct array construction produces a completed result, including a single sample. Inputs and returned layout or initial-array copies cannot mutate stored data.

Use `.times`, `.values`, `.components()`, `.split()`, `.positions()`, `.velocities()`, `.particle_count` and `.select(indices)`. Velocity access is supplied by the FC convention and fails for GC. Selection retains original times, layout and initial snapshot; whole-run diagnostics remain attached on their own axes. `components()` and `split()` return ordinary tuples of component views.

Dynamics declare `GCCoordinates()` for fixed `(x, y)` blocks or `FCCoordinates()` for fixed `(x, y, vx, vy)` blocks, from `dynamics.coordinates`. `PhysicalCoordinates` is a stateless base with common array operations; dimensions are fixed properties and no names or role-index fields are configured. `split()` returns an ordinary tuple. `JaxGCCoordinates` and `JaxFCCoordinates`, from `dynamics.jax_coordinates`, inherit their respective operations and only select `jnp` through `array_namespace`.

`GCCoordinates` and `FCCoordinates` have no instance data or constructor controls.
GC has dimension 2; FC has dimension 4. A matching dimension alone does not make
another convention compatible with either model.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a model from dimension. `StateLayout.snapshot()` explicitly supplies an immutable NumPy convention; `describe_coordinates` validates that snapshot and its dimension. Compatibility compares the snapshot class and equality, even for equally sized arrays. Initial conditions and results own their arrays and conventions without retaining live dynamics or a potential. Custom conventions derive from `PhysicalCoordinates` and implement fixed operations instead of role tables; external adapters may supply these snapshots without inheritance. Packing and views preserve `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)`. The respective JAX classes support tracing; incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. States stores time rows `(T,d*N)`, with `states[i]` at `t[i]`; component accessors retain `(N,T)`. New solution archives use schema 2; the reader converts schema 1 columns explicitly. Archives can be interpreted without constructing dynamics or loading a field.

## STA-002: Independent initial geometry and diagnostics

`states.generators` exposes `circle`, `square`, `radial_star` and `ranked_radial_star`, returning packed NumPy physical arrays. Generator parameters, defaults, validation, particle ordering, endpoint/center conventions and coordinate units remain unchanged. Arrays enter `InitialValueProblem` directly with an optional explicit layout; generators own no time.

`diagnostics.states.geometry.calculate_area(values, layout=..., period=None)` evaluates signed polygon area for at least three ordered GC vertices, implicitly closing the contour. Histories retain trailing sample axes. A positive finite period enables minimum-image reconstruction, not a general rule for winding torus contours. Geometry labels belong in study metadata, defaulting to `polygon`. `save_solution` and `load_solution` retain their names, write schema 2 time rows, read schema 1 columns and preserve absent initial snapshots when loading legacy archives.

## STA-003: Current internal state and independent sampling

Every accepted `IntegrationState(time, values)` contains only a scalar time and an owned physical vector `(d*N,)`. Sampling snapshots carry no quadrature or momentum. `DoubleFormulation(physical_size)` creates temporary copies only for doubled methods; its size belongs to that representation, not to IntegrationMethod. Shadow samples use independent shortened maps from the accepted left endpoint and do not change the main trajectory.

Fixed off-grid samples use independent shortened maps from the same accepted interval start, never a prior shadow or the next main state. Endpoints reuse accepted states. Adaptive sampling uses accepted dense output. Sample times and main-step statistics retain separate axes. JAX carries equivalent functional device arrays inside compiled loops and constructs completed `States` at the host boundary. Saved histories, observers and serialized results must not alias mutable solver work arrays.

`States.record_history(values)` copies a complete physical history into an empty
open recorder, validating shape, finite values and packed layout before writing.
Invalid input leaves it empty; partially recorded or completed histories cannot
be overwritten. Recording does not finalize the result. Compiled execution uses
this bulk operation and the executor then applies the same finalization boundary
as streaming CPU recording.

## Explicit physical initial conditions

`states.InitialConditions` owns a validated packed array and copied structural
layout. GC/FC factories accept one-dimensional component arrays; all public data
access returns independent copies. The class is immutable and remains protected
after pickle. `InitialValueProblem.initial_conditions` owns this input; a supplied
object already carries its layout, so an additional layout argument is rejected.
Raw arrays remain accepted and normalize through the same class. Compatibility
with dynamics is checked by the problem; `initial_state` still returns an array.
Time, generated geometries, internal integration state and saved results retain
their existing contracts.

Private preparation functions own initial values and layouts before assignment.
Complete-array construction and `record_history` share the same shape, finiteness
and layout checks, preserving input dtype and the existing validation order.
Recording lifecycle guards remain in `States`.

## Structural internal representations

`DoubleFormulation(physical_size)` is the only state formulation. It prepares two temporary physical copies through `lift`, `join_copies` and `split_copies`, preserving NumPy/JAX backends and complete FC velocity blocks. It stores only a positive vector size; it owns no dynamics, layout, time, momentum, solver or projection. Methods using ordinary physical arrays need no formulation; `SingleFormulation` is removed.

`EnergyObserver()` records physical H and the positive analytic partial time derivative `hamiltonian_time_derivative` at accepted endpoints and requested physical samples. It does not consume internal stages or integrate momentum. Per-step diagnostics are `energy_start`, `energy_end`, `energy_change`, `energy_time_derivative_start` and `energy_time_derivative_end`, with shape `(steps,N)`. Saved histories `physical_hamiltonian` (alias `hamiltonian`), `energy_drift` and `hamiltonian_time_derivative` have shape `(N,T)`. Energy variation in a time-dependent Hamiltonian is not a numerical conservation error.

## Structural declaration

Dynamics declare `GCCoordinates()` for fixed `(x, y)` blocks or `FCCoordinates()` for fixed `(x, y, vx, vy)` blocks, from `dynamics.coordinates`. `PhysicalCoordinates` is a stateless base with common array operations; dimensions are fixed properties and no names or role-index fields are configured. `split()` returns an ordinary tuple. `JaxGCCoordinates` and `JaxFCCoordinates`, from `dynamics.jax_coordinates`, inherit their respective operations and only select `jnp` through `array_namespace`.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a model from dimension. `StateLayout.snapshot()` explicitly supplies an immutable NumPy convention; `describe_coordinates` validates that snapshot and its dimension. Compatibility compares the snapshot class and equality, even for equally sized arrays. Initial conditions and results own their arrays and conventions without retaining live dynamics or a potential. Custom conventions derive from `PhysicalCoordinates` and implement fixed operations instead of role tables; external adapters may supply these snapshots without inheritance. Packing and views preserve `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)`. The respective JAX classes support tracing; incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. States stores time rows `(T,d*N)`, with `states[i]` at `t[i]`; component accessors retain `(N,T)`. New solution archives use schema 2; the reader converts schema 1 columns explicitly. Archives can be interpreted without constructing dynamics or loading a field.
