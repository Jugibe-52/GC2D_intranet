+++
schema_version = 1
kind = "spec"
id = "states"
sources = ["src/contracts/methods/quadrature.py", "src/states/initial.py", "src/execution/state.py", "src/states/history.py", "src/dynamics/coordinates.py", "src/states/generators.py", "src/diagnostics/states/geometry.py"]
documents = ["docs/simulation/initial-solvers-formulations.md", "docs/simulation/api-migration.md", "docs/dynamics/protocols.md"]

[[requirements]]
id = "STA-001"
tests = ["tests/test_initial_solvers_formulations.py", "tests/test_states.py", "tests/test_state_ownership.py", "tests/test_physical_workspaces.py", "tests/test_core.py"]

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

Use `.times`, `.values`, `.components()`, `.split()`, `.positions()`, `.velocities()`, `.particle_count` and `.select(indices)`. Velocity access requires named velocity components and fails for GC. Selection retains original times, layout and initial snapshot; whole-run diagnostics remain attached on their own axes. `CoordinateValues` provides tuple views with declared component names.

Dynamics declare immutable `PhysicalCoordinates(component_names, position_names=(), velocity_names=())` from `dynamics.coordinates`. GC declares `(x,y)`; FC declares `(x,y,vx,vy)`. Position and velocity roles are explicit. The structural protocol is `contracts.dynamics.coordinates.StateLayout`; external implementations require no inheritance.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a physical model from dimension 2 or 4. It checks dimension, component order and position/velocity roles, even for equally sized arrays. Initial conditions and results retain an immutable structural description and owned arrays, without retaining live dynamics or a potential. Generic packing, component views and role access support `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)` while preserving NumPy/JAX tracing. Incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. Saved numerical archives retain existing GC/FC schemas and can be interpreted without constructing dynamics or loading a field.

## STA-002: Independent initial geometry and diagnostics

`states.generators` exposes `circle`, `square`, `radial_star` and `ranked_radial_star`, returning packed NumPy physical arrays. Generator parameters, defaults, validation, particle ordering, endpoint/center conventions and coordinate units remain unchanged. Arrays enter `InitialValueProblem` directly with an optional explicit layout; generators own no time.

`diagnostics.states.geometry.calculate_area(values, layout=..., period=None)` evaluates signed polygon area for at least three ordered GC vertices, implicitly closing the contour. Histories retain trailing sample axes. A positive finite period enables minimum-image reconstruction, not a general rule for winding torus contours. Geometry labels belong in study metadata, defaulting to `polygon`. `save_solution` and `load_solution` retain their names and archive schemas and preserve absent initial snapshots when loading legacy archives.

## STA-003: Current internal state and independent sampling

`execution.state.IntegrationState(time, values, quadrature=None)` owns a finite scalar time and a finite nonempty one-dimensional physical snapshot. A controller manages successive accepted states; the class has no stepping or storage algorithm. Optional quadrature transfers sample data to an independent energy observer; no momentum belongs to the physical snapshot. Saved histories and energy diagnostics retain separate ownership.

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

Every accepted snapshot is `IntegrationState(time, values)`: scalar time and physical `values` of shape `(d*N,)`. GC remains 2N and FC 4N with or without energy observation. Doubled GC work has 4N entries; generic doubled FC work has 8N. Workspaces are temporary and never become accepted states. Optional quadrature accompanies a sample without storing energy momentum. Generic copy support does not extend GC numerical methods to FC.

Execution supplies `f(t, z)`, independent initial data and an optional Jacobian to the numerical method. `advance(t, z, h)` returns physical `StepResult.state`, numerical work and optional quadrature. `EnergyObserver`, supplied through `simulate(..., observers=(EnergyObserver(),))`, owns a fresh passive accumulator for each run. Accepted steps update it once; each shadow sample starts independently from the accepted left endpoint. JAX carries a separate functional accumulator through the compiled loop.

Execution gives the energy observer H and `-partial_t H`. The observer evaluates saved physical H once; `states.energy.energy_diagnostics` combines supplied H and physical kappa without evaluating dynamics. Existing energy labels and `(N,T)` auxiliary histories remain unchanged; `extended_time` is broadcast from scalar sample times. The default initial kappa is zero; `EnergyObserver(initial_momentum=...)` accepts an explicit physical value per particle. Numerical archives require no migration; old pickled configured objects should be recreated.

## Structural declaration

Dynamics declare immutable `PhysicalCoordinates(component_names, position_names=(), velocity_names=())` from `dynamics.coordinates`. GC declares `(x,y)`; FC declares `(x,y,vx,vy)`. Position and velocity roles are explicit. The structural protocol is `contracts.dynamics.coordinates.StateLayout`; external implementations require no inheritance.

`InitialValueProblem` takes the dynamics declaration or requires an explicit layout. It never infers a physical model from dimension 2 or 4. It checks dimension, component order and position/velocity roles, even for equally sized arrays. Initial conditions and results retain an immutable structural description and owned arrays, without retaining live dynamics or a potential. Generic packing, component views and role access support `(d*N,*samples)`, `(d,N,*samples)` and `(N,*samples)` while preserving NumPy/JAX tracing. Incomplete blocks are rejected.

Built-in dynamics keep physical parameters immutable. A new problem is required for revised initial values. Saved numerical archives retain existing GC/FC schemas and can be interpreted without constructing dynamics or loading a field.
