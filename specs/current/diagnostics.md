+++
schema_version = 1
kind = "spec"
id = "diagnostics"
sources = ["src/diagnostics/execution/horizontal_transport.py", "src/diagnostics/methods/energy.py", "src/contracts/methods/quadrature.py", "src/diagnostics/__init__.py", "src/diagnostics/methods/implicit_iterations.py", "src/diagnostics/methods/projection/observer.py", "src/diagnostics/states/geometry.py", "src/diagnostics/io/persistence.py"]
documents = ["src/diagnostics/README.md", "AGENTS.md"]

[[requirements]]
id = "DIA-001"
tests = ["tests/test_package_layout.py::PackageLayoutTests.test_diagnostics_follow_processes_without_retired_flat_modules"]

[[requirements]]
id = "DIA-002"
tests = ["tests/test_package_layout.py::PackageLayoutTests.test_diagnostic_public_names_resolve_to_process_implementations", "tests/test_package_layout.py::PackageLayoutTests.test_public_exports_are_explicit_and_resolve"]

[[requirements]]
id = "DIA-003"
tests = ["tests/test_horizontal_transport.py", "tests/test_energy_observation.py", "tests/test_package_layout.py::PackageLayoutTests.test_numerical_processes_do_not_load_diagnostic_implementations", "tests/test_diagnostic_buffering.py", "tests/test_solution_persistence.py"]
+++

# Process-owned diagnostics

## DIA-001: Locate implementations by the observed process

Specialized historical generalized-energy studies explicitly use stage diagnostics under `diagnostics.methods.stage_energy` and `studies._generalized_energy`. They retain their signed quadrature, physical kappa normalization and archive labels. Generic execution has no momentum accumulator. `States.k` and `States.err` remain readers for historical diagnostics and return `None` when absent. Old numerical archives remain readable; recreate configured-object pickles after the API change.

## DIA-002: Explicit public facade and canonical imports

Public exports are explicit. AdaptiveTrajectoryObserver and adaptive method event types are retired with the adaptive method API. Other supported facade names resolve to their single process-owned implementation; internal consumers import defining modules.

## DIA-003: Independent observation and unchanged artifacts

`EnergyObserver()` records physical H and the positive analytic partial time derivative `hamiltonian_time_derivative` at accepted endpoints and requested physical samples. It does not consume internal stages or integrate momentum. Per-step diagnostics are `energy_start`, `energy_end`, `energy_change`, `energy_time_derivative_start` and `energy_time_derivative_end`, with shape `(steps,N)`. Saved histories `physical_hamiltonian` (alias `hamiltonian`), `energy_drift` and `hamiltonian_time_derivative` have shape `(N,T)`. Energy variation in a time-dependent Hamiltonian is not a numerical conservation error.

## Horizontal transport during execution

`HorizontalTransportObserver(cycle_duration=1.0)` is defined in
`diagnostics.execution.horizontal_transport` and exported from `diagnostics`.
Pass it through `simulate(..., observers=(HorizontalTransportObserver(),))`.
It integrates the absolute physical horizontal velocity on every accepted
complete interval with the composite trapezoidal rule. Internal composition
stages, backward splitting substeps and shadow output samples are excluded.
The width is the largest minus smallest unwrapped x coordinate across the
initial state and all accepted endpoints. It is not the width of folded
Poincare samples. The dimensionless rate is travel / (elapsed cycles * width),
where elapsed cycles = elapsed runtime time / `cycle_duration`.

The final per-particle vectors are `horizontal_travel`, `horizontal_min`,
`horizontal_max`, `horizontal_width` and `horizontal_transport_rate`.
A zero width yields NaN for the rate and is identified by
`horizontal_zero_width="undefined_nan"`. Scalar metadata includes
`horizontal_cycles`, `horizontal_cycle_duration`, `horizontal_observed_steps`,
`horizontal_coordinates="physical_unwrapped"` and
`horizontal_quadrature="trapezoid_accepted_endpoints"`.
NumPy and JAX use the same accepted endpoints and O(particles) accumulation;
no per-step transport history is retained. The cycle-only Modal worker preserves
these five final vectors while discarding other diagnostic arrays.

This is a numerical quadrature, not an exact integral. Both the integral and
sampled width may miss excursions below the integration step; check step
refinement when assessing accuracy. The observer does not change the method,
its physical trajectory, saved output times or other observers.
