+++
schema_version = 1
kind = "spec"
id = "diagnostics"
sources = ["src/diagnostics/methods/energy.py", "src/contracts/methods/quadrature.py", "src/diagnostics/__init__.py", "src/diagnostics/methods/implicit_iterations.py", "src/diagnostics/methods/projection/observer.py", "src/diagnostics/states/geometry.py", "src/diagnostics/io/persistence.py"]
documents = ["src/diagnostics/README.md", "AGENTS.md"]

[[requirements]]
id = "DIA-001"
tests = ["tests/test_package_layout.py::PackageLayoutTests.test_diagnostics_follow_processes_without_retired_flat_modules"]

[[requirements]]
id = "DIA-002"
tests = ["tests/test_package_layout.py::PackageLayoutTests.test_diagnostic_public_names_resolve_to_process_implementations", "tests/test_package_layout.py::PackageLayoutTests.test_public_exports_are_explicit_and_resolve"]

[[requirements]]
id = "DIA-003"
tests = ["tests/test_energy_observation.py", "tests/test_package_layout.py::PackageLayoutTests.test_numerical_processes_do_not_load_diagnostic_implementations", "tests/test_diagnostic_buffering.py", "tests/test_solution_persistence.py"]
+++

# Process-owned diagnostics

## DIA-001: Locate implementations by the observed process

Specialized historical generalized-energy studies explicitly use stage diagnostics under `diagnostics.methods.stage_energy` and `studies._generalized_energy`. They retain their signed quadrature, physical kappa normalization and archive labels. Generic execution has no momentum accumulator. `States.k` and `States.err` remain readers for historical diagnostics and return `None` when absent. Old numerical archives remain readable; recreate configured-object pickles after the API change.

## DIA-002: Explicit public facade and canonical imports

Public exports are explicit. AdaptiveTrajectoryObserver and adaptive method event types are retired with the adaptive method API. Other supported facade names resolve to their single process-owned implementation; internal consumers import defining modules.

## DIA-003: Independent observation and unchanged artifacts

`EnergyObserver()` records physical H and the positive analytic partial time derivative `hamiltonian_time_derivative` at accepted endpoints and requested physical samples. It does not consume internal stages or integrate momentum. Per-step diagnostics are `energy_start`, `energy_end`, `energy_change`, `energy_time_derivative_start` and `energy_time_derivative_end`, with shape `(steps,N)`. Saved histories `physical_hamiltonian` (alias `hamiltonian`), `energy_drift` and `hamiltonian_time_derivative` have shape `(N,T)`. Energy variation in a time-dependent Hamiltonian is not a numerical conservation error.

## Main reflected campaign

Projected palindromic steps expose signed `projection_multipliers` with shape
`(d*N,)` in step statistics, in addition to scalar solver diagnostics. Consumers
must compare these array values elementwise. The main campaign reduces the
accepted-step history to ten aligned samples/cycle, per-particle projection
norms/block maxima and Simpson endpoint energy balance using JAX float64.
Worker timings, allocated resources and explicitly estimated USD compute costs
are retained per rho/method case. Actual billing is not inferred from estimates.
See [the campaign protocol](../../docs/studies/main-reflected-campaign.md).

## Periodic particle verification

The separate periodic-particle study retains unwrapped zero-winding closure, step-refined seed positions, independent-method checks, minimum-period separation and phase-duplicate diagnostics. Its accepted JSON includes measured Modal resources and explicitly estimated compute costs. Candidate status is not a verification result. See [the periodic selection protocol](../../docs/studies/periodic-particles.md).
