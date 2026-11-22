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
norms/block maxima using JAX float64. All methods evaluate the same physical H
and analytic partial_t H directly at their saved states. Schema-2 archives retain
H and H-H0 without energy quadrature; H-H0 is not a conservation error. Schema-1
archives remain readable with their original diagnostics.
Worker timings, allocated resources and explicitly estimated USD compute costs
are retained per rho/method case. Actual billing is not inferred from estimates.
See [the campaign protocol](../../docs/studies/main-reflected-campaign.md).

## Periodic particle verification

The separate periodic-particle study retains unwrapped zero-winding closure, step-refined seed positions, independent-method checks, minimum-period separation and phase-duplicate diagnostics. Its accepted JSON includes measured Modal resources and explicitly estimated compute costs. Candidate status is not a verification result. See [the periodic selection protocol](../../docs/studies/periodic-particles.md).

## Midpoint geometric defects

The separate midpoint study differentiates complete physical ABBA2Midpoint and BM4Midpoint steps and records signed determinant defects, correctly timed forward/backward infinity-norm displacements, copy separation and the determinant-identity residual. Symmetric-projection controls retain their nonlinear tolerances. Accumulated log absolute determinants and long-time return displacements are distinct diagnostics; energy variation is not a surrogate for either geometric property. Raw archives, parameters and provenance support saved-data visualization without reintegration. See [the study protocol](../../docs/studies/midpoint-geometry.md).

The current thesis sweep uses Euclidean per-particle reversal distances and ordered products of local physical Jacobians (`M_(n+1)=A_n @ M_n`) and direct evaluation of `abs(det(M_N)-1)` on the four-step/four-horizon grid (20,40,80,160 steps per cycle; 10,20,40,80 cycles). It compares the two existing eight-particle groups, with 16 evaluations per group. Each evaluation includes both midpoint methods and both diagnostics. Study composition is in `studies.midpoint_geometry_sweep`; validation, reduction and immutable archive/recovery are in `diagnostics.studies.midpoint_geometry_sweep`. The prior local/infinity-norm record remains historical. See [the accumulated protocol](../../docs/studies/midpoint-geometry-sweep.md).

Sweep matrix products are accumulated at 80 decimal digits and checked at 120, starting from saved float64 local Jacobians. New schema 2 preserves accumulated matrices as decimal strings; schema-1 logarithmic archives remain historical and readable.
