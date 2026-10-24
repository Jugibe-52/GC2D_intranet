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

`diagnostics` follows `src` responsibilities through `execution`, `methods`,
`solvers`, `states` and `studies`; shared output and transport belong to
`io`. Adaptive and extended method diagnostics have their own method-family
packages. Complete-step iteration summaries belong to methods; future
individual algebraic-iteration observers belong to solvers. Shared metrics have
one implementation. Packages are created only when needed by actual diagnostics.

## DIA-002: Explicit public facade and canonical imports

Existing top-level public names remain available. Internal consumers import
their defining modules. Former flat implementation routes are removed without
aliases, import hooks or duplicate implementations. Supported consumers and
notebook imports use the process-based paths documented in the diagnostic guide.

## DIA-003: Independent observation and unchanged artifacts

Numerical processes emit neutral observations through `contracts` without importing diagnostic implementations. Execution registers `observers=()` and prepares each built-in `EnergyObserver` from H and `-partial_t H`, with fresh per-run momentum starting at zero unless explicitly supplied. Accepted weighted stages or an accepted adaptive interpolant update the accumulator once; independent output queries never update it. Adaptive quadrature retains eight Gauss nodes; doubled GC retains physical signed weights and its sum-before-normalization arithmetic. JAX uses a pure device reduction without host callbacks or a retained history of all stages. Existing caller-owned callbacks keep identity, order and lifecycle, receiving the original dynamics identity. `states.energy.energy_diagnostics` combines supplied saved H and kappa, preserving physical conventions, keys, output destinations and archive formats. Existing GC/BM4 energy callbacks reuse this accumulator.