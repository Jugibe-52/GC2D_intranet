+++
schema_version = 1
kind = "spec"
id = "diagnostics"
sources = ["src/diagnostics/__init__.py", "src/diagnostics/methods/implicit_iterations.py", "src/diagnostics/methods/projection/observer.py", "src/diagnostics/states/geometry.py", "src/diagnostics/io/persistence.py"]
documents = ["src/diagnostics/README.md", "AGENTS.md"]

[[requirements]]
id = "DIA-001"
tests = ["tests/test_package_layout.py::PackageLayoutTests.test_diagnostics_follow_processes_without_retired_flat_modules"]

[[requirements]]
id = "DIA-002"
tests = ["tests/test_package_layout.py::PackageLayoutTests.test_diagnostic_public_names_resolve_to_process_implementations", "tests/test_package_layout.py::PackageLayoutTests.test_public_exports_are_explicit_and_resolve"]

[[requirements]]
id = "DIA-003"
tests = ["tests/test_package_layout.py::PackageLayoutTests.test_numerical_processes_do_not_load_diagnostic_implementations", "tests/test_diagnostic_buffering.py", "tests/test_solution_persistence.py"]
+++

# Process-owned diagnostics

## DIA-001: Locate implementations by the observed process

`diagnostics` follows `src` responsibilities through `execution`, `methods`,
`methods`, `states` and `studies`; shared output and transport belong to
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

Numerical processes emit neutral observations through `contracts` and do not
depend on diagnostic implementations. The reorganization preserves observation
timing, numerical formulas, physical conventions, diagnostic keys, output
destinations and archive formats. Study-specific artifact formats remain
independent of study orchestration; shared persistence writes completed `States`.
