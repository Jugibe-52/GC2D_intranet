+++
schema_version = 1
kind = "spec"
id = "results"
sources = ["src/solution.py", "src/contracts/result.py", "src/simulation/runner.py"]
documents = ["docs/simulation/integration-architecture.md"]

[[requirements]]
id = "SOL-001"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_invalid_method_output", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_output_times_that_do_not_match_the_request"]

[[requirements]]
id = "SOL-002"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_solution_is_immutable_and_preserves_read_only_aliases", "tests/test_state_ownership.py::StateOwnershipTests.test_solution_uses_problem_snapshot_and_owns_its_layout"]

[[requirements]]
id = "SOL-003"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_solution_is_immutable_and_preserves_read_only_aliases"]
+++

# Sampled physical results

`Solution` is the canonical public result. This baseline covers its in-memory
storage contract, not persisted archive formats or the meaning of each optional
diagnostic. See [state ownership](../../docs/simulation/integration-architecture.md).

## SOL-001: Finite physical history and layout

Times have shape `(T,)` with at least two finite, strictly increasing samples.
States have shape `(physical_state_size, T)`, with nonempty finite values and
complete particle blocks accepted by the supplied layout. Components use the
layout's block convention and the physical dynamics' coordinate normalization;
the result does not convert units. Internal doubled coordinates are not the
public physical history. Agreement with a particular request and initial state
is owned by the simulation boundary, not by `Solution` alone.

## SOL-002: Owned arrays, layout snapshot, and provenance

Construction copies time and state arrays and exposes them as read-only.
Subsequent edits to input arrays do not change the result. The layout is copied,
and the layout accessor returns an independent copy. When an initial state is
present it is copied, validated against the history's first dimension, and
returned through an independent copy. `source` retains the original initial
configuration as provenance; it is not itself deeply frozen. Simulated results
use the problem's captured layout and initial state even if that source later
changes. Read-only NumPy flags describe the supported interface, not protection
against deliberate low-level mutation.

## SOL-003: Read-only diagnostic mapping and array values

Diagnostic keys are nonempty strings. The mapping cannot be modified through
the public accessor; NumPy diagnostic arrays are copied and exposed as read-only.
This contract does not claim recursive freezing of arbitrary non-array objects.
Diagnostic presence and numerical meaning belong to the producing method or
observer. Existing `y`, `trajectory`, `n_steps`, `k`, and `err` compatibility
properties remain read-only; adding SDD does not remove or rename them.
