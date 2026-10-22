+++
schema_version = 1
kind = "spec"
id = "results"
sources = ["src/states/history.py", "src/contracts/result.py", "src/simulation/runner.py", "src/contracts/comparison.py", "src/contracts/study_results.py", "src/diagnostics/comparison.py", "src/diagnostics/bm4_recurrence.py"]
documents = ["docs/simulation/integration-architecture.md"]

[[requirements]]
id = "SOL-001"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_invalid_method_output", "tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_runner_rejects_output_times_that_do_not_match_the_request"]

[[requirements]]
id = "SOL-002"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_solution_is_immutable_and_preserves_read_only_aliases", "tests/test_state_ownership.py::StateOwnershipTests.test_solution_uses_problem_snapshot_and_owns_its_layout"]

[[requirements]]
id = "SOL-003"
tests = ["tests/test_extensible_architecture.py::ExtensibleArchitectureTests.test_solution_is_immutable_and_preserves_read_only_aliases", "tests/test_prepared_constructor_contracts.py::PreparedConstructorContractTests.test_failed_diagnostic_preparation_does_not_seal_the_recorder"]
+++

# Sampled physical results

`States` is the canonical public result and the recording container. This contract covers in-memory storage; diagnostic meaning belongs to its producer. See [state ownership](../../docs/simulation/integration-architecture.md).

## SOL-001: Finite physical history and layout

Times have shape `(T,)` with at least one finite, strictly increasing sample. Physical values have shape `(physical_state_size,T)`, with finite, nonempty complete particle blocks accepted by the layout. No unit conversion occurs. Internal duplicated coordinates are not stored as physical results. `States` validates structure; the simulation boundary checks agreement with the particular request, layout dimension and initial state. Simulation requests still require both endpoints and at least two samples.

## SOL-002: Owned arrays, layout and initial snapshot

Recording copies samples; completed arrays are read-only. Construction from complete arrays preserves their numerical dtype. The layout is copied, and its accessor returns an independent copy. The optional initial array is copied, dimension-checked and independently returned. Simulated results preserve the problem's initial snapshot; loaded archives may lack it. There is no retained mutable provider identity. Read-only NumPy flags describe the supported interface, not protection against deliberate low-level mutation. Pickling a completed result restores these guarantees; incomplete results cannot cross that boundary.

## SOL-003: Read-only diagnostics and explicit axes

Diagnostic keys are nonempty strings and values are scalars or NumPy arrays. The public mapping cannot be modified; diagnostic arrays are copied and read-only. Sample diagnostics follow the requested times; accepted-step diagnostics follow their own `step_times`, `step_start_times` and `step_sizes`. `select` keeps whole-run diagnostics unchanged. `.with_diagnostics(...)` makes a completed copy with added metadata or explicitly replaced diagnostics. `.t`/`.states` remain read-only aliases for `.times`/`.values`; `.n_steps`, `.k` and `.err` remain convenience accessors. `.y`, `.trajectory` and `.source` are removed.

Finalization delegates diagnostic validation and copying to a private preparation
function. Rejected diagnostics leave the recorder open with no published
diagnostics, so valid diagnostics can be supplied in a later finalization call.

## Shared study records and analysis

Comparison and recurrence records remain in `contracts.comparison` and
`contracts.study_results`. They own data validation and storage; reference RMS
floors and energy reductions belong to `diagnostics.comparison`, and periodic
initial-return distances to `diagnostics.bm4_recurrence`. Formulas, units and
archive schemas are unchanged. Readers do not depend on study orchestration.
`tests/test_comparison_diagnostics.py` checks known reductions, periodic wrapping,
input/output ownership and independent record imports. Archive round trips are
covered by `tests/test_study_result_contracts.py`.
