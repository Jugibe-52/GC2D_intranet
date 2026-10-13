# Specification-driven development

SDD connects substantial GC2D changes to explicit requirements, a reviewed plan,
and recorded verification. Begin with the relevant current contract, inspect
its implementation and tests, and specify the proposed difference before
changing scientific behavior. Small corrections may use the ordinary workflow.

## Entry points

Use the repository-local [sdd skill](../.agents/skills/sdd/SKILL.md):

| Operation | Result |
| --- | --- |
| `$sdd spec <request>` | Investigate and draft the change's `spec.md`: what and why |
| `$sdd plan <change-id>` | Derive `plan.md`: design, compatibility, and verification strategy |
| `$sdd tasks <change-id>` | Derive `tasks.md`: ordered tasks, dependencies, and concrete checks |
| `$sdd prepare <request>` | Prepare all three documents and present them together |
| `$sdd implement <change-id>` | Execute the accepted plan and tasks |
| `$sdd verify <change-id>` | Check criteria, record evidence, and reconcile current contracts |
| `$sdd status [change-id]` | Inspect state, revisions, pending work, and evidence without edits |

These are skill operations expressed in a prompt, not shell commands. To revise
an existing proposal, use its ID with `spec` or `prepare` and describe the change.
Separate preparation steps do not require intermediate approvals.
Natural-language requests are supported. If the request is only a consultation,
discuss the proposal without saving it. Automatic skill selection does not
grant permission to write. Follow the existing personal authorization rules;
an explicit skill invocation authorizes only the requested operation and scope.
Preparing a change never authorizes implementing it. An implementation request
that already accepts the reviewed specification and plan needs no second
confirmation. Plan mode remains read-only regardless of skill invocation.

Run the structural check from any working directory by passing the script path:

```bash
python scripts/check_sdd.py
python scripts/check_sdd.py --root /path/to/GC2D_codigo
```

The script defaults to its own repository root. It needs Python 3.11 or later
and only the standard library. CI runs the same check, independently of the
scientific tests. It never runs commands found in records or changes files.

## Current contracts and change history

The initial baseline is deliberately limited to:

| Contract | Requirement IDs | Existing detailed documentation |
| --- | --- | --- |
| [Potential import](current/potential-import.md) | POT-001 through POT-005 | [HDF5 import](../docs/dynamics/gc2d-h5-import.md) |
| [Simulation](current/simulation.md) | SIM-001 through SIM-004 | [Integration](../docs/simulation/integration-architecture.md), [execution](../docs/simulation/execution.md) |
| [Results](current/results.md) | SOL-001 through SOL-003 | [Integration and state ownership](../docs/simulation/integration-architecture.md) |

`current/` describes accepted, implemented behavior. Each change has a directory
`changes/<change-id>/`, conventionally `YYYY-MM-DD-short-description`, containing:

| File | Responsibility |
| --- | --- |
| `spec.md` | Problem, scope, exclusions, requirements, acceptance criteria, lifecycle state, and approval |
| `plan.md` | Design, compatibility, and verification strategy derived from the specification |
| `tasks.md` | Ordered tasks and dependencies, concrete verification procedures, execution evidence, and limitations |

Completed directories stay in place as history. Approval and state live only in
`spec.md`; task completion and results live only in `tasks.md`. Do not move an unimplemented proposal into the current contracts.
Local Spanish reading copies use `*.es.md` (or `*.es.<number>.md` for existing
numbered copies). The checker excludes these filenames in both `current/` and
`changes/`, independently of Git ignore rules. They are not records and cannot
replace a missing canonical document. Keep canonical documents in English and
maintain their Spanish copies under the shared personal translation instructions.
Detailed mathematical derivations and architecture keep their existing owners
under `docs/`; link to them instead of duplicating them here.

Requirement IDs are stable references, not line numbers. Allocate a new ID for
a distinct capability. Preserve an existing ID when refining the same contract.
For a removed requirement, retain its ID and a retirement explanation in the
current specification, with a manual compatibility verification procedure when
its old tests no longer apply. Preserve historical results when a file moves;
update reference paths and explain the move without rewriting past outcomes.

The [README HDF5 selection proposal](changes/2026-10-13-readme-hdf5-selection/spec.md)
is a draft awaiting review. Its existence does not authorize the correction.

## Lifecycle

| State | Meaning | Next action |
| --- | --- | --- |
| `draft` | Investigated proposal; unresolved questions may remain | Present specification and plan together |
| `approved` | User accepted the specification and plan, with tasks available, for this scope | Implement when the request authorizes it |
| `implementing` | Accepted work is in progress or validation is incomplete | Continue tasks and collect evidence |
| `verified` | All criteria checked, tasks completed, current contracts reconciled | Retain as history |

Record a concise approval date and description, never invent acceptance or copy
conversation transcripts. Approval is a recorded human decision, not something
the checker can authenticate. Re-read the record and actual diff when resuming.
An unfinished implementation resumes within its existing accepted scope.

When an authorized revision incorporates a material change to behavior, scope,
API compatibility, or acceptance criteria, return the record to `draft`:
explain the revision, clear `approval`, and reset affected checks to `pending`
with empty evidence/date. Preserve previous outcomes
in the narrative. A routine implementation choice within the accepted plan does
not require another approval. Discussing a possible expansion does not itself
authorize edits or invalidate the existing approval. Editing code after
verification invalidates the affected evidence; rerun those checks before
closing again. Failures and missing
dependencies remain visible and cannot be counted as passes.

## Revisions and staged preparation

A new draft may contain only `spec.md`, then add `plan.md`, then `tasks.md`.
Tasks require a plan. All three files must exist and agree before approval;
review the specification and plan together, with the tasks available for review.
Preparing any of these files does not authorize implementation.

`spec.md` and `plan.md` each have a positive integer `revision`. The plan records
its source `spec_revision`; tasks record both `spec_revision` and `plan_revision`.
Increment a document's revision when its substantive content changes. Reinspect
and update affected downstream documents before advancing their references:
changing the specification requires reviewing the plan and tasks; changing the
plan requires reviewing tasks. Never update revision numbers merely to silence
the checker. Spelling corrections and result/date updates need no new revision.

In a draft, a declared revision mismatch produces a warning so a staged revision
can remain open; `status` must identify the outdated documents. In approved or
later states it is an error. References to missing files and malformed metadata
remain errors even in drafts. The checker compares declared revisions, not
semantic changes in Git. The skill must detect content changes and evaluate
whether existing tasks and evidence still apply.

A spec-only revision that removes a criterion can leave old task checks with
invalid references. These remain errors, alongside revision warnings, until the
downstream phases are reconciled. Report the errors and next required phase;
do not silently finish an unrequested phase or advance its revision references.

A material scope change follows the lifecycle reset above. Routine design
refinements within an accepted scope may update plan/tasks without another
approval, after reconciling dependent work and resetting affected evidence.
For a completed change, prefer a new change directory for new work; migration
of record layout preserves historical approvals and outcomes.

## Record format

All scanned records start with TOML between two lines containing exactly `+++`,
followed by Markdown. The current-contract schema remains version 1; split
change documents use version 2. Legacy single-file change records must be
migrated; the checker reports their old locations instead of ignoring them.

Common metadata is `schema_version`, `kind`, `id`, `sources`, and `documents`.
Reference arrays contain existing repository-relative POSIX file paths. Unknown
metadata fields are errors. IDs use lowercase hyphen-separated words or numbers.
The current filename stem or change directory name equals `id`. The three parts
share one change ID; distinct contracts and changes must have unique IDs.
Templates live outside scanned directories and are not current contracts.

### Current specifications (version 1)

Use [contract.md](../.agents/skills/sdd/assets/contract.md), with `kind = "spec"`.
Each `[[requirements]]` table has `id` and `tests`. IDs use an uppercase domain
and at least three digits, such as `POT-001`, and are globally unique. Define
the behavior under a matching `## POT-001: Title` Markdown heading. Provide test
references or a non-empty `verification` string describing a manual procedure.
A reference locates evidence; it does not claim a test was just run or covers
every assertion.

Test references are Python files under `tests/`, optionally followed by
`::TestClass.test_method` or `::test_function`. Selectors name definitions directly
in that file; inherited or generated tests can use file references. The checker
inspects syntax without importing the test module.

### Change specification (version 2)

Use [spec.md](../.agents/skills/sdd/assets/spec.md), with `kind = "change-spec"`:

| Field | Meaning |
| --- | --- |
| `revision` | Positive specification revision |
| `status`, `approval` | Lifecycle state and acceptance note; draft approval must be empty |
| `requirements` | All current or proposed requirement IDs affected |
| `proposed_requirements` | Subset introduced here; may be absent from current contracts until verification |
| `open_questions` | Unresolved specification decisions |
| `[[criteria]]` | Unique local `id`, `description`, and affected `requirements` array |

Accepted specifications require non-empty Problem, Scope, and Proposed behavior
sections. A process-only change may have an empty requirement list. Requirements
can be proposed by only one change and must exist in current contracts before
that change can become verified.

### Implementation plan (version 2)

Use [plan.md](../.agents/skills/sdd/assets/plan.md), with `kind = "change-plan"`.
In addition to common metadata, use positive `revision` and `spec_revision`
integers and an `open_questions` array. Accepted plans require non-empty Design,
Compatibility, and Verification plan sections. Explain how the strategy tests
the specification's criteria; concrete procedures belong in the task record.

### Tasks and evidence (version 2)

Use [tasks.md](../.agents/skills/sdd/assets/tasks.md), with `kind = "change-tasks"`.
Add positive `spec_revision`, `plan_revision`, and an `open_questions` array:

| Table | Fields |
| --- | --- |
| `[[tasks]]` | Unique local `id`, `description`, `depends_on` array, Boolean `done` |
| `[[checks]]` | Unique local `id`, covered `criteria`, `procedure`, existing `tests`, `result`, `evidence`, and `checked_on` |

Use the prefixes `AC-`, `T-`, and `V-` followed by at least three digits for
criterion, task, and check IDs within a change, such as `AC-001`, `T-001`, and
`V-001`. List each task after its dependencies; independent tasks use `[]`.
Dependencies must name earlier tasks, excluding unknown references and cycles.
A completed task cannot have an unfinished dependency. Accepted task records
require non-empty Execution and Outcome and limitations sections. Use the prose
for context and history, without duplicating completion flags or check results.

Results are `pending`, `passed`, `failed`, or `blocked`. Pending evidence and date
may be empty strings. Every other result requires factual evidence and a quoted
ISO date (`"2026-10-13"`). Procedures can be commands or manual inspections; the
checker never executes them. Record counts, measured values/tolerances, relevant
environment, and inspected artifacts where useful.

At approval, all documents must have no open questions, and criteria, tasks, and
checks must be non-empty. Every affected requirement maps to a criterion, and
every criterion to a check. Verification requires all tasks done and all checks
passed with evidence. Failed required checks keep the change `implementing`.

New files described in a plan need not exist yet. Keep them in Markdown or a
procedure until creation, then add checked file references. Draft arrays may be
empty (`criteria = []`, `tasks = []`, `checks = []`) and decisions may stay open.
Supplied references must resolve even in drafts.

## Scientific review and validation

For a numerical change, state the applicable physical model, coordinates and
units, array shapes, tolerances, reference method, and measurable acceptance
criteria. Distinguish requested solver tolerances from measured errors. Reuse
existing deterministic tests where they cover the contract; add tests for
changed behavior or genuine gaps, not to duplicate implementation details.

The skill reviews semantic agreement among requirements, code, tests, and
results. The checker validates frontmatter and its references, not arbitrary
Markdown links, numerical correctness, approval authenticity, or test execution.
Passing it alone is insufficient to close a change. CI checks existing records;
it does not require every code diff to have a specification.

Follow the repository's focused tests and required CI checks. Run optional
backend comparisons when affected. Validate notebooks with a representative
reduced subset; do not overwrite canonical outputs with reduced results.
At closure, update the relevant current contracts and detailed documentation,
run the checker, and report remaining limitations explicitly. No commits,
pushes, deployments, or personal configuration changes are implied by SDD.
