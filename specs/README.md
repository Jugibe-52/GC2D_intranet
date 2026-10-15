# Specification-driven development

SDD connects substantial GC2D changes to explicit requirements, a reviewed plan,
and recorded verification. Begin with the relevant current contract, inspect
its implementation and tests, and specify the proposed difference before
changing scientific behavior. Small corrections may use the ordinary workflow.

## Entry points

Use the repository-local [sdd skill](../.agents/skills/sdd/SKILL.md):

- `$sdd prepare <request>` investigates and prepares a change for review.
- `$sdd implement <change-id>` implements an accepted specification and plan.
- `$sdd verify <change-id>` checks acceptance criteria and closes completed work.
- `$sdd status [change-id]` reports progress and missing evidence without edits.

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

`current/` describes accepted, implemented behavior. `changes/` contains one
Markdown file per change, named after its ID, conventionally
`YYYY-MM-DD-short-description`. Completed records stay in that directory as
history. Do not move an unimplemented proposal into the current contracts.
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
| `approved` | User accepted both for this scope | Implement when the request authorizes it |
| `implementing` | Accepted work is in progress or validation is incomplete | Continue tasks and collect evidence |
| `verified` | All criteria checked, tasks completed, current contracts reconciled | Retain as history |

Record a concise approval date and description, never invent acceptance or copy
conversation transcripts. Approval is a recorded human decision, not something
the checker can authenticate. Re-read the record and actual diff when resuming.
An unfinished implementation resumes within its existing accepted scope.

When an authorized revision incorporates a material change to behavior, scope,
API compatibility, or acceptance criteria, return the record to `draft`:
explain the revision, clear `approval`, and reset
affected checks to `pending` with empty evidence/date. Preserve previous outcomes
in the narrative. A routine implementation choice within the accepted plan does
not require another approval. Discussing a possible expansion does not itself
authorize edits or invalidate the existing approval. Editing code after verification invalidates the
affected evidence; rerun those checks before closing again. Failures and missing
dependencies remain visible and cannot be counted as passes.

## Record format (version 1)

All scanned records start with TOML between two lines containing exactly `+++`,
followed by Markdown. Use the [specification template](../.agents/skills/sdd/assets/spec.md)
or [change template](../.agents/skills/sdd/assets/change-spec.md). Templates are outside
the scanned directories and are not current contracts.

Common metadata: `schema_version = 1`, `kind` (`spec` or `change`), `id`,
`sources`, and `documents`. The two reference arrays contain existing files,
using repository-relative POSIX paths. The filename stem equals `id`; document
IDs are unique throughout both directories. Unknown metadata fields are errors.

### Current specifications

Each `[[requirements]]` table has `id` and `tests`. IDs use an uppercase domain
and at least three digits, such as `POT-001`, and are globally unique. Define
the behavior under a matching `## POT-001: Title` heading in the Markdown.
For every requirement, provide test references or a non-empty `verification`
string describing a manual procedure. Existing tests are references to evidence
locations, not a claim that those tests have just run or cover every assertion.

A test reference is an existing Python file under `tests/`, optionally followed
by `::TestClass.test_method` or `::test_function`. Selectors must name definitions
directly in that file; inherited or generated tests can use a file reference.
The checker inspects syntax without importing the test module.

### Changes

In addition to common fields, a change contains:

| Field | Meaning |
| --- | --- |
| `status`, `approval` | Lifecycle state and acceptance note; approval is empty before acceptance |
| `requirements` | All current or newly proposed requirement IDs affected by the change |
| `proposed_requirements` | Subset introduced by this change; may be absent from current contracts until verification |
| `open_questions` | Unresolved decisions; must be empty after approval |
| `[[criteria]]` | Unique local `id`, `description`, and affected `requirements` array |
| `[[tasks]]` | Unique local `id`, `description`, and Boolean `done` |
| `[[checks]]` | Unique local `id`, covered `criteria`, `procedure`, existing `tests`, `result`, `evidence`, and `checked_on` |

Use `AC-001`, `T-001`, and `V-001` for local criterion, task, and check IDs.
Results are `pending`, `passed`, `failed`, or `blocked`. Pending evidence and date
may be empty strings. Every other result requires a concise factual evidence
description and a quoted ISO date (`"2026-10-13"`). Procedures may be commands or
manual inspections; no command is executed by the checker. Record counts,
measured values/tolerances, relevant environment, and inspected artifacts in the
evidence as appropriate. Do not replace observed results with expected results.

At approval, require non-empty criteria, tasks, and checks; every affected
requirement must map to a criterion, and every criterion to a check. A process-only
change may have an empty requirement list. Before verification, all tasks must
be done, every check must have passed with evidence, and new requirements must
exist in the current contracts. Requirements can be proposed by only one change.

New files described in the plan need not exist yet. Keep them in the Markdown
design or procedure; add them to checked file references only after creation.
Drafts may have empty table arrays (`criteria = []`, `tasks = []`, `checks = []`)
and explicit open questions. References supplied even in drafts must resolve.

Accepted changes use the template's non-empty sections: Problem, Scope,
Proposed behavior, Design, Compatibility, Verification plan, and Outcome and
limitations. Structured metadata owns task completion and verification status;
avoid maintaining a second checklist with conflicting states in the prose.

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
