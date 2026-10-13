---
name: sdd
description: Prepare, implement, verify, or inspect specification-driven changes in GC2D. Use for substantial behavior, public-contract, numerical-algorithm, and study changes, or explicit SDD requests. Small corrections and read-only code explanations do not require the full workflow.
---

# GC2D specification-driven development

Read the repository instructions and memory, then the
[SDD guide and record schema](../../../specs/README.md). Read only the current
contracts and existing change records relevant to the request. The guide owns
the lifecycle and schema; the templates provide the editable document shape.

## Select the operation

| Operation | Requested work |
| --- | --- |
| `$sdd spec <request>` | Investigate and write the proposed behavior in `spec.md` |
| `$sdd plan <change-id>` | Derive the implementation design in `plan.md` |
| `$sdd tasks <change-id>` | Derive ordered tasks and checks in `tasks.md` |
| `$sdd prepare <request>` | Perform `spec`, `plan`, and `tasks` together |
| `$sdd implement <change-id>` | Execute accepted work |
| `$sdd verify <change-id>` | Validate and reconcile accepted work |
| `$sdd status [change-id]` | Inspect progress without edits |

Accept equivalent natural language. To revise an existing proposal, accept its
ID plus the requested revision with `spec` or `prepare`. Use `prepare` for an
unplanned substantial change. Find an existing record before creating another;
if several match, clarify the target. Read all existing parts when resuming.

Invocation authorizes only the requested operation, within existing permissions.
Selecting this skill implicitly does not bypass the personal `&` rule. Editing
the skill does not authorize executing its workflow. Consultation and Plan mode
remain read-only. `status` is always read-only.

## Prepare, or work through individual preparation steps

Read the guide's schema and revision rules. Save authorized proposals under
`specs/changes/<change-id>/`, using an unused date-and-description ID. A draft
may be incomplete: spec alone, then plan, then tasks. Separate operations do not
introduce approval gates between these steps. When a prerequisite is missing,
report the missing document and next operation; do not invent an accepted plan.

- **Spec:** inspect affected implementation, tests, and documentation. Use
  [assets/spec.md](assets/spec.md) for problem, scope, exclusions, proposed
  behavior, and measurable criteria. Distinguish current behavior from desired
  changes and unresolved facts. Reference existing requirements; allocate unused
  domain IDs through `proposed_requirements` for genuinely new contracts.
  Include units, coordinates, shapes, tolerances, physical conditions, and
  reference accuracy where they determine numerical correctness.
- **Plan:** read the current proposal; use [assets/plan.md](assets/plan.md) for
  design, compatibility, and the verification strategy. Record the specification
  revision actually reviewed. Explain how checks will establish each criterion.
  Essential unresolved decisions may remain explicit in a draft.
- **Tasks:** read both preceding documents; use [assets/tasks.md](assets/tasks.md)
  for concrete ordered tasks, their dependencies, and reproducible checks tied to
  criteria. Record the reviewed specification and plan revisions. If the plan is
  stale, reconcile it only when that work is authorized; otherwise report the
  prerequisite. Retain valid completion and evidence when regenerating tasks.
- **Prepare:** carry out those three steps, reconciling existing records instead
  of replacing their history. Resolve decisions that can be inferred from the
  request; ask only for material missing information.

For a substantive spec or plan edit, increment its revision and inspect the
impact on downstream documents and evidence. Leave unreconciled downstream
revision references unchanged so `status` exposes them. Preserve prior outcomes
in tasks.md and reset affected completion/checks; a material scope revision also
returns spec.md to draft and clears approval. Do not merely change revision
numbers to make validation pass. An operation limited to one phase does not
silently complete the later phases or authorize implementation.

If a spec revision removes a criterion still referenced by tasks, report the
resulting reference errors as well as revision warnings. Identify the pending
plan/tasks reconciliation; do not complete unrequested phases merely to make
the checker pass. Keep the proposal in draft while that work remains.

Keep supplied file references valid; planned new files belong in prose until
they exist. Use [assets/contract.md](assets/contract.md) only for an accepted,
implemented current contract, not for a proposal. Run the structural checker on
saved records and report draft revision warnings. Present prepared specification
and plan together, with the tasks available, for the single review before
implementation. Stop at draft unless the user has already accepted this exact
scope and authorized implementation. In consultation, present the concrete
proposal in the conversation without writing records.

## Implement

Read spec.md, plan.md, and tasks.md and inspect the actual working-tree changes.
Confirm that the user has accepted the specification and plan and authorized this
implementation. An explicit implementation request accepting a previously
reviewed plan satisfies this step; do not ask again. If acceptance is missing,
present the concrete proposal and request the missing decision, citing the
guide's lifecycle. Never fill `approval` based only on a pre-existing status.

Ensure all three parts agree and checks cover every criterion. Record the
acceptance date and scope in spec.md, clear resolved questions in all parts, and
use `approved`, then `implementing` as work starts. Resume unfinished accepted
work without repeating completed tasks or discarding user edits. If code and
the recorded plan disagree, investigate before choosing which to change.

Carry out the accepted tasks and validation, recording progress in tasks.md.
Incorporating an authorized material scope or contract revision returns to
`draft` for review as described in the guide. Discussing a possible expansion
alone does not authorize record edits or
invalidate the existing approval. Routine technical choices within scope do not
require renewed approval. Update affected evidence after edits. Keep failed or
blocked checks visible and leave the change `implementing` until they pass.
Continue directly into verification when it is part of the authorized work.

## Verify

Match every criterion to the diff and observed evidence. Run its checks and
the repository-required validation appropriate to the change. For a check
already run against the unchanged implementation, reuse the confirmed result;
do not rerun expensive integrations merely to repeat a status. Never treat a
recorded command or a passing structural check as proof it was executed.

Record each actual result, date, procedure and concise evidence in tasks.md,
including measured numerical discrepancies and limitations. Reconcile current
contracts and affected model documentation with accepted, implemented behavior. New
requirement IDs must exist in current contracts before verification completes.
Keep the change directory as history. Set spec.md to `verified` only when all
tasks and checks are complete and semantic review agrees; run the structural
checker again.

Report the completed behavior and validation, or the exact outstanding work.
Do not silently waive failed checks, publish, commit, push, install services, or
change personal configuration as part of this workflow.

## Status

Read all available parts, or list change IDs and states if no ID is supplied.
Run the checker when useful. Report missing phases, stale revision references,
pending tasks, decisions, failed or missing evidence, and the next action. Do not
edit metadata, grant approval, run the recorded verification procedures, or begin
implementation merely to report status. Distinguish recorded results from checks
performed in this operation.
