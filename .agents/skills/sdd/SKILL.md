---
name: sdd
description: Prepare or carry out substantial GC2D changes using short, readable documents. Use for functionality, numerical methods, public behavior, and studies; small fixes and explanations can use the ordinary workflow.
---

# Plan and carry out a GC2D change

Read the repository instructions, memory, and [change guide](../../../specs/README.md).
Inspect only the code, tests, and existing documents relevant to the request.

## Two operations

- **`prepare` / `prepara`:** investigate the request and write the first four
  documents below. Present them together and stop before implementation.
- **`execute` / `ejecuta`:** implement the prepared proposal, check the outcome,
  and create the fifth document. An explicit request to execute the proposal
  already presented to the user accepts that scope and authorizes its
  implementation; do not ask for the same approval again.

Accept equivalent natural language. Find an existing change before creating
another; infer the target from the conversation when unambiguous. Questions
about progress are read-only, even when the skill is named. Follow the personal
permission rules: choosing this skill yourself does not grant write permission,
and editing the skill does not authorize running it. Preparation alone never
means implementation is accepted.

## Five short documents

Use one folder under `specs/changes/`, named `YYYY-MM-DD-short-description`.
Use these templates as writing prompts, replacing their guidance with concrete
content. Choose headings that help the reader; there are no mandatory section
names, requirement codes, metadata blocks, or revision numbers.

| File | Explain |
| --- | --- |
| [goal.md](assets/goal.md) | The problem and why it is worth solving. |
| [proposal.md](assets/proposal.md) | What will change and how it should work, with an example when helpful. |
| [impact.md](assets/impact.md) | Which parts are affected, directly or indirectly, and why. Include relevant tests, documentation, notebooks, ways of using the program, and saved data. |
| [steps.md](assets/steps.md) | A short, ordered checklist of concrete work, including how to check the proposed behavior. |
| [results.md](assets/results.md) | What was implemented, what was checked, the observed results, and anything unfinished or limited. Create this during execution. |

Write brief, direct prose in the project's required language and maintain its
Spanish reading copies. Use everyday words; explain an unavoidable technical
term on first use. Mention files or functions only when they clarify the impact
or the work. Avoid repeating information across documents. Link to existing
mathematical and architecture explanations instead of copying them.

For scientific changes, retain the physical assumptions, units, coordinates,
array shapes, tolerances, and reference accuracy that determine correctness.
Explain their significance. Distinguish requested solver tolerances from
measured errors.

## Prepare

Investigate the current behavior and likely consequences. Fill in `goal.md`,
`proposal.md`, `impact.md`, and `steps.md`; do not create `results.md` for a new
proposal. Describe the intended outcome and how it will be checked before any
implementation. Keep unresolved decisions visible and ask only about information
that materially affects the proposal and cannot be inferred.

When revising a proposal, reconcile all four documents and preserve valid user
edits. Keep any previous results as history, explaining which conclusions will
need fresh checks if the revised work is executed. Preparation must not change
implementation files or describe an unperformed check as passed.

Run `python scripts/check_sdd.py` after saving the documents. Present the goal,
main change, affected parts, and steps together with links to the four files.
Do not introduce separate approvals for each file.

## Execute

Read all the prepared documents and inspect the actual working tree. If the
proposal is missing or has not been presented, prepare and present it first;
continue only when the scope has been accepted and implementation authorized.
Honor acceptance and authorization already given for that scope. Resolve
material unanswered questions before depending on them.

Carry out the steps and mark real progress in `steps.md`. Create `results.md`
when execution begins and update it with the implementation and actual checks.
Record the acceptance date and scope in a short sentence there. Keep incomplete
work and failed checks visible; a results file does not mean the work succeeded.
Resume accepted work without repeating approvals or completed steps.

Use the repository checks appropriate to the change and cover the promised
behavior. Reuse confirmed results for unchanged work. For numerical changes,
record measured discrepancies and relevant conditions; use reduced notebook
checks under the repository policy. Update affected permanent documentation and
`specs/current/` to describe accepted, implemented behavior, preserving their
existing format. Keep the change folder as history.

Explain results in ordinary language, retaining enough procedure, command, or
measurement detail to reproduce a check when needed. Report completion only
when the promised behavior and relevant checks are satisfied. Finish with
`python scripts/check_sdd.py`; it checks document structure, not scientific
correctness or whether a recorded check actually ran.

A material change to the agreed scope needs a revised proposal and review;
routine implementation decisions do not. Do not publish, commit, push, or change
personal configuration as part of this workflow.

## Existing history

Keep older `spec.md`, `plan.md`, and `tasks.md` records and their outcomes intact;
there is no bulk migration. When an unfinished older change is explicitly
resumed, read and update its existing documents, retaining valid acceptance and
evidence. Use the five-file format for new changes. A status request reads the
available format and explains progress, outstanding decisions, and next steps
without changing files or running the recorded checks.
