# Preparing and carrying out changes

Use the [sdd skill](../.agents/skills/sdd/SKILL.md) for substantial changes to
GC2D. Small corrections can be made and checked without a change folder.
The aim is to agree on the intended result, understand what it affects, and
check the work afterwards.

## Prepare

Ask `$sdd prepare <request>` or `$sdd prepara <request>`. The skill investigates
the request and creates four short documents in
`specs/changes/YYYY-MM-DD-short-description/`:

| File | What you will find |
| --- | --- |
| `goal.md` | The problem and why we want to solve it. |
| `proposal.md` | What will change, how it should work, and an example if useful. |
| `impact.md` | The affected parts and consequences, including related uses, tests, documentation, notebooks, and saved data where relevant. |
| `steps.md` | A short checklist of work and how to check the expected behavior. |

Read the four documents together. Preparation stops here: it does not change
the implementation or create a results file. Important unanswered questions
are stated plainly. Revising a proposal updates these documents together.

## Execute

Ask `$sdd execute <change>` or `$sdd ejecuta <change>` after reviewing the
proposal. This accepts the presented scope and requests its implementation;
there is no additional approval for each file or routine technical decision.
The change can be identified by its folder name or unambiguous conversation
context. Equivalent natural-language requests are also supported.

Execution marks progress in `steps.md` and creates the fifth document,
`results.md`. It records what was implemented, how it was checked, the actual
outcome, and any remaining work or limitations. It includes a short sentence
with the acceptance date and scope. If work stops or a check fails, the document
says so; its existence does not imply success.

If no proposal has been presented, the skill prepares one for review before
implementation, unless that exact scope is already accepted and authorized.
A material scope change requires review; existing acceptance remains valid for
its original scope. Asking about progress is read-only. Creating or editing the
skill does not run this workflow. Personal authorization rules still apply.

## Keep the documents easy to read

Use short paragraphs, useful examples, and a short checklist. Each file has a
single purpose. No requirement codes, metadata blocks, revision counters, or
fixed section headings are needed. The [templates](../.agents/skills/sdd/assets/)
are writing aids, not forms to fill with boilerplate.

Use everyday language and explain necessary technical terms. Scientific work
still needs clear physical assumptions, units, coordinates, array shapes,
tolerances, and reference accuracy when these affect the result. Explain why
those details matter and distinguish measured errors from requested tolerances.
Put planned checks in `steps.md` and their observed outcomes in `results.md`.

Canonical documents use English under the project language rule. Local Spanish
reading copies use `name.es.md`; they are auxiliary copies of the same five
documents and are excluded from Git and from the checker.

## Permanent documentation and history

Keep detailed mathematical and architecture explanations in `docs/` and link to
them. Update affected descriptions in `specs/current/` only for accepted,
implemented behavior, retaining their existing format and stable references.
New change documents do not need to copy that format or repeat those details.

Keep completed change folders as history. Older changes with `spec.md`,
`plan.md`, and `tasks.md` remain readable and checked in their existing format.
There is no automatic conversion. When resuming an unfinished older change,
preserve its acceptance and results and update its existing documents.

## Document check

After saving changes, run:

```bash
python scripts/check_sdd.py
```

From another working directory, use the script's absolute path; `--root` can
select a different repository. The script uses Python 3.11 or later and the
standard library. CI runs the same check.

For the new format, the checker checks the four preparation files, optional
`results.md`, and readable content, without requiring fixed headings or parsing
prose as approval or proof. It rejects mixed old/new formats within one folder
and keeps the existing checks for older records and current specifications.
It does not edit files or execute recorded commands. A passing result does not
establish scientific correctness or prove that implementation and checks ran.
