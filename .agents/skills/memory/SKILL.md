---
name: memory
description: Maintain repository MEMORY.md with confirmed reusable development patterns, decisions, and corrections. Use when asked to remember project knowledge or when authorized implementation reveals a durable lesson.
---

# Memory

Maintain concise project knowledge that helps future tasks follow established
development patterns. Use the repository-root `MEMORY.md` as the single memory
file, preserving useful existing entries.

## Authorization and evidence

- Read the applicable `AGENTS.md` and existing `MEMORY.md` before updating memory.
  Follow the repository's `implements:` requirement. During consultation,
  present proposed entries without modifying files. Skill invocation alone
  does not authorize writing.
- During an authorized implementation, record relevant durable lessons within
  that task's scope. Do not expand the implementation to unrelated changes.
- Record explicit user decisions, confirmed corrections, and verified reusable
  patterns. State their context and scope; do not turn one experiment's choice
  into a universal requirement without evidence.
- Do not store speculation, unaccepted proposals, temporary progress, raw logs,
  credentials, or conversation transcripts. If nothing reusable was learned,
  leave memory unchanged.

## Updating memory

- Write in English. Group entries by topic and update the matching entry rather
  than appending a chronological history. Explain the actionable pattern and
  its reason when that reason is useful for future decisions.
- Preserve unrelated valid entries. Replace superseded guidance when supported
  by an explicit correction or verified change; do not resolve ambiguity by
  inventing a rule.
- Avoid repeating `AGENTS.md` or detailed project documentation. Use concise
  repository-relative references to existing canonical documentation where
  helpful, without creating extra memory files to bypass the size limit.
- Keep the complete file at **300 lines or fewer**, counting headings, blank
  lines, and any final line without a newline. Count the proposed content before
  saving; first merge duplicates, then condense wording and remove obsolete or
  low-value entries while preserving active decisions and numerical meaning.
  Never truncate blindly or drop valid constraints simply to make room.
- If a meaningful addition cannot fit without losing essential existing
  information, leave the file unchanged and explain the conflict to the user.

## Verification

- After saving, count lines with `len(path.read_text().splitlines())` or an
  equivalent check and confirm the total does not exceed 300.
- Review the diff for unintended omissions, unsupported claims, duplicated
  rules, and changes outside the authorized scope.
- Briefly report what was remembered or consolidated and the final line count.
