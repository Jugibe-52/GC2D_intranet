---
name: review-lines
description: Explain and review a specific code line, short line range, editor selection, or pasted snippet, expanding block headers to their complete indented blocks, with focused improvements and optional brief code comments. Use for detailed local questions rather than whole-file reviews or sequential walkthroughs.
---

# Review Lines

Accept a file and line number or range, an editor selection, or a pasted snippet:

```text
$review-lines src/path/to/module.py:42
$review-lines src/path/to/module.py:42-48
```

## Scope and explanation

- Resolve the selected lines from the current file contents. If no selection can be identified, ask for the file and lines or a snippet.
- Unless the user explicitly limits the scope, expand a selection that touches a block header to the complete compound statement. This applies to any header opening an indented body, including `def`, `class`, `with`, `if`, `for`, `while`, `try`, `match`, and their asynchronous forms. Include the entire header even when it spans multiple lines, its full indented body and nested blocks, and associated branches such as `elif`, `else`, `except`, `finally`, and `case`. If the selected header is an associated branch, include the compound statement it belongs to.
- Determine block boundaries from syntax and indentation; blank lines do not end a block. Selecting an ordinary indented statement does not by itself select its enclosing block. Complete a selected multiline statement without expanding to unrelated enclosing code. For pasted snippets, use only the supplied content and identify any missing block content.
- State the expanded line range before explaining it, when file locations are available. Treat that range as the selected lines for the review and keep any explicit user scope limits.
- Read the enclosing function, definitions, contracts, or consumers as needed to understand the selection. Keep the review focused on the resolved selection, including any block expansion above; do not start a whole-file review or advance to another range automatically.
- Explain each substantive selected line: its operations, variables, types, purpose, and relevant assumptions. Keep trivial syntax explanations brief and group only blank lines or purely structural delimiters.
- Assess correctness, clarity, and useful simplifications. For numerical code, consider relevant shapes, units, signs, precision, and boundary conditions. Do not assume that an algebraic simplification preserves floating-point behavior.
- Distinguish supported problems, questions requiring more context or execution, and optional improvements. When the lines are appropriate, say so without inventing changes. For pasted snippets, state material context limitations and do not invent file locations.
- Show a small replacement snippet when a concrete improvement is useful, explain its benefit, and identify any behavior or API change. Do not claim tests or measured performance results without execution evidence.

## Brief code comments

- Suggest brief comments when requested or when they materially clarify a non-obvious operation. Explain why the operation exists, a physical meaning, units, an array shape or coordinate convention, or a numerical invariant.
- Avoid comments that merely restate syntax and avoid unsupported claims about physical meaning or units. Usually one short comment near the relevant operation is enough; do not annotate every line mechanically.
- Write proposed and applied project comments in English, following repository conventions. Use the user's language for the conversational explanation.

## Applying requested changes

- Default to explanation and proposals. Skill invocation alone does not authorize edits; follow the repository's `implements:` requirement.
- When implementation is authorized, apply only the requested local improvement or comments. A request to add comments authorizes comment edits, not accompanying logic changes.
- If a correction requires changes beyond the authorized scope, explain the dependency and propose the broader change separately.
- Validate in proportion to the change and repository requirements. Comment-only edits normally need inspection of the diff; executable changes need appropriate focused checks. Report what was checked and any remaining uncertainty.
- Finish with the selected lines' explanation, useful findings or changes, and optional comment suggestions. Keep the response proportional to the selection; no fixed quota of findings or mandatory report sections.
