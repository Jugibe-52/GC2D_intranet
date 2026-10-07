---
name: explain-syntax
description: Explain the syntax of a specific code line, editor selection, or pasted snippet by breaking down its tokens and grammatical structure. Use for questions about how a line is written and parsed, rather than algorithmic, numerical, or whole-file reviews.
---

# Explain Syntax

Accept a file and line number, a short range, an editor selection, or a pasted snippet:

```text
$explain-syntax src/path/to/module.py:42
$explain-syntax src/path/to/module.py:42-45
$explain-syntax values[:, None]
```

## Resolve the selection

- Read selected lines from the current file. If no selection can be identified, ask for a file and line number or a snippet. Do not invent file locations for pasted code.
- Complete a selected multiline expression or statement only as needed to explain its syntax. For block headers, include the complete header but do not expand to the indented body or associated branches. Explain the colon and indentation requirement without reviewing the block.
- State the resolved file and line range when available. Respect explicit scope limits; if necessary syntax lies outside them, describe the missing context instead of silently expanding the selection.
- Inspect nearby definitions or imports only when needed to disambiguate a construct or identify an object's type. For pasted snippets, state any material uncertainty rather than assuming missing context.

## Explain the syntax

- Start with a brief plain-language reading of the complete line. Then break it into meaningful parts, identifying keywords, identifiers, literals, operators, and delimiters and explaining each part's grammatical role.
- Explain how the parts combine: grouping, precedence, associativity, and evaluation order when relevant. Distinguish precedence from evaluation order and mention short-circuiting or conditional evaluation when it affects the reading.
- Describe punctuation in its actual context. For example, distinguish a type annotation colon from a slice or block-header colon, and a keyword argument from an assignment. Avoid a generic catalog of unrelated syntax.
- Separate language syntax from library or object behavior. For `values[:, None]`, explain Python subscription and slice notation separately from NumPy's interpretation of `None` as a new axis; only attribute NumPy behavior when the type is established or explicitly assumed.
- Explain just enough behavior to make the syntax understandable. Do not turn the answer into a review of physical meaning, numerical correctness, architecture, performance, or refactoring unless the user also requests it.
- When useful, give one small, self-contained example with simpler values or names. Label illustrative rewrites as such and do not claim equivalence if evaluation, types, or side effects may differ. Do not imply that an example was executed unless it was.
- If the syntax is invalid, identify the specific issue and propose the smallest correction. Distinguish a partial snippet requiring context from an invalid construct, and syntax errors from runtime or type errors.
- Use the user's language for the explanation, with English for project content and code comments. Keep the detail proportional to the selected syntax and the user's question; no fixed report sections or mandatory findings.

## Changes

- Default to explanation and proposals. Invocation alone does not authorize edits; follow the repository's `implements:` requirement.
- When an edit is explicitly authorized, apply only the requested local syntax correction or clarification and validate in proportion to the change. Do not introduce unrelated logic changes or comments.
