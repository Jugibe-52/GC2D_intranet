# Notebook scope

When inspecting or modifying notebooks, work only in `notebooks/developements/`
by default. Do not read or alter `notebooks/experiments/` unless the user
explicitly asks for it.

Experiment notebooks are versioned research artifacts and are outside the
supported interactive API.

# Project language

Use English for all newly written or modified project content. This includes
documentation, comments, docstrings, user-facing and error messages, plot
labels and titles, and every notebook's Markdown, code text, and stored textual
outputs. Regenerate affected notebook figures and animations when they contain
non-English labels. When editing existing non-English prose, translate it into
English. Keep established identifiers stable unless a rename is explicitly
requested; proper names and mathematical notation do not require translation.

# Git tracking policy

Respect `.gitignore` when creating or modifying files. In particular, do not
use `git add --force` (or `git add -f`) to stage ignored files, and do not
change a file's tracked or ignored status, unless the user explicitly requests
that Git-tracking change. Development notebooks are local working files: create
them freely, but leave them ignored unless the user asks to version them.
Before staging a newly created notebook, verify its status with
`git check-ignore --no-index <path>` when its intended tracking status is not
clear.

# Commenting style

Use a medium level of comments throughout the project:

- Add concise docstrings to modules, classes, and non-trivial methods.
- Comment mathematical steps, numerical algorithms, invariants, and decisions
  whose purpose is not immediately clear from the code.
- For important variables, state their physical or numerical meaning, expected
  shape and coordinate/block convention, plus units or normalization when that
  information is known. Keep this explanation near the declaration or in the
  containing docstring.
- Explain why a non-obvious operation is necessary rather than restating its
  syntax.
- Avoid line-by-line narration and comments on trivial assignments, accessors,
  or otherwise self-explanatory code.
- Keep comments accurate when behavior changes, and remove comments that no
  longer describe the implementation.
