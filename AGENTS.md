# Notebook scope

When inspecting or modifying notebooks, work only in `notebooks/developements/`
by default. Do not read or alter `notebooks/experiments/` unless the user
explicitly asks for it.

Experiment notebooks are versioned research artifacts and are outside the
supported interactive API.

# Commenting style

Use a medium level of comments throughout the project:

- Add concise docstrings to modules, classes, and non-trivial methods.
- Comment mathematical steps, numerical algorithms, invariants, and decisions
  whose purpose is not immediately clear from the code.
- Explain why a non-obvious operation is necessary rather than restating its
  syntax.
- Avoid line-by-line narration and comments on trivial assignments, accessors,
  or otherwise self-explanatory code.
- Keep comments accurate when behavior changes, and remove comments that no
  longer describe the implementation.
