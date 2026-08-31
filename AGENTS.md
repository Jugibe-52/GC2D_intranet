# Notebook scope

When inspecting or modifying notebooks, work only in `notebooks/developements/`
by default. Do not read or alter `notebooks/experiments/` unless the user
explicitly asks for it.

Experiment notebooks are versioned scientific artifacts and are outside the
supported interactive API.

# Notebook study policy

Keep notebooks focused on the scientific definition and interpretation of an
experiment. Parameters that affect reproducibility must remain explicit in the
notebook, including potential parameters and seeds, initial-condition geometry,
physical and numerical parameters, integration spans and steps, and sampling
choices.

Put reusable experiment composition in `src/studies/`. This includes
common potential and initial-condition construction, system assembly, parameter
validation, repeated-run orchestration, diagnostic extraction, and summaries.
Put reusable plotting and notebook display helpers in
`src/visualization/`. Put opt-in numerical observers and persistence in
`src/diagnostics/`. Do not duplicate project-root discovery, `sys.path`
mutation, observer lifecycle management, result-dictionary assembly, or
presentation helpers across notebooks.

Keep generic geometry, dynamics, potentials, numerical methods, and result
behavior in `src/initial_conditions/`, `src/dynamics/`, `src/potential/`, and
`src/simulation/`; studies should compose those APIs rather than reimplement
them. A notebook-local helper is appropriate only when its behavior is unique
to that study and would not provide stable reusable composition.

# Project language

Use English for all newly written or modified project content. This includes
documentation, comments, docstrings, user-facing and error messages, plot
labels and titles, and every notebook's Markdown, code text, and stored textual
outputs. Regenerate affected notebook figures and animations when they contain
non-English labels. When editing existing non-English prose, translate it into
English. Keep established identifiers stable unless a rename is explicitly
requested; proper names and mathematical notation do not require translation.

# Model architecture documentation

Keep architecture documentation separated by numerical model under
`docs/models/<model>/`. Each model owns independent `dynamics/` and
`simulation/` directories containing one Markdown explanation and its companion
PlantUML diagram. Mathematical LaTeX sources and their deliberate PDFs belong
at the model-directory root.

For the complete ABBA method family, the authoritative numerical-architecture
files are:

- `docs/models/abba/simulation/abba-numerical-architecture.md`; and
- `docs/models/abba/simulation/abba-numerical-architecture.puml`.

For the `ABBA2Implicit`-specific dynamics and mathematical derivation, the
authoritative files are:

- `docs/models/abba2-implicit/dynamics/gc2d-h5-import.md` and
  `gc2d-h5-potential-architecture.puml`;
- `docs/models/abba2-implicit/ABBA2_implicit.tex` and its compiled PDF.

Update the relevant model documents whenever code changes affect their public
API, dependencies, dynamics, initial configuration, simulation lifecycle,
numerical method, or result model. Do not recreate a global architecture
diagram unless that cross-model document is explicitly requested.

# Git tracking policy

Respect `.gitignore` when creating or modifying files. In particular, do not
use `git add --force` (or `git add -f`) to stage ignored files, and do not
change a file's tracked or ignored status, unless the user explicitly requests
that Git-tracking change. Only notebooks under `notebooks/experiments/` and
`notebooks/sympy/` are intended to be versioned. Development notebooks under
`notebooks/developements/` are local working files and must remain ignored.
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
