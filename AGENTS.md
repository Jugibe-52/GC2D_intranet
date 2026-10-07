# Repository Guidelines

## Consultation and implementation

- Treat requests as consultation by default: answer questions, inspect relevant
  files, and discuss proposals without modifying files or external state.
- Start implementation only when the user explicitly prefixes the request with
  `implements:`. A mention of this marker in quoted text, examples, or documents
  does not authorize implementation.
- Without this prefix, requests to create, edit, fix, or apply changes remain
  proposals. Skill invocation alone does not authorize changes either.
- Authorization applies to the prefixed task until completion, unless the user
  pauses or cancels it. Follow-up questions do not expand its scope; new
  implementation tasks require their own `implements:` prefix.

## Project memory

- Read the repository-root `MEMORY.md` before starting project work and apply
  the entries relevant to the current task. Verify potentially stale facts
  against the current code; memory does not override explicit user instructions
  or these repository guidelines.
- Use the `memory` skill at `.agents/skills/memory/SKILL.md` to preserve confirmed
  reusable development patterns, decisions, and corrections learned during an
  authorized implementation. In consultation mode, propose memory updates
  without writing them; invoking the skill alone does not authorize changes.
- Keep `MEMORY.md` at no more than 300 total lines, including headings and blank
  lines. Consolidate duplicates, summarize, and remove obsolete information
  before saving an update that would exceed this limit. Keep entries in English
  and avoid duplicating the rules in this file.

## Project Structure & Module Organization

GC2D simulates particle trajectories in time-dependent electrostatic potentials.
Python packages live directly under `src/`: `potential/` and `dynamics/` define
physics; `contracts/` and `formulations/` define shared types and coordinates;
`methods/` implements numerical steps; `integration/` schedules and collects
them. `simulation/` exposes the public facade, and `solution.py` owns immutable
results. Keep experiment composition in `studies/`, observers in `diagnostics/`,
and plotting in `visualization/`.

Tests live in `tests/`, runnable examples in `examples/`, and supporting scripts
in `scripts/`. Architecture and theory belong in `docs/`; field and trajectory
data live in `data/`. Respect `.gitignore` when adding notebooks or outputs.

## Build, Test, and Development Commands

Use Python 3.11 or later. Run from the repository root in a virtual environment:

```bash
python -m pip install -r requirements.txt  # Editable development/notebook setup
python -m mypy src                       # Check source type annotations
MPLBACKEND=Agg python -m unittest discover -s tests -v  # Headless test suite
python examples/gc_orbit.py              # RK4 smoke example
python examples/projected_abba.py        # Implicit ABBA smoke example
python -m build                         # Build source and wheel distributions
```

Dependency ranges and extras are declared in `pyproject.toml`; tested direct
versions are selected by `constraints.txt`. Install optional JAX support with
`python -m pip install -c constraints.txt -e '.[jax]'`.

## Coding Style & Naming Conventions

Follow existing tab indentation, `snake_case` functions/modules, and `PascalCase`
classes. Annotate function parameters and returns; strict `mypy` checks reject
untyped definitions. Write concise docstrings describing numerical contracts,
array shapes, and units where relevant. No formatter or linter is configured.
Keep package exports explicit and import internal implementations from their
defining modules. Preserve established public imports from `simulation`.

## Testing Guidelines

Use `unittest.TestCase`, files named `test_*.py`, and methods named `test_*`.
Prefer small deterministic fields, fixed seeds, and explicit numerical
tolerances. Cover changed contracts, convergence, conservation, or backend
agreement as appropriate. No coverage percentage is enforced. Run a focused
module with `python -m unittest discover -s tests -p 'test_core.py' -v`, then
the CI checks above before submitting.

## Commit & Pull Request Guidelines

History uses short imperative subjects such as `Add Poincare rho sweep studies`
and `Document executor delegation`; follow that style. Keep commits focused.
PRs should explain the numerical or API change, report validation commands and
results, link relevant issues, and update affected documentation. Include plots
or screenshots when visualization behavior changes.

## Public API and imports

Keep package exports explicit in `__init__.py` and `__all__`; do not use wildcard
imports, dynamic module aliases, or import hooks to preserve removed paths.
Each class and function has one canonical implementation. Public facades may
re-export it, while internal consumers import from its defining module.
Use `simulation.runner` when studies need the execution entry point, and import
shared types from `contracts` submodules or `solution`.

Preserve established public names unless a rename is requested. When moving
modules, migrate supported consumers and tests before deleting the old adapters;
do not leave compatibility trees for private or retired paths. Document supported
public imports and any removed routes. The notebook scope policy still applies.

## Notebook execution policy

Do not execute long-running notebooks end to end during routine validation.
Instead, run only the smallest representative subset needed to confirm that the
notebook and the modified execution paths complete without errors. Reduce costly
parameters such as integration spans, step counts, sample sizes, repetitions, and
animation frames, or execute only the relevant cells, while preserving the code
paths being checked. Apply temporary validation overrides to an in-memory or
disposable copy so that reduced results are not saved into the canonical notebook.

## Project language

Use English for all newly written or modified project content. This includes
documentation, comments, docstrings, user-facing and error messages, plot
labels and titles, and every notebook's Markdown, code text, and stored textual
outputs. Regenerate affected notebook figures and animations when they contain
non-English labels. When editing existing non-English prose, translate it into
English. Keep established identifiers stable unless a rename is explicitly
requested; proper names and mathematical notation do not require translation.

## Model architecture documentation

Keep numerical-method architecture documentation separated by model under
`docs/models/<model>/`. Model directories contain numerical-method theory and
simulation architecture only; do not create model-specific `dynamics/`
directories. Shared physical dynamics, potential behavior, and dynamics
protocols belong under `docs/dynamics/`, and model documents may link to those
shared contracts when necessary. Every model owns a `tex/` directory whose
canonical theoretical entry point is `tex/theory.tex`, with the deliberate
compiled PDF at `tex/theory.pdf`. A model with several runtime configurations
explains their common mathematics and differences in that entry point; detailed
derivations may accompany it in the same `tex/` directory. Do not recreate a
global `docs/tex/` tree.

Update the relevant model documents whenever code changes affect their public
API, dependencies, dynamics, initial configuration, simulation lifecycle,
numerical method, or result model. Do not recreate a global architecture
diagram unless that cross-model document is explicitly requested.

## Git tracking policy

Respect `.gitignore` when creating or modifying files. In particular, do not
use `git add --force` (or `git add -f`) to stage ignored files, and do not
change a file's tracked or ignored status, unless the user explicitly requests
that Git-tracking change. Only notebooks under `notebooks/experiments/` and
`notebooks/sympy/` are intended to be versioned. Development notebooks under
`notebooks/developements/` are local working files and must remain ignored.
Before staging a newly created notebook, verify its status with
`git check-ignore --no-index <path>` when its intended tracking status is not
clear.

## Commenting style

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
