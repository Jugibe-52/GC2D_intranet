# Affected parts

The method, contract, and diagnostic packages move together. Update their
explicit exports and all active consumers in execution, studies,
visualization, examples, scripts, and tests. In particular, JAX dispatch must
continue recognizing the same method classes after relocation. The reference
implementations used by tests retain their formulas; only affected imports
change. Existing uncommitted Runge–Kutta changes must be preserved.

Update the current simulation specification, API and execution guides,
diagnostics package guide, and affected ABBA/BM4 model documents. Move the
shared model documentation to `docs/models/palindromic/`, rename architecture
assets carrying the family name, and update their generator references. Keep
the model's `tex/theory.tex` and deliberate `tex/theory.pdf` together. Historical
change records and archived notebook copies remain historical evidence.

Four active local notebooks contain affected imports or documentation links:
`notebooks/developements/bm4_simple_inspection.ipynb`, the calculation notebook
under `bm4_h5_simple_inspection/`, and the calculation/visualization pair under
`energy/BM4_RK4/`. Update those references while retaining their ignored status.
Validate only relevant cells with reduced parameters in disposable copies;
do not run complete studies or replace saved scientific results.

Public imports from `simulation` remain supported. Callers using the retired
family paths must migrate. Existing pickles that encode those old Python
module paths may require recreating their configured methods; no compatibility
loader is planned. Completed trajectory arrays and archive schemas remain
unchanged. Check new serialization and worker preparation locally without
submitting remote jobs.

Existing tests cover recipes, projection, energy accounting, reversibility,
convergence, diagnostic records, and CPU/JAX agreement. Add focused layout
checks for individual defining modules and the absence of retired paths.
Compare representative trajectories and observations against a baseline
captured before moving code; documentation validation alone cannot establish
numerical equivalence.
