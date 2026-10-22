# Preparation functions with explicit contracts

Use private, typed functions that take named inputs and return prepared values.
Keep helpers in the owning module unless multiple modules use the same rule.
Constructors assign the prepared values; plotting functions draw prepared series.
Avoid a universal validator, reflection-based object mutation, or a second
configuration hierarchy.

Extract initial-state ownership, complete-history validation and diagnostic
freezing in `states`; GC/FC constructor parameters; accuracy-series preparation;
the common Gauss--Legendre/SDIRK Newton controls; and meaningful sequence,
sampling and alignment checks in common comparison and Poincare configuration.
Existing short calls to scalar helpers remain explicit in their constructors.

Prepare all data before creating figures in generalized energy, full and reduced
time-extended symplecticity, HBVM42 energy, implicit iteration comparison,
implicit accuracy refinement/Newton refinement/cost, ten-method nonlinear work,
and accuracy/runtime tradeoff plots. Preserve plotted values, ordering and styles.

Keep lifecycle checks and checks of newly computed numerical values alongside
the operation they protect. Existing reader helpers, animation policies and
NumPy/JAX boundaries remain in place. Checkpoint assertions, changed input
policies and unrelated scientific corrections are outside this refactor.
