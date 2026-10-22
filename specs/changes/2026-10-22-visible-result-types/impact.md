# Affected parts

All `src/` packages contain calls whose results can benefit from visible types:
physical inputs, dynamics, formulations, solvers, numerical methods, execution,
diagnostics, studies and visualization. Existing schemas and supported imports
remain intact. Reused variables and optional initialization need scope-aware
annotations rather than repeated declarations.

The concrete device contract affects `execution._jax`, the execution coordinator,
compiled-step contracts and the JAX method adapters. Type-only imports must preserve
optional dependency loading and avoid package initialization cycles.

Update development documentation and memory, with complete local Spanish copies.
Document the preparation return contract in the execution guide and current
simulation specification. Existing package-layout, execution and JAX tests cover
the affected import and execution paths. Run source type checking, the test suite,
smoke examples and distribution builds; compare executable syntax before and after
the annotation sweep. No notebook execution or saved-data regeneration is needed.
