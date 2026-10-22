# Mirror the source domains

Move the 37 existing Protocol declarations and two TypedDict declarations outside
`contracts` into matching submodules beneath it. Their associated implementation
code stays in its current domain. Include only aliases needed by their signatures.

Reorganize existing contracts into `dynamics`, `execution`, `formulations`,
`methods`, `solvers`, `states`, `studies`, `diagnostics`, and `visualization`.
Split method observations between common observations, adaptive SciPy,
classical Gauss--Legendre, extended ABBA, and extended BM4 submodules.
Split study records between comparison, BM4 recurrence, and GC energy-bound
submodules. Keep shared study validation under `contracts/studies/_validation.py`.
Keep `contracts/arrays.py` as the existing cross-domain array boundary.

For example, `NumericalMethod` moves from `methods.base` to
`contracts.methods.base`; `IntegrationMethod` remains in `methods.base`.
`StateLayout` moves from `contracts.configuration` to
`contracts.formulations.layout`. Public package facades re-export canonical
classes where they already did so. Internal consumers use defining modules.

Migrate source, tests, examples, supported notebooks, and standalone worker
bundles. Remove replaced pure-contract modules without compatibility trees.
Do not move unrelated configuration/result implementations out of their domains,
introduce new contracts, or change numerical algorithms or archive schemas.
