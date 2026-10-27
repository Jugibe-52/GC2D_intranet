# Affected parts

The plan record belongs to `contracts/execution`; its builder and both consumers belong to `execution`. The coordinator builds it before calling the host or JAX preparation path. Existing controller extension contracts remain supported.

Update focused scheduling, execution and compiled-contract tests, the execution guides and current simulation specification. Align relevant model guides and their Spanish reading copies. No notebooks, physical equations, solver tolerances or archives require migration. Existing uncommitted preparation changes are the baseline and must be retained.
