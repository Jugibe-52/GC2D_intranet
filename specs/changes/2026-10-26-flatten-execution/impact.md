# Impact

Implementation changes are limited to `execution/execution.py`, `execution/_jax.py`
and `execution/_jax_step.py`. The fixed integrator keeps using `kernel.device`;
the hybrid adapter retains its explicit NumPy/JAX boundary. No numerical formulas,
precision policy, cache identity, saved-data formats or notebook inputs change.

Update the execution guide and current simulation specification, with Spanish
reading copies. Reuse lifecycle, backend, observer, cache and import-boundary
tests, and cover direct step preparation restrictions. Check source types and
the repository's CI commands. No live cloud job or long notebook is needed.
