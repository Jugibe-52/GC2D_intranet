# Affected parts

Change methods/base.py, add methods/implicit.py and its public export, and adapt
the Runge--Kutta, palindromic and HBVM inheritance and preparation. Update the
NumericalMethod protocol, host preparation, JAX signatures and historical
generalized-energy replay. Keep stage equations and solver implementations intact.

Migrate tests and direct new_run callers that currently pass an unused derivative
to explicit methods. Add coverage for explicit providers with only two arguments,
implicit bindings, independent repeated runs and NumPy/JAX behavior. Reuse the
existing convergence, projection, Broyden, observation and serialization checks.

Update current simulation contracts and affected execution, API and model guides,
with Spanish reading copies. Inspect notebooks for direct preparation calls;
ordinary simulate calls retain their signature and need no migration. Completed
result archives retain their schema; recreate old serialized prepared runs.
No campaigns, remote deployment or generated scientific results are requested.
