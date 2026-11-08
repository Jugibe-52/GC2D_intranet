# Common and implicit bases

IntegrationMethod.new_run(f, initial_conditions) prepares a fresh integrator
without a Jacobian or algebraic solver. ImplicitIntegrationMethod derives from
it and adds new_run(f, initial_conditions, *, jacobian=None), the field derivative
and the solver contract. Concrete families keep their solver types and defaults.
Do not introduce an ExplicitIntegrationMethod class.

Keep FixedStepMethod and family bases as independent shared responsibilities.
Implicit Runge--Kutta, projected palindromic methods and HBVM combine them with
the implicit base. Explicit Runge--Kutta and arithmetic-mean palindromic methods
use the common preparation. For example, RK4().new_run(f, initial) has no
jacobian attribute; GaussLegendre4().new_run(f, initial, jacobian=df) retains df.
Passing jacobian to explicit preparation becomes an unsupported keyword.

Execution passes derivatives only to implicit configurations; JAX only reads
their solver and derivative settings. The structural NumericalMethod protocol
requires the common two-input preparation. External implicit extensions derive
from ImplicitIntegrationMethod to receive automatic derivative injection.
Update direct callers and document this extension boundary.

The implicit derivative stays optional: preserve analytic/auto/finite-difference
selection, Broyden's projection matrix and deferred validation. Supplied field
Jacobians remain independent particle blocks (N,d,d); differences remain dense.
Keep the fresh-run lifecycle and numerical formulas described in the
[execution guide](../../../docs/simulation/execution.md).
