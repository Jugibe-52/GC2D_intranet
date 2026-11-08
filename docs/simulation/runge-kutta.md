# Runge–Kutta family

`methods.runge_kutta` replaces `methods.classical`. The abstract `RungeKutta`
base owns tableau validation, metadata, physical-step results, stage quadrature
and observation replay. The three stage algorithms are reusable with arbitrary
stage counts:

| Algorithm class | Restriction on A | Existing recipes |
| --- | --- | --- |
| `ExplicitRungeKutta` | Strictly lower triangular | `ExplicitEuler`, `RK4` |
| `DiagonallyImplicitRungeKutta` | Lower triangular | `SDIRK4` |
| `ImplicitRungeKutta` | Any validated square matrix | `GaussLegendre4` |

Concrete recipes declare a fixed `tableau`, an order and optional metadata;
they inherit all stage calculations. Gauss additionally retains its specialized
observation builder for the two accepted stage snapshots. The implicit families
share Newton configuration through an internal base; the solver owns tolerances,
Jacobian policy, correction limits and iteration callbacks.

## Coefficients and construction

`ButcherTableau(A, B, C)` owns dimensionless finite real coefficients. A has
shape `(s,s)` and B/C have shape `(s,)`, with a positive stage count. C must
match the row sums of A and B must sum to one within 64 machine epsilons scaled
by the absolute coefficient sums (with a minimum scale of one). Coefficient
sums must remain finite. Validation establishes consistency, not higher order,
stability or geometric properties. Negative coefficients and repeated or
unordered nodes are allowed.

Arrays use immutable buffers, including after copying or pickle restoration.
Tableaux compare and hash by coefficient values, so equal coefficients share
configuration identity and different ones cannot alias compiled caches.
The run metadata records `tableau_A`, `tableau_B` and `tableau_C`.

```python
from methods.runge_kutta import ButcherTableau, ExplicitRungeKutta, RK4

heun = ExplicitRungeKutta(
    tableau=ButcherTableau([[0, 0], [1, 0]], [0.5, 0.5], [0, 1]),
    step=0.05,
)
rk4 = RK4(step=0.05)
```

Existing recipes retain `step` and, when implicit, `solver`. Host progress belongs to `ExecutionOptions(progress=True)`.
Their tableau is fixed by the recipe and is not a constructor argument. Use a
generic family instance for a custom tableau. Generic methods do not infer or
report an unverified `designed_order`.

## Stage equations and execution

For physical state z, stages satisfy
`Y_i = z + h sum_j A[i,j] f(t + C[j]*h, Y_j)` and the output is
`z_next = z + h sum_i B[i] f(t + C[i]*h, Y_i)`.
Every temporary stage is a physical component-major vector `(d*N,)`.
The coefficients are dimensionless; h has the runtime time unit.

Explicit stages use only previous fields. DIRK solves one stage at a time;
a zero diagonal uses an explicit evaluation and contributes zero nonlinear work.
Different nonzero diagonal values are supported. Fully implicit stages solve
one concatenated vector of all stage states. The Newton block `(i,j)` is
`delta_ij I - h A[i,j] J_j`. Analytic Jacobians have independent particle blocks `(N,d,d)`; finite differences build a dense matrix. Default solver controls are
absolute tolerance `1e-14`, relative tolerance `1e-13` and at most 20 corrections;
these are residual controls, not bounds on trajectory error.

`EnergyObserver` evaluates H and positive analytic partial_t H only at accepted endpoints and saved samples. Specialized historical diagnostics can use stage states, times `t + h*C` and signed weights `h*B`. Replay for step diagnostics suppresses Newton callbacks. Execution
still owns interval clipping and independent shadow samples.

NumPy and JAX use the same algorithms. The three generic family classes and the
four existing recipes are explicitly supported at the JAX execution boundary;
arbitrary user subclasses retain the existing CPU-only policy. Coefficients
participate in the generic compiled configuration key. JAX still requires
float64 and rejects Python callbacks. HBVM retains its separate implementation. Reference studies call adaptive SciPy solvers directly, outside the simulation API.

## Migration and verification

Use `methods.runge_kutta` and `contracts.methods.runge_kutta`; the old namespaces
have no aliases. Existing imports of the four recipes from `methods` and
`simulation`, the specialized `GaussLegendre4IntegrationStep` facade export and
`SDIRK4_TABLEAU_A/B/C` remain available. Recreate configured-object pickles that
refer to the old modules. Completed solution archive formats are unchanged.

`tests/test_runge_kutta.py` checks coefficient ownership, family restrictions,
inherited recipes, a second-order explicit method, mixed explicit/diagonal
stages, a three-stage coupled method against an independent linear solution,
and custom NumPy/JAX agreement and cache signatures. Existing method tests cover
convergence, energy, callbacks, physical observations and full execution.
Floating-point summation order can change the last bits of existing trajectories.

## Shared preparation

RungeKutta inherits FixedStepMethod and the common IntegrationMethod contract.
ExplicitRungeKutta binds only `f` and initial conditions, with no Jacobian or
solver attribute. `_NewtonRungeKutta` also inherits ImplicitIntegrationMethod,
which supplies optional derivative binding before the shared initialization.
Both diagonal and coupled algorithms therefore retain their solver controls
and derivative strategies. Family equations and tableaux remain shared.
