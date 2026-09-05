# Numerical model documentation

Every numerical model owns its documentation below `docs/models/<model>/`.
The canonical theoretical entry point is always `tex/theory.tex`, and its
deliberate compiled artifact is `tex/theory.pdf`. Runtime architecture remains
separate under `simulation/`. Physical dynamics and potential contracts are
shared across methods and therefore live only under [`docs/dynamics/`](../dynamics/).

Configuration selectors do not create extra documentation models. The theory
entry point explains the common method and the mathematical differences among
the configurations accepted by that model.

| Model | Theoretical source | Compiled theory | Configuration scope |
|---|---|---|---|
| ABBA family | [source](abba/tex/theory.tex) | [PDF](abba/tex/theory.pdf) | Five public classes and the shared configuration axes |
| `ABBA2Implicit` | [source](abba2-implicit/tex/theory.tex) | [PDF](abba2-implicit/tex/theory.pdf) | Two residual formulations, two solvers, and physical or fully extended state with optional physical energy tracking |
| `ABBA2Midpoint` | [source](abba2-midpoint/tex/theory.tex) | [PDF](abba2-midpoint/tex/theory.pdf) | Physical or fully extended state with optional physical energy tracking |
| `ABBA4Implicit` | [source](abba4-implicit/tex/theory.tex) | [PDF](abba4-implicit/tex/theory.pdf) | Two residual formulations, two solvers, and physical or fully extended state with optional physical energy tracking |
| `ABBA4ImplicitSingleProjection` | [source](abba4-implicit-single-projection/tex/theory.tex) | [PDF](abba4-implicit-single-projection/tex/theory.pdf) | Two residual formulations, two solvers, and physical or fully extended state with optional physical energy tracking |
| `ABBA6Implicit` | [source](abba6-implicit/tex/theory.tex) | [PDF](abba6-implicit/tex/theory.pdf) | Two residual formulations, two solvers, and physical or fully extended state with optional physical energy tracking |
| BM4 family | [source](bm4/tex/theory.tex) | [PDF](bm4/tex/theory.pdf) | Six projection placements/formulations and their solver choices |
| `ExplicitEuler` | [source](explicit-euler/tex/theory.tex) | [PDF](explicit-euler/tex/theory.pdf) | One explicit method |
| `GaussLegendre4` | [source](gauss-legendre4/tex/theory.tex) | [PDF](gauss-legendre4/tex/theory.pdf) | Jacobian strategy and energy tracking |
| `HBVM42` | [source](hbvm42/tex/theory.tex) | [PDF](hbvm42/tex/theory.pdf) | Jacobian strategy and energy tracking |
| `RK4` | [source](rk4/tex/theory.tex) | [PDF](rk4/tex/theory.pdf) | Physical and energy-tracking configurations |

Detailed derivations that support a model remain next to its theory:

- `abba/tex/` contains the nonlinear-solver and long-time energy notes;
- `abba2-implicit/tex/` contains simultaneous-projection and Jacobian
  derivations;
- `abba4-implicit/tex/` contains the corresponding three-solve simultaneous
  formulation, composed-Jacobian summary, and diagnostic workflow; and
- `bm4/tex/` contains the reduced and simultaneous implicit derivations.
