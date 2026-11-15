# Implicit midpoint: physical stages and execution

The theoretical entry point is [theory.tex](../tex/theory.tex), with its compiled
[theory.pdf](../tex/theory.pdf). `ImplicitMidpoint` lives in `methods.runge_kutta.implicit_midpoint`
and is exported from `methods`, `methods.runge_kutta` and `simulation`.

Construct `ImplicitMidpoint(step=...)` and `NewtonSolver` controls. The recipe inherits the
shared [Runge–Kutta implementation](../../../simulation/runge-kutta.md), with
physical component-major stage vectors `(d*N,)`, no doubled formulation and
no retained dynamics object. Both NumPy and JAX float64 are supported.
Execution supplies the field, initial conditions and, for an implicit method,
particle Jacobians. The immutable request supplies the interval and saved times.
Each execution creates a fresh numerical run; the method configuration remains reusable.

`EnergyObserver` independently records H and positive analytic partial_t H at
accepted endpoints and saved times. The method does not own the campaign's
energy quadrature. Solver tolerances control algebraic residuals, not trajectory
error. `tests/test_main_campaign.py` verifies second-order convergence and
NumPy/JAX agreement; it additionally checks quadratic conservation for implicit
midpoint. The [main campaign](../../../studies/main-reflected-campaign.md)
uses the named recipe with a fixed 40-step physical cycle.
