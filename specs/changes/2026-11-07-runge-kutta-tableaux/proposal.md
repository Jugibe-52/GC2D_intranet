# Runge–Kutta tableaux: proposal

Rename methods.classical and its matching contracts namespace to runge_kutta, migrating supported consumers without aliases. Introduce immutable, hashable ButcherTableau(A, B, C): finite real coefficients, A shape (s,s), B/C shape (s,), s > 0, C equal to row sums and sum(B)=1 within roundoff. Validation does not prove order.

RungeKutta owns shared validation, quadrature and observation reuse. ExplicitRungeKutta, DiagonallyImplicitRungeKutta and ImplicitRungeKutta implement arbitrary-stage explicit, sequential diagonal and coupled algorithms. ExplicitEuler/RK4, SDIRK4 and GaussLegendre4 inherit their corresponding algorithms and supply fixed tableaux and metadata; retain the specialized Gauss observation. Generic constructors accept a tableau, while concrete recipes retain existing controls.

Keep component-major physical vectors (d*N), dense/particle analytic Jacobians, finite differences, independent shadow sampling and passive energy quadrature. Newton remains in solvers with existing defaults (absolute 1e-14, relative 1e-13, 20 corrections). Zero-diagonal DIRK stages are explicit. NumPy and JAX share algorithms; compiled cache identity includes coefficients. Adaptive SciPy and HBVM implementations remain separate.
