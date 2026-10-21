# Explicit initial conditions and nonlinear components

Make the preparation of simulations readable by separating physical initial data,
integration methods, projection equations and algebraic solvers. Today initial
arrays and layouts are separate, while reusable Newton operations and projection
formulations are embedded inside methods. The discussed design gives each a
clear owner without changing the physical or numerical problem.
