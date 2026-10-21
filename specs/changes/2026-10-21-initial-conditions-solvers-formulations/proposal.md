# Proposed behavior

Add immutable `states.InitialConditions`, defined in `states/initial.py`, with
`gc(x=..., y=...)`, `fc(x=..., y=..., vx=..., vy=...)` and explicit-layout construction.
It owns validated packed arrays and a copied layout; access cannot mutate it.
`InitialValueProblem` accepts this object and checks compatibility with dynamics.
Existing array input and array-valued `initial_state` remain supported. Existing
geometry generators continue to return arrays that can be wrapped explicitly.
Time stays in `SimulationRequest`; `States` remains the completed physical result.

Create `solvers` for shared Newton, Broyden and linear operations, including the
optional JAX implementations. Immutable solver configurations contain tolerances
and iteration limits; each solve owns its temporary state. Add `solver=` to the
local implicit methods. Newton supports Gauss, SDIRK, HBVM and projections; Broyden
retains the existing projection support. An explicit solver supplies the stopping
controls; old constructor controls remain supported when no solver is passed.
Keep HBVM's matrix norm and backtracking, particle blocks, cached evaluations,
observer events and failure behavior. SciPy's adaptive solvers retain their own
internal nonlinear algorithms.

Add `ReducedProjection` and `SimultaneousProjection` in `formulations`, with a
shared projection contract. They own initial guesses, residuals, Jacobian assembly
and physical reconstruction. Methods supply their composition and its derivative;
formulations never import numerical methods. ABBA accepts either class and the
existing string names. BM4 retains its existing reduced projection. Gauss retains
its stage equations next to the method and uses the same general Newton solver.

For example, construct `InitialValueProblem(dynamics, InitialConditions.gc(x=x,
y=y))`, then `ABBA4Implicit(projection_formulation=ReducedProjection(),
solver=NewtonSolver(absolute_tolerance=1e-12, relative_tolerance=1e-11,
max_iterations=40))`, and call the unchanged `simulate(problem, method, request)`.

The reduced and simultaneous unknowns have dimensions 2N and 6N for N planar
particles; the physical state has dimension 2N. Preserve coordinates, units,
coefficients, physical-state tolerance scaling, independent shadow samples and
saved archive schemas. Solver tolerances are residual thresholds, not measured
trajectory errors. See [execution](../../../docs/simulation/execution.md) and
[projection theory](../../../docs/models/abba4-implicit/tex/simultaneous-formulation.tex).
