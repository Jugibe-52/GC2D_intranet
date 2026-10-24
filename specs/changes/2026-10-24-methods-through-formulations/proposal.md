# Proposed change

## Functions at the method boundary

`InitialValueProblem` remains the owner of the association between dynamics and
initial conditions. Execution binds the selected backend, checks physical
capabilities once and supplies `f(t, z)` to method preparation, together with
independent initial data and the request. Supply analytic derivatives or exact
flows only to algorithms that need them. Plain callable functions are sufficient;
do not require a dynamics object, universal formulation or generic service bundle.

The direct preparation API becomes conceptually
`new_run(f, initial_conditions, request, *, jacobian=None)`, with explicit
additional operations only where necessary. Derivative contracts distinguish a
dense matrix from independent-particle blocks. Methods must never inspect a
callable's bound owner to recover dynamics. Execution retains source identity
for observations and backend caches. It connects operations without implementing
physical equations, numerical stages or a second method dispatch table.

Remove method resources `problem`, `dynamics`, `evaluated_dynamics` and
`_energy_dynamics`. Keep initial snapshots, numerical controls and per-run work
independent. Apply the same boundary to host, compiled JAX and hybrid SciPy/JAX
paths. Standalone method use remains possible with ordinary functions.

## Only a doubled workspace helper

Delete `SingleFormulation` and its exports. Ordinary methods use physical arrays
and shared shape/ownership checks. Retain `DoubleFormulation(physical_size)` as
an independent helper with `lift`, `join_copies` and `split_copies`, used internally
by doubled methods. It contains no dynamics, field function, clock, momentum,
projection policy or solver. Do not introduce a common ODE formulation or a
hierarchy replacing the removed single-copy class.

Keep tableaux, signed compositions, time translations, projection placement and
residual assembly in methods. Shared GC shear/composition helpers receive `f`
and optional derivatives; FC maps receive the required exact flows or physical
operations explicitly. Keep physical FC flow equations beside dynamics and
compose their supplied operations in methods. Neither family needs a new
model-specific formulation. Preserve `GCDoubledMaps`, `FCSplitMaps` and
`HarmonicCoupling` public names while migrating their dynamics-taking preparation
and energy arguments. Spatial direct/adjoint maps and coupling keep their
current arithmetic and signed durations.

## One source of nonlinear configuration

Each implicit method has one immutable solver configuration. Move algebraic
absolute/relative tolerances, iteration limits, Jacobian strategy and difference
scale, damping/backtracking policy and iteration observation into the solver.
The solver class selects Newton or Broyden. Remove parallel method fields,
legacy precedence rules, copied controls and the redundant `SolverOptions`
representation. JAX may derive one immutable compiled view from this configuration;
it must not maintain separate defaults or validation rules.

The method supplies the residual, initial guess, analytic residual derivative
or specialized correction, scale and any equation-specific norm. The solver
owns iteration, stopping, damping and algebraic work statistics. Keep field,
composed-map and residual Jacobians distinct: changing the configuration owner
must not replace the existing differentiation target. Share centered differences,
block packing and validation without losing batched particle solves or assuming
independence for a general field.

Default solver configurations retain each family's effective settings:
Gauss/SDIRK use absolute tolerance 1e-14, relative tolerance 1e-13 and 20
iterations; ABBA/BM4/generic implicit palindromes/HBVM use 1e-13, 1e-12 and 12.
Retain current derivative defaults, Broyden initialization, HBVM damping and
its matrix infinity norm. Preserve HBVM's first residual before derivative
validation and avoid evaluating a Jacobian when convergence needs none. Explicit
solver controls are authoritative; migrate callers using their full effective
configuration, including previously implicit policies. Adaptive integration
accuracy tolerances and temporal step controls stay with the adaptive algorithm,
separate from algebraic convergence.

## Energy belongs to an observer

Execution prepares an optional energy observer with H and minus its explicit
time derivative. The observer owns zero initial physical momentum, its per-run
accumulation, energy histories and errors. Methods no longer evaluate these
functions or own energy flags, momentum arguments/results or energy diagnostics.
Numerical step records carry physical states and numerical details; observer
state remains separate and is attached to saved auxiliary output by execution.

Methods expose the already computed stages, their evaluation times and quadrature
weights when requested. Weights retain signed durations and physical normalization.
Adaptive methods expose the accepted interpolant; the observer retains the
existing eight-node quadrature. The observer consumes this information without
reimplementing tableaux or replaying accepted maps. It must not integrate energy
only on the sparse output grid. Share the existing energy observers' computation
instead of introducing a second competing momentum history.

Accumulate main-step momentum exactly once after acceptance. Interior fixed-step
samples use independent shortened maps and the left endpoint's observer state;
adaptive samples use the accepted interpolant. Neither replaces the main
accumulator. Rejected nonlinear/adaptive trials, diagnostic map callbacks and
repeated sample queries must not accumulate energy. Retain only the current
step's temporary data unless an explicitly requested diagnostic needs history.

Register step/energy observers through an optional `observers` argument to
`simulate` and executor entry points. Move `step_observer` there and
`newton_observer` to the solver's iteration observation option. Execution creates
fresh built-in observer state per run and attaches original-source identity to public
step and nested stage events. Preserve existing event classes, failure behavior
and ordering. User-supplied callbacks retain their identity and caller-owned
lifecycle; do not silently clone or reset them. Methods depend only on
observation contracts, never diagnostic
implementations; disabled observers cause no extra physical evaluations.

In JAX, energy observation is an equivalent functional array accumulator in the
compiled execution loop, separate from the physical step kernel. Do not insert
Python callbacks or transfer every stage to the host. Keep current restrictions
on arbitrary callbacks and supported method/backend combinations. Check hybrid
sampling and local remote-worker serialization; no paid remote run is needed.

## Resulting use and invariants

The intended interface is illustrated below; `EnergyObserver` denotes the new
public configuration, bound by execution to the problem's energy functions.

```python
method = GaussLegendre4(
    step=0.02,
    solver=NewtonSolver(
        absolute_tolerance=1e-13,
        relative_tolerance=1e-12,
        max_iterations=20,
    ),
)
result = simulate(problem, method, request, observers=(EnergyObserver(),))
```

Without observers, `simulate(problem, RK4(step=0.02), request)` stays valid.
Preserve accepted Z shaped `(d*N,)`, scalar time, and separately saved physical
momentum `(N,)`; temporary copies never become accepted storage. Keep GC `(x,y)`
and FC `(x,y,vx,vy)` component-major order, units, H normalization and the doubled
GC factor of one half. Energy observation never changes physical trajectories,
nonlinear decisions or adaptive error control. Numerical archive schemas and
labels remain unchanged; descriptive metadata reflects the new owners.

No code under `methods/` may import physical dynamics implementations or
`contracts.dynamics.protocols`, retain the problem or access raw dynamics.
`formulations/` remains a small workspace package independent of physics,
execution, methods and solvers. Keep optional imports lazy and numerical guards
compatible with tracing. Remove replaced helpers and retired import routes after
migrating consumers, with no compatibility trees or universal callback container.
Report eliminated duplication and total production-code counts; relocation alone
does not establish simplification. No new physics or integration method is added.
