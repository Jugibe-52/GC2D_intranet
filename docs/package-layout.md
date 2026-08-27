# Source package layout

The Python implementation is stored directly below `src/` and divided by
responsibility. There is no umbrella import package: callers import each public
name from the package that owns it.

| Package | Responsibility | Representative public names |
| --- | --- | --- |
| `potential` | Periodic electrostatic field model | `Potential` |
| `dynamics` | Equations of motion and capability protocols | `GuidingCenterDynamics`, `FullCyclotronDynamics` |
| `initial_conditions` | Initial state layouts and geometry | `GCInitialConfiguration`, `FCInitialConfiguration`, `Area` |
| `simulation` | Problems, methods, formulations, nonlinear-solver selection, requests, and solutions | `InitialValueProblem`, `ExplicitEuler`, `RK4`, `ImplicitABBA1`, `ABBA4Implicit1`, `ABBA4SingleProjectionImplicit1`, `ABBA6`, `ABBA_implicit2`, `ABBA4_implicit2`, `BM4_implicit2`, tangent-Taylor methods, `NonlinearSolver`, `SimulationRequest`, `Solution` |
| `diagnostics` | Optional observers and diagnostic persistence | `StoredReferenceTrajectory`, generalized-energy observers, `GCFullyExtendedSymplecticityObserver`, time-extended and trajectory symplecticity observers, iteration observers, and diagnostic writers |
| `studies` | Reusable experiment assembly and summaries | `HighPrecisionReferenceConfig`, `ImplicitGeneralizedEnergyConfig`, `FullyExtendedImplicitConfig`, `TangentTaylorEulerAccuracyConfig`, `ABBA4ProjectionComparisonConfig`, trajectory comparison results, symplecticity configurations, and `run_*_study` functions |
| `visualization` | Optional plots, animations, tables, and notebook display | Generalized-energy component/error/refinement plots, `plot_fully_extended_symplecticity`, ABBA4 projection-comparison plots, tangent-Taylor and reference-trajectory plots, runtime/accuracy plots, animations, symplecticity plots, and `plot_potential` |

## Import pattern

```python
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import InitialValueProblem, RK4, SimulationRequest, simulate

potential = Potential.random(A=0.7, M=25, nx=64, ny=64, seed=27)
configuration = GCInitialConfiguration.from_components(x=[1.0], y=[2.0])
problem = InitialValueProblem(
    GuidingCenterDynamics(potential, rho=0.3),
    configuration,
)
solution = simulate(
    problem,
    RK4(),
    SimulationRequest.uniform(
        t_span=(0.0, 1.0),
        max_step=0.01,
        sample_count=101,
    ),
)
```

Imports from implementation submodules are reserved for functionality that is
intentionally specialized. For example, symbolic or diagnostic code may use a
specific method module when it needs an internal map that is not part of the
general simulation API.

`ImplicitABBA1` and `ImplicitABBA2` expose the two nonlinear formulations of
Hairer's symmetric ABBA projection. The first solves the reduced multiplier
equation with independent `2 x 2` Newton blocks; the second implements the
equivalent simultaneous equation (21) with independent `6 x 6` blocks.

`ABBA4Implicit1` is the public fourth-order Yoshida composition of three
complete reduced `ImplicitABBA1` roots. One outer step uses the signed
durations `(gamma h, delta h, gamma h)`, with
`gamma = 1 / (2 - 2**(1/3))` and `delta = -2**(1/3) * gamma`; the middle
substep therefore runs backward. Each substep performs a separate nonlinear
solve and owns a separate projection multiplier. Its complete-step observation
contains the three accepted `ImplicitABBAIntegrationStep` values, and exact
diagnostics compose their physical tangents in flow order as `J3 @ J2 @ J1`.

`ABBA4SingleProjectionImplicit1` uses the same signed Yoshida coefficients but
keeps the two copies independent through all three unprojected ABBA maps. It
solves one reduced Hairer projection around the complete base composition, with
no projection between signed maps. If `M1`, `M2`, and `M3` are their accepted
doubled-map tangents in flow order, the exact outer residual Jacobian is
`K = G (M3 @ M2 @ M1) N + 2 I`. Consequently, one complete step owns one
nonlinear solve and one multiplier, whose leading scaling is `O(h**5)` for the
fourth-order symmetric composition. `ABBA4SingleProjectionIntegrationStep`
stores that root and its aggregate solver metrics, while its three
`UnprojectedABBAIntegrationStep` entries expose the continuous stage history
without assigning a multiplier or nonlinear solve to an internal map.
The implementation advances the exact signed local times and the physical
`(u, v)` part of the autonomous kernel. Since the conjugate momentum does not
feed back into those variables, this is the same physical map, but the method
does not materialize `k` or expose a separate `6 x 6` extended tangent.

`ABBA6` is the public sixth-order Yoshida composition of seven complete
reduced `ImplicitABBA1` roots. Its real coefficients are palindromic, sum to
one, and cancel the degree-three and degree-five modified-flow terms. Each
signed substep has an independent nonlinear solve and projection multiplier;
complete-step diagnostics aggregate all seven solves.

`ImplicitABBA1TangentTaylor` and `ABBA4Implicit1TangentTaylor` use those exact
base-map tangents inside `z + h f + h**2 / 2 D(Psi_base) f`. They recompute the
base nonlinear root and tangent at every state of their own trajectory; they
do not sample a precomputed original trajectory. The paired comparison runners
in `studies` report aligned periodic minimum-image drift from each base method.

`BM4Implicit1` and `BM4Implicit2` apply the same two Hairer projection
formulations around one complete twelve-stage `BM4Composition` cycle. Their
Newton matrices use centered differences of the doubled BM4 map; projection is
performed once per complete cycle and is distinct from `ProjectedBM4Composition`,
which averages and re-embeds the copies after every internal stage.

`run_projected_bm4_symplecticity_study` analyzes that stage-projected method.
Its observer differentiates each projected internal stage, composes the twelve
stage tangents into one local physical-step Jacobian, and separately propagates
the accumulated physical-flow Jacobian. The result reports symplecticity and
determinant defects, transported-area error, copy separation, and refinement
slopes without treating area preservation alone as a symplecticity proof.

`MidpointBM4` is the uncoupled explicit midpoint-projection variant. It starts
each complete step on the doubled diagonal, executes all twelve BM4 stages,
and averages the copies exactly once after that cycle. Its dedicated observer
uses exact guiding-center field Jacobians to compose independent `4 x 4` stage
factors into local and accumulated `2 x 2` physical tangents. The reusable
study reports the arithmetic mean of the individual trajectory defects; it
does not substitute the packed-system Frobenius defect, which is an RMS.

`GCTrajectorySymplecticityObserver` provides the same independent-trajectory
measurement protocol for `MidpointABBA`, `ImplicitABBA1`, `ABBA4Implicit1`, and
`BM4Implicit1`. It differentiates the explicit ABBA shears directly, applies the
implicit-function theorem to the accepted implicit projection, and, for BM4,
first composes the twelve exact coupled extended-stage factors. Every local
physical `2 x 2` tangent is then accumulated in time before the observer takes
the arithmetic mean of the per-trajectory relative defects. Its reusable
study runners share one initial configuration and saved-time grid across all
requested step sizes; trajectory plots use sampled markers without connecting
lines.

All public implicit ABBA/BM4 methods select either `newton` or `broyden`
through the same `nonlinear_solver` field. The Broyden path is shared across
method families and depends only on each formulation's residual evaluator.
Reduced formulations start from `4 I`; simultaneous formulations use the
corresponding zero-step
output--multiplier Jacobian. Per-step diagnostics distinguish nonlinear
corrections from explicit residual evaluations.

`ImplicitABBAJacobianObserver` evaluates the local physical tangent
`D Psi_{h,t_n}(z_n)` from converged ABBA stage data. Its spectral analysis is
separate from the area and symplecticity observers: it records the matrix,
trace, determinant, discriminant, eigensystem, and SVD for each independent
particle without accumulating the tangent flow.

`ImplicitABBAIterationObserver` consumes solver metrics emitted by each
accepted implicit ABBA step. It records how many nonlinear corrections were
required and the final residual relative to the effective stopping tolerance;
it does not reevaluate the step or its nonlinear equations.

`ImplicitBM4IterationObserver` records the equivalent metrics around each
complete projected BM4 cycle. Its records share the generic
`ImplicitIterationRecord` schema, allowing the same plots and persistence
format to present either method family.

`run_implicit_trajectory_comparison` advances the same initial configuration
with the four implicit methods, one common step, and one nonlinear solver. Its
result provides six pairwise periodic trajectory summaries and four aligned
iteration summaries. `animate_implicit_method_trajectories` assigns one color
to each method while displaying every particle trajectory.
`plot_implicit_trajectory_differences` presents the four methods on both axes
of a symmetric matrix and annotates every cell with the mean periodic particle
distance over all trajectories and saved states.

`run_ten_method_trajectory_comparison` extends the aligned comparison to
`MidpointABBA`, `MidpointBM4`, and Newton/Broyden instances of both implicit
ABBA and BM4 formulations. It exposes 45 pairwise periodic-distance summaries,
ten wall-clock runtimes, and eight nonlinear-work summaries. The matching
visualization uses a grouped `10 x 10` logarithmic distance matrix, a runtime
chart, aligned implicit-work plots, and an animation of sampled trajectory
points without connected paths.

`run_high_precision_reference_trajectory` integrates the same interpolated GC
ODE with DOP853 and an independent Radau audit on a prescribed output grid. It
persists a stable `example_trajectory/vN` artifact as checksummed NPZ arrays, a
JSON manifest, and a standalone explanation. The manifest records both solver
controls, exact initial-condition reconstruction data, the periodic audit
floor, and a fingerprint formed from the gyroaveraged sampled field and
canonical off-grid electric-field probes. `run_ten_method_accuracy_study`
validates the exact initial state, time grid, physical parameter, grid,
metadata, and dynamics fingerprint before measuring minimum-image errors for
the ten variants. The refinement runner evaluates nested steps on one saved
cadence that is an integer multiple of every complete step, then reports error
gains and observed time-integrated/final RMS orders. Its visualization shows
error evolution, log-log accuracy against step, global/final RMS accuracy, and
the runtime trade-off; reference paths use markers without connected lines.

`run_abba4_implicit_1_accuracy_study` provides a focused refinement of the
fourth-order composition against the same versioned reference, including
periodic RMS errors, observed orders, nonlinear work, runtime, and the margin
above the reference audit floor.
`run_abba4_projection_comparison_study` advances `ABBA4Implicit1` and
`ABBA4SingleProjectionImplicit1` on the same nested steps, saved-time grid,
initial state, tolerances, and verified reference. Its summaries compare
periodic accuracy, Newton iterations, raw residual evaluations, normalized
ABBA-map residual and tangent evaluations, residuals relative to tolerance,
multiplier scaling, and median wall time with an interquartile interval. Timing
repetitions alternate which method runs first to limit thermal and cache-order
bias. Its convergence rows report
the explicit deficit from designed fourth order and only diagnose order
reduction when the errors remain resolved above the reference floor. The
matching visualization functions plot
accuracy, observed-order reduction, nonlinear work, multiplier scaling, and
runtime from those same result rows.
`run_abba6_accuracy_study` provides the matching seven-stage refinement and
uses an expected sixth-order slope in the shared accuracy visualization.
`run_abba4_implicit_1_trajectory_symplecticity_study` uses the exact ideal-root
product `J3 @ J2 @ J1` to report local and accumulated per-trajectory defects
for the same initial configuration across step sizes.

`GCGeneralizedEnergyObserver` reconstructs `kappa = k / 2` from the accepted
stage snapshots of `ImplicitABBA1`, `ABBA4Implicit1`, and `BM4Implicit1`.
For ABBA it evaluates the four endpoint-time shears, for ABBA4 it sums the
three signed substeps, and for BM4 it follows the twelve direct/adjoint stages
including the exact physical coupling. `run_implicit_generalized_energy_study`
then reports `K = h + kappa`, signed drift, running error envelopes, refinement
orders, and solver summaries at every genuine main-grid node.
`GCTimeExtendedSymplecticityObserver` complements those energy records with
the centered-difference Jacobian of the accepted splitting on
`(u_x, u_y, v_x, v_y, t, k)`. It reports the relative `6 x 6` form defect and
the determinant error. ABBA and BM4 measure their complete accepted base
cycles; ABBA4 aggregates its three signed ABBA base maps. The diagonal Hairer
projection is excluded because it is not a diffeomorphism from `R^6` to
`R^6`.
`GCReducedTimeExtendedSymplecticityObserver` separately differentiates the
complete projected method on `(x, y, t, kappa)`. Each perturbation re-solves
the implicit step, so the resulting `4 x 4` Jacobian includes the nonlinear
multiplier, all internal projections, the time dependence, and the discrete
momentum increment. This separates physical area preservation from full
extended symplecticity.

`ABBA_implicit2`, `ABBA4_implicit2`, and `BM4_implicit2` instead carry two
complete autonomous copies
`(x_1, y_1, t_1, k_1, x_2, y_2, t_2, k_2)`. Every Hamiltonian shear is an
`R^8` map and the implicit projection has a four-component multiplier that
enforces the full diagonal, including both `t` and `k`. Their accepted `R^4`
state contains the directly integrated conjugate momentum, so
`GCFullyExtendedEnergyObserver` reads `K = h(t,z) + k` without reconstruction.
`GCFullyExtendedSymplecticityObserver` differentiates both the accepted base
maps against the cross-coupled `Omega_8` and the complete projected map against
`Omega_4`. The base tangent is the analytic product of the triangular shear
Jacobians (and exact BM4 binding matrices), Newton uses
`D_lambda R = G D Psi A + 2 I`, and the projected tangent follows from the
implicit-function theorem. Independent centered differences audit all three
analytic matrices without entering the nonlinear solve.
`run_fully_extended_implicit_study` combines those histories with
step-refinement orders and nonlinear-work summaries.

## Dependency direction

The core dependency flow is intentionally one-way:

```text
potential <- dynamics
initial_conditions <- simulation -> dynamics
              diagnostics -> simulation
visualization -> potential + initial_conditions + simulation + diagnostics
studies -> potential + dynamics + initial_conditions + simulation + diagnostics
```

The core simulation packages do not import notebook or Matplotlib helpers.
This keeps a minimal installation usable with NumPy and SciPy alone.

## Type information and packaging

Each top-level package carries a `py.typed` marker. Package discovery, static
checking, tests, examples, and distribution builds all operate on the same
flat `src/` layout:

```bash
python -m mypy src
python -m unittest discover -s tests -v
python -m build
```
