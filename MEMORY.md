# Standard Poincare viewer layout

Use two coordinated panels for time-periodic Poincare studies. On the right,
show saved particle returns and mark each selected particle's position at the
chosen **Start cycle** with an unfilled circle in its persistent particle color.
Update those circles when Start cycle, rho, or particle selection changes.
Include cycle 0 when the original initial state is saved, and distinguish the
start rings from the current return and subsequent accumulated returns.

On the left, show the effective potential that generates the actual dynamics
(including gyroaveraging and the study's Hamiltonian normalization), overlaid
with its physical guiding-center vector field. Provide an independent phase
control with 51 samples, steps 0 through 50 over one forcing period; evaluate
and verify that the first and last fields coincide. Label phase and units,
provide potential/vector visibility controls and a potential color scale, and
keep color and arrow-length scales fixed across phases. Changing rho must
update both the potential and the vector field. Transform velocities and the
Hamiltonian consistently when plotting normalized coordinates.

Generate these backgrounds from the verified original field and saved study
parameters without repeating trajectory integrations. Reuse shared study and
visualization helpers. This is the default for new or revised Poincare viewers;
document any scientifically necessary exception.

## Standard for fourth-order method comparisons

Use the following protocol as the project standard for long-time comparative
studies of fourth-order GC2D integrators. A narrower or different protocol is
acceptable only when the scientific question requires it; document the reason
and retain every applicable control and diagnostic below.

- Compare ABBA4 with one reduced projection around the complete composition,
  `BM4Implicit`, two-stage Gauss--Legendre, SDIRK4 S54b, and classical RK4.
  Treat DOP853 as the accuracy reference and use an independently tighter Radau
  integration to quantify the reference floor.
- Give every compared method identical physical data, initial states, time
  interval, effective step, saved times, and distance convention. Give all
  implicit methods identical Newton tolerances and analytic guiding-centre
  Jacobians. Keep explicit methods out of nonlinear-work statistics.
- Use the measured, nondimensionalized GC2D potential in
  `data/potential/V1/PHI_2.h5` with magnetic field `1.5`, characteristic length
  `0.06`, variable source-field selection `(15,)` with constant field 0, and
  cubic interpolation unless the
  study explicitly investigates one of those choices.
- Use three jointly integrated guiding-center trajectories initially situated
  along one radius from the periodic-cell center. The default radial distances
  are `(0.1, 0.2, 0.3)` times the cell period, with angle `0` radians toward +x.
  Keep the distances and angle explicit and editable in notebooks. This spatial
  radius is distinct from the gyro-radius `rho`; the particles subsequently
  follow the HDF5 guiding-center field without a radial constraint or mutual
  interaction. Plot and tabulate the initial positions.
- Maintain a reproducible high-precision HDF5 reference and a viewing notebook
  under `notebooks/developements/`. Use the existing DOP853/Radau reference
  pipeline, persist its solver settings and field fingerprint, and show the
  per-particle audit discrepancy. Report the measured discrepancy rather than
  treating requested tolerances as a guaranteed trajectory error. A focused
  viewing notebook may use a shorter, explicitly documented time interval.
- Use `rho = 0.3`, coupling frequency `pi/8`, and 200 normalized cycles with ten
  complete steps per cycle. Save every effective step, producing 2000 steps and
  2001 aligned states. Use minimum-image periodic distance for trajectory error.
- Use Newton absolute tolerance `1e-12`, relative tolerance `1e-11`, at most 40
  corrections, and a Jacobian relative step equal to the cube root of machine
  epsilon. Configure DOP853 with relative/absolute tolerances `1e-10`/`1e-12`
  and maximum step `0.025`; configure the Radau audit with `1e-11`/`1e-13` and
  maximum step `0.0125`.
- Time at least three complete integrations per method in alternating order and
  report the median and interquartile range. Advance all trajectories together
  in each vectorized integration, and exclude reference-generation time from
  per-method runtime comparisons.
- Verify method identities and structural claims in executable assertions. In
  particular, audit all eight Runge--Kutta order conditions through order four
  for SDIRK4 S54b, its common diagonal coefficient `1/4`, stiff accuracy, and
  its nonzero symplecticity and adjoint-symmetry defects. Assert expected stage,
  step, diagnostic-array, projection-formulation, and nonlinear-solver metadata.
- Report, for every method, space-time RMS and final periodic trajectory error,
  median runtime and quartiles, and space-time RMS and maximum absolute physical
  Hamiltonian error relative to DOP853. Interpret the Hamiltonian diagnostic as
  agreement with the reference energy history, not conservation, because the
  measured potential is time dependent.
- For implicit methods, report nonlinear solves per step, mean and maximum
  Newton corrections per step, total corrections, mean and total residual
  evaluations, and the maximum residual-to-tolerance ratio. For projection
  methods, also report and plot the mean, RMS, maximum, final value, and complete
  time history of the projection-multiplier infinity norm.
- Include the full trajectory-error history, accuracy summary, accuracy/runtime
  tradeoff, absolute and relative runtime comparison, per-step nonlinear work,
  physical-energy error history, and a downsampled trajectory animation. Retain
  all saved states; use 201 uniformly spaced animation frames at ten frames per
  second for the standard 200-cycle run.
- Derive conclusions from computed records rather than hard-coding them. At a
  minimum, identify the fastest median integration, smallest space-time RMS
  trajectory error, smallest space-time RMS physical-energy error, least total
  Newton work, and smallest peak projection-multiplier norm, and summarize the
  SDIRK4 and classical-RK4 long-time results explicitly.

## Scalar validation in potential imports

In `src/potential/load.py`, prefer named scalar validators returning the
required concrete type. Validate Boolean rescaling controls and optional sigma
before opening the file; do not use an optional validator followed by assert
for a required scale.

## HDF5 coordinates and cyclic source frequencies

The producer uses fftfreq with dt declared in seconds: source `freqs` is a
one-dimensional array of cyclic f in Hz, not angular omega. All Potential
constructors store cycles per runtime time unit; reconstruction uses
exp(i*2*pi*f*t). Random fields store f=1/(2*pi) to preserve omega=1.
Known unmarked processed archives are cyclic; explicitly angular processed
arrays divide by 2*pi once at their read boundary. Preserve their actual phase
and validate original hashes before conversion, regardless of source provenance.

`Potential.load` defaults to raw samples, axes and f. Independent flags apply
q=(X-X0)/L, u=t/T with T=1/f_ref, and P=(T/L**2)*Phi when both are enabled;
use s=1 or T=1 for disabled flags and store f*T. The reference is the first
selected mode. Preserve matrix orientation. B belongs to GC dynamics, H=P/B.
Source units and physical/legacy time conventions have separate metadata;
missing old provenance stays legacy_unspecified.

Historical studies retain their recorded clock 2*pi/f_ref. The radial
Hamiltonian Phi/(f_ref*B) equals the corrected physical-cycle Hamiltonian as a
function of cycle index, but archived seconds must not be silently relabeled.
The legacy cycle_time Hamiltonian is 2*pi times radial. Keep the radial extra
1/(2*pi) inside the historical construction with its unchanged upstream clock.
Use `studies/historical_h5.py` for recorded length, custom clock, B and sampling;
its result is already Hamiltonian. Keep RhoStarConfig coordinates separate from
loader flags and folded display. See `docs/dynamics/gc2d-h5-import.md`.

## Optional HDF5 smoothing

`Potential.load` uses `sigma=None` to disable Gaussian smoothing; a finite
non-negative width enables it in grid samples (zero leaves fields unchanged).
The separate `denoising` argument is removed. Migrate disabled calls to
`sigma=None` and enabled calls to their previous width (formerly default `1.0`).

## Direct HDF5 field selection

`Potential.load` always uses `fields[0].real` as the constant field and requires
`freqs[0] == 0` exactly. `indx` selects original HDF5 variable-field indices,
defaulting to `(15,)` via `potential.load.DEFAULT_FIELD_INDICES`; selected
frequencies must be finite and strictly positive. Keep the requested order and
pair each field with its original frequency, without amplitude sorting or
one-based index conversion. `indx=()` loads only the constant field without
temporal rescaling; `None` selects all finite positive-frequency fields in
source order. Temporal rescaling requires a selected positive-frequency mode.

For historical reconstruction of `PHI_2.h5`, the old `(0, 1)` rank selection
corresponds to original index `(15,)`; preserve the recorded study scales and
field fingerprint. Archive readers use recorded `source_field_indices` to recover
historical rank selections; never reinterpret old ranks as raw HDF5 indices.
See `docs/dynamics/gc2d-h5-import.md` for notebook and study migration.

## Potential and execution separation

`Potential` directly owns read-only physical fields and canonical SciPy splines,
and its `evaluate` uses NumPy/SciPy. `JaxPotential` in `potential.jax_potential`
(publicly re-exported from `potential`) inherits that representation and overrides
only `evaluate`, containing the JAX implementation directly. Class choice selects
the evaluation backend, including scalar/NumPy inputs. Inherited `electric_field`
calls `self.evaluate`, including omitted coordinates. `evaluate_grid` stays
NumPy-only and rejects JAX time. `load`, `random` and `gyroaverage` preserve the
class. There is no `execution=` keyword or separate prepared/evaluator class.
JAX execution preparation binds shared GC/FC equations to JAX potentials sharing
existing samples and splines without refitting or mutating the source dynamics.

Execution preparation selects devices and caches compiled runs; DOP853/Radau
retain an explicit NumPy/JAX boundary. Reuse spline coefficients without
refitting. Persistent potential device caches must contain concrete arrays only:
zero-argument JIT closures over JAX arrays can trace conversions even when the
original arguments are not tracers. See `docs/dynamics/jax-potential-evaluation.md`.

## SDD record references

The SDD schema and lifecycle are maintained in `specs/README.md`. Machine-checked
file references must already exist, even in drafts; describe planned new files
in the proposal prose until implementation creates them. Reserve new requirement
IDs through the change's `proposed_requirements`, then establish them in current
contracts before verification. Keep prior outcomes in the narrative when a
scope revision resets approval or evidence. The checker uses Python's standard
library and inspects test selectors statically without importing test modules.
Current contracts retain schema version 1. Change records use version 2 in
`specs/changes/<id>/{spec,plan,tasks}.md`; migrate their layout without rewriting
historical acceptance or results. Declared upstream revisions expose stale
plans/tasks: warnings in drafts, errors after approval. The skill must review
semantic changes; revision numbers alone do not prove alignment.

## Fixed 48-particle Poincare automation

Use `studies.poincare_automation`, `studies.poincare_automation_io` and
`visualization.poincare_automation` through `$simular-poincare` or
`scripts/run_poincare_automation.py`. The fixed reference set is the ranked
40-particle star plus eight literal probes, integrated jointly; keep IDs 1-48
and the original 40-star colors plus eight probe colors. Defaults are 5000
cycles, 50 complete steps/cycle, B=1.5 and both loader rescalings enabled.
Requested rho retains rho_m=rho*0.06/(2*pi), not a cell-width fraction.
The new physical-cycle study applies B only in GC dynamics; do not reconstruct
it through a historical clock and then undo scaling. One viewer contains all
cases and all particles, using the existing panel renderer. Prepare/run/view
are separate operations; validated complete archives are reused and failed
cases remain explicit. See `docs/studies/poincare-automation.md`.

## Current state and saved physical states

Initial values are packed NumPy arrays; `states.generators` returns arrays.
`InitialValueProblem` snapshots values and a structural layout. Physical GC/FC
layouts live in `formulations.layout` and preserve NumPy/JAX operations; named
GCState/FCState tuples are component views, not container subclasses.
`request.SimulationRequest` alone owns the initial time and output schedule.
Controllers manage current internal `IntegrationState(time, values)` snapshots.
`states.States` records physical samples and separate accepted-step statistics,
then becomes the completed read-only result returned directly by `simulate`.
Shadow samples start independently from an accepted interval's left endpoint;
they never replace the main state. JAX keeps functional device arrays inside JIT.
Initial snapshots compare by value, not provider identity. Retired State,
Solution and IntegrationCollector routes have no aliases. Archive schemas and
save_solution/load_solution stay unchanged; old archives may lack initial data.
Area remains in `diagnostics.geometry`, geometry labels in study metadata.
See `docs/simulation/api-migration.md` and `specs/current/states.md`.

## Executor-owned integration

`Execution.run` owns environment preparation, fresh runs and driver selection;
`simulate` validates the complete job/result boundary. `NumericalMethod` exposes
`new_run(problem, request, evaluated_dynamics=...)` and `jax_mode`, not integrate.
`IntegrationMethod` retains isolated preparation, numerical advancement,
observations and controller access. CompiledFixedMethod and method integration
entry points are retired; call simulate or Execution.run for complete jobs.
Execution resolves devices and supplies metadata to the driver before finalizing
States. Numerical adapters receive prepared dynamics/device and bind algorithm
controls; fixed JAX uses integrate_fixed, adaptive JAX keeps SciPy control with
prepared field callbacks. Keep executors generic and reject unsupported compiled
overrides explicitly. See `docs/simulation/execution.md` for extension/migration.
