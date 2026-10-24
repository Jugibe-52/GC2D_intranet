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
- Use `rho = 0.3`, no BM4 harmonic coupling, and 200 normalized cycles with ten
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

## Input preparation and plotting

Keep constructor-specific conversion policies, error order and array ownership
when extracting typed helpers. Gauss/SDIRK share methods.classical._configuration;
Poincare automation deliberately has stricter numeric/count policies. Validate
every plot series before creating a figure, including later invalid entries.
See docs/simulation/integration-architecture.md and the plot-input and prepared-
constructor contract tests; runtime numerical guards stay with their operation.

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

## SDD documents

See `specs/README.md` for the readable change workflow. The checker accepts new
plain Markdown changes alongside older three-part records and current contracts;
it never executes recorded procedures. Keep historical records in their existing
format, preserving acceptance and results rather than converting them merely to
report status. Spanish reading copies are excluded from document checks. File
presence and structural validation do not establish that work was completed.

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

`dynamics.coordinates.PhysicalCoordinates` stores immutable component names and position/velocity roles. Dynamics declare `coordinates`; otherwise an explicit layout is required. Never infer GC/FC from dimension. InitialConditions, problems and States own structural data and arrays without retaining field physics; validate coordinate order and roles, not size alone.
`SingleFormulation(physical_size)` and `DoubleFormulation(physical_size)` retain only size. lift/join_copies/split_copies build temporary complete copies, including shifted projection candidates and FC velocities, preserving NumPy/JAX. No clock, momentum, tracking or model data belongs in them.
Every accepted IntegrationState(time, values, momentum=None) contains one physical Z (d*N), scalar time and optional separate physical kappa (N). Methods initialize zero kappa, prepare workspaces, project and return physical StepResult.state plus separate momentum. StepInfo carries momentum_before. Never persist equal duplicate copies between steps.
Methods evaluate physical H once; states.energy combines supplied H and kappa. GC map work uses summed momentum and divides by two on projection. Physical auxiliaries and archive labels retain their previous normalization.
Fixed methods own positive step (default 0.1); CPU/JAX clip only the last main step. Adaptive methods own max_step. Requests own interval/output times, never step controls. Shadow samples start independently from an accepted interval's left Z and kappa, and never replace the main state. JAX carries separate functional arrays.
States records physical samples and separate accepted-step statistics, then becomes the read-only result returned by simulate. Archives keep schemas and save_solution/load_solution; legacy archives can lack initial data. Recreate old configured-object pickles after this API change.
Generic packing does not extend GC-only methods to FC. GC/FC map operations belong to methods.extended.maps and methods.splitting, projection observation to diagnostics.methods.projection. Retired layout/state/wrapper classes and modules have no aliases.
See docs/simulation/initial-solvers-formulations.md, docs/simulation/api-migration.md and specs/current/states.md.

## Reusable solvers and projection equations

`solvers` owns immutable NewtonSolver/BroydenSolver configurations and shared
algebraic operations; each solve owns its work arrays. Explicit solver= controls
tolerances and iteration limits; omitted solvers retain legacy method defaults.
Gauss, SDIRK and HBVM accept Newton; ABBA/BM4 also accept Broyden. Preserve HBVM's
matrix infinity norm, damping and initial-residual-before-Jacobian validation.
`methods.projection` owns ReducedProjection (2N unknowns) and
SimultaneousProjection (6N), including guesses, residuals, Jacobian assembly and
reconstruction. Methods supply compositions/derivatives; Gauss keeps its stage
equations. ABBA accepts both classes or legacy strings; BM4 remains reduced.
Use canonical solvers.nonlinear/linear/_jax imports, not retired methods private
modules. Built-in solver/formulation types are exact to preserve CPU/JAX behavior.
See `docs/simulation/initial-solvers-formulations.md` for contracts and examples.

## Executor-owned integration

`Execution.run` owns environment preparation, fresh-run validation, running/finished transitions and one States.prepare/finalize boundary; simulate validates
the job and completed result. Host and JAX run_steps workers fill that open recorder and return numerical diagnostics without finalizing or changing run status.
Use States.record_history for complete bulk physical arrays from JAX; it requires an empty recorder and validates before writing. Preserve independent shadow
samples.

contracts.methods.base.NumericalMethod exposes new_run(problem, request, evaluated_dynamics=...) and jax_mode; methods.base.IntegrationMethod retains numerical preparation,
advancement, observations and controller access. Keep exact-type JAX adapters beside algorithms; do not repeat their fixed-method support list. Unknown
overrides fail explicitly. Configured methods are reusable; consumed runs finish on success and failure.

execution._jax owns optional environment preparation and the cached SciPy/JAX host adapter. Dynamics has no dependency on execution. JaxPotential stays
independently usable, with its own precision guards and tracing-safe concrete caches; executor preparation shares physical samples/splines without refitting or
mutation. execution._host owns host control, sampling, progress and step-count helpers; execution._jax_fixed owns the compiled loop; execution.state owns
IntegrationState.

The integration package, execution._fixed, execution._jax_run, dynamics._jax_host, CompiledFixedMethod and complete integrate_method/integrate_fixed functions
are retired without aliases. Use simulate/Execution.run for complete jobs. Import numerical bases from methods and controller/state extensions from execution.
Methods use type-only references and local controller/snapshot imports to avoid package-initialization cycles. See docs/simulation/execution.md.

## Contract responsibilities

Existing Protocol/TypedDict interfaces and shared records live under contracts in subpaths mirroring src domains.
Import defining modules, retaining public facade exports; see docs/simulation/contracts.md. Concrete implementations
stay in their domains; type-only references prevent runtime cycles. Standalone workers bundle lightweight contracts.
problem owns InitialValueProblem; execution.options owns ExecutionOptions. Shared records retain validation;
diagnostics.studies.comparison owns RMS/energy reductions, and diagnostics.studies.bm4_recurrence owns recurrence_distances.
studies._validation owns require_uncoupled_bm4; archive readers retain historical rates. contracts.arrays keeps shared
NumPy/JAX compatibility without device or execution policy.

## Uncoupled BM4 and reusable C

BM4Implicit and BM4Midpoint have no coupling_frequency constructor option and
bind uncoupled maps on CPU/JAX. Keep their signed weights, endpoint clock and
projections. HarmonicCoupling in methods.extended.maps
provides the independent spatial C flow for future compositions. Retained study
schema rates must be zero for new calculations; never relabel or resume coupled
historical trajectories through the current methods. Observer rates stay zero.

## Explicit palindromic methods

All ABBA/BM4 presets inherit PalindromicMethod; shared implicit and midpoint
bases own advancement. Composition.from_half reflects explicit operations:
A updates u from v, B updates v from u, only T moves the clock, and C applies
independent HarmonicCoupling. Each coefficient multiplies h directly, without
an extra half. Generic PalindromicImplicit/Midpoint accept recipes on CPU/JAX;
arbitrary subclasses remain outside exact-type JAX dispatch. C operations and
a coupling object must be supplied together; BM4 presets remain uncoupled.
Preserve named recipes and grouped observations. Elementary traces drive
derivatives and passive energy without replaying the map. Operation coefficients
are scalar JSON metadata: block consumers reserve diagnostic arrays for histories.
Moving T creates a different method; palindromy alone proves no higher order,
and arithmetic projection need not preserve reversal or symplecticity. See
docs/models/extended/simulation/palindromic-operations.md.

## Visible result types

In src/, annotate one receiving name inline and declare new unpacked names before
assignment. Declare each name once per scope, including earlier optional values;
keep return annotations and optional imports lazy. Use existing result types and
NumPy/JAX distinctions. See docs/development/typing.md for scope and examples.
