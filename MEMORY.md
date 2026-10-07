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
  `0.06`, source-field selection `(0, 1)`, and cubic interpolation unless the
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

In `src/potential/load.py`, prefer named scalar validators to keep
`_validated_import_controls` consistent in abstraction level, even with a single
caller. Required positive scales should use a validator returning `float`;
avoid an optional validator followed by `assert`.

## HDF5 coordinate convention

`Potential.load` uses only `2*pi*(X-X0)/characteristic_length`; the loader's
spatial-mode argument and `SpatialNormalization` type are removed. Keep the
length configurable: a 0.18 m cell at length 0.06 m has period `6*pi`.
`RhoStarConfig.spatial_normalization` is a separate study setting: dimensional
studies restore meters after loading, and folded unit-cell viewers transform
results afterward. See `docs/dynamics/gc2d-h5-import.md` for archive migration.

## Optional HDF5 smoothing

`Potential.load` uses `sigma=None` to disable Gaussian smoothing; a finite
non-negative width enables it in grid samples (zero leaves fields unchanged).
The separate `denoising` argument is removed. Migrate disabled calls to
`sigma=None` and enabled calls to their previous width (formerly default `1.0`).
