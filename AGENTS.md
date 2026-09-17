# Notebook scope

When inspecting or modifying notebooks, work only in `notebooks/developements/`
by default. Do not read or alter `notebooks/experiments/` unless the user
explicitly asks for it.

Experiment notebooks are versioned scientific artifacts and are outside the
supported interactive API.

# Token efficiency

Use the minimum context and tool work needed for a correct result. Start with
targeted file searches, inspect only relevant sections, and avoid repository-wide
reviews unless requested. Run the narrowest meaningful checks and do not repeat
successful checks unless subsequent changes can affect them. Keep progress updates
brief and final responses concise unless the user asks for detail.

# Notebook execution policy

Do not execute long-running notebooks end to end during routine validation.
Instead, run only the smallest representative subset needed to confirm that the
notebook and the modified execution paths complete without errors. Reduce costly
parameters such as integration spans, step counts, sample sizes, repetitions, and
animation frames, or execute only the relevant cells, while preserving the code
paths being checked. Apply temporary validation overrides to an in-memory or
disposable copy so that reduced results are not saved into the canonical notebook.

# Notebook study policy

Keep notebooks focused on the scientific definition and interpretation of an
experiment. Parameters that affect reproducibility must remain explicit in the
notebook, including potential parameters and seeds, initial-condition geometry,
physical and numerical parameters, integration spans and steps, and sampling
choices.

Put reusable experiment composition in `src/studies/`. This includes
common potential and initial-condition construction, system assembly, parameter
validation, repeated-run orchestration, diagnostic extraction, and summaries.
Put reusable plotting and notebook display helpers in
`src/visualization/`. Put opt-in numerical observers and persistence in
`src/diagnostics/`. Do not duplicate project-root discovery, `sys.path`
mutation, observer lifecycle management, result-dictionary assembly, or
presentation helpers across notebooks.

Keep generic geometry, dynamics, potentials, numerical methods, and result
behavior in `src/initial_conditions/`, `src/dynamics/`, `src/potential/`, and
`src/simulation/`; studies should compose those APIs rather than reimplement
them. A notebook-local helper is appropriate only when its behavior is unique
to that study and would not provide stable reusable composition.

## Standard saved radial trajectory reference

Use `data/trajectory/h5_three_radial_dop853_t35` as the default saved DOP853
reference for the standard three-particle HDF5 radial case. Load it with
`diagnostics.load_reference_trajectory`; `.times` and `.states` contain the
reference, while `.audit_states` and `.audit_distances` contain the Radau audit.
Reuse these saved data instead of repeating reference integrations.

This is an exact prefix of `outputs/developements/accuracy/h5_three_radial_2x_precision/v1`,
covering **normalized time [0, 35], not 35 oscillation cycles**, with 3501 samples
at spacing 0.01. Arithmetic is float64. DOP853 uses rtol=5e-13, atol=5e-15 and
maximum step=0.0025; Radau uses 5e-14, 5e-16 and 0.00125, respectively.
The measured maximum periodic discrepancy is approximately 5.5044721e-6;
per-particle maxima are approximately (5.5044721e-6, 2.8218455e-7, 5.8219384e-8).
These audit discrepancies are not rigorous error bounds. Standard status does
not establish accuracy below that measured scale. Check matching physical
settings, initial states and field fingerprint before using the reference.

Keep its NPZ, JSON and README together. Reproduce or verify the copy using
`notebooks/developements/accuracy/h5_three_radial_reference_2x_precision/export_standard_reference_t35.ipynb`.
Do not extrapolate it beyond time 35. Longer studies, including the protocol
below, require a separately audited reference covering their full interval.

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

# Project language

Use English for all newly written or modified project content. This includes
documentation, comments, docstrings, user-facing and error messages, plot
labels and titles, and every notebook's Markdown, code text, and stored textual
outputs. Regenerate affected notebook figures and animations when they contain
non-English labels. When editing existing non-English prose, translate it into
English. Keep established identifiers stable unless a rename is explicitly
requested; proper names and mathematical notation do not require translation.

# Model architecture documentation

Keep numerical-method architecture documentation separated by model under
`docs/models/<model>/`. Model directories contain numerical-method theory and
simulation architecture only; do not create model-specific `dynamics/`
directories. Shared physical dynamics, potential behavior, and dynamics
protocols belong under `docs/dynamics/`, and model documents may link to those
shared contracts when necessary. Every model owns a `tex/` directory whose
canonical theoretical entry point is `tex/theory.tex`, with the deliberate
compiled PDF at `tex/theory.pdf`. A model with several runtime configurations
explains their common mathematics and differences in that entry point; detailed
derivations may accompany it in the same `tex/` directory. Do not recreate a
global `docs/tex/` tree.

Update the relevant model documents whenever code changes affect their public
API, dependencies, dynamics, initial configuration, simulation lifecycle,
numerical method, or result model. Do not recreate a global architecture
diagram unless that cross-model document is explicitly requested.

# Git tracking policy

Respect `.gitignore` when creating or modifying files. In particular, do not
use `git add --force` (or `git add -f`) to stage ignored files, and do not
change a file's tracked or ignored status, unless the user explicitly requests
that Git-tracking change. Only notebooks under `notebooks/experiments/` and
`notebooks/sympy/` are intended to be versioned. Development notebooks under
`notebooks/developements/` are local working files and must remain ignored.
Before staging a newly created notebook, verify its status with
`git check-ignore --no-index <path>` when its intended tracking status is not
clear.

# Commenting style

Use a medium level of comments throughout the project:

- Add concise docstrings to modules, classes, and non-trivial methods.
- Comment mathematical steps, numerical algorithms, invariants, and decisions
  whose purpose is not immediately clear from the code.
- For important variables, state their physical or numerical meaning, expected
  shape and coordinate/block convention, plus units or normalization when that
  information is known. Keep this explanation near the declaration or in the
  containing docstring.
- Explain why a non-obvious operation is necessary rather than restating its
  syntax.
- Avoid line-by-line narration and comments on trivial assignments, accessors,
  or otherwise self-explanatory code.
- Keep comments accurate when behavior changes, and remove comments that no
  longer describe the implementation.
