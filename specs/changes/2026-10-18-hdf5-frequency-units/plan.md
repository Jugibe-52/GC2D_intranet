+++
schema_version = 2
kind = "change-plan"
id = "2026-10-18-hdf5-frequency-units"
revision = 1
spec_revision = 2
sources = ["src/potential/load.py", "src/potential/potential.py", "src/potential/_evaluation.py", "src/potential/jax_potential.py", "src/dynamics/gc.py", "src/studies/historical_h5.py", "src/studies/dimensional_h5_midpoint.py", "src/studies/poincare_rho_sweep.py", "src/studies/h5_provenance.py", "src/studies/_potential_snapshot.py", "src/studies/poincare_probe.py", "src/diagnostics/io/persistence.py", "tests/test_h5_rescaling.py", "tests/test_gc2d_h5_potential.py", "tests/test_potential_snapshot.py", "tests/test_solution_persistence.py"]
documents = ["specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/spec.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/plan.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/tasks.md"]
open_questions = []
+++

# HDF5 frequency units: implementation plan

## Design

### Input and evaluation

The user selected public cyclic frequencies on 2026-10-18. Read `freqs` as f,
retain raw source provenance, and store f or f*T in `Potential.frequencies`.
`Potential.load` keeps its current keyword interface. Compute physical T=1/f_ref
only when temporal rescaling is requested; s=L only with spatial rescaling.
Apply T/s**2 directly to the mean and mode arrays. Preserve source indices,
first-selected reference, orientation, filtering and validation policies.

Use one clear evaluation convention across NumPy, JAX, direct construction and
subclasses: form omega=2*pi*frequencies for exp(i*omega*t), then use powers of
i*omega for time and mixed derivatives. Do not repeatedly normalize and undo
normalizations. Inspect every frequency use, including cached arrays and JAX
tracing, rather than changing only the loader. Keep public names stable.

The currently angular random-field default with omega=1 must become f=1/(2*pi)
to preserve its phase. Migrate other constructed fields according to their actual
intended phase, with explicit units near definitions. Keep B exclusively in the
GC dynamics contract; map rho, states and physical times with the same s and T.

### Historical studies and viewer

Capture baselines from the current working tree before implementation, including
uncommitted work from the previous change. HEAD is not the reproduction baseline.
Record radial and cycle_time sample fields, derivatives, trajectories, stored
time arrays, source axes, selected modes, metadata and existing fingerprints.

Keep legacy equations in the existing historical study boundary, with explicit
legacy time provenance. Use raw `source_frequencies` for its reference f rather
than guessing units from runtime arrays. Preserve the legacy clock 2*pi/f_ref,
cycle_time amplitude 2*pi*Phi/(f_ref*B), radial amplitude Phi/(f_ref*B), custom
archived clocks and characteristic-coordinate factors. Store the resulting
runtime cyclic rates directly; evaluate with the common kernel. Avoid copying
the physical rescaling path and then undoing its transformations.

Inspect `historical_h5.py`, `dimensional_h5_midpoint.py`, `poincare_rho_sweep.py`
and every caller. Preserve public helper names where required (including any
legacy angular-reference name), but document their actual legacy meaning. A
physical import uses the corrected loader; a historical study uses its explicit
construction. Updating the dimensional helper's clock globally would break the
viewer and is prohibited by the preservation criterion.

Preserve original-axis spline construction and sampling order for historical
fingerprints. Compare the existing representative native-grid hash
`ba7fd7e8b846e90d7f1668c123a6052666e935d0076439f4711ea6247b8ef8e3`
and 32x32 hash
`a851e116bb6ebe3cf121bd47f823b46e1bc0bff2dceb555f4366e308dd98e273`.
Confirm their fixture/procedure against the previous change record before use.
Do not silently replace an expected hash when an arithmetic change differs.

### Processed archives and provenance

Separate three concepts: original source frequency units, stored runtime
frequency convention, and physical or legacy time scaling. Existing explicitly
angular processed arrays are converted once by dividing by 2*pi on read into
the cyclic runtime. Known unmarked legacy cyclic arrays remain unchanged.
Never reinterpret a processed array as raw source Hz. Reject unknown conventions;
do not infer them from frequency magnitudes or filenames.

Apply that policy consistently to pickle state, potential snapshots, solution
persistence, study payloads and probe/viewer readers. Verify an original stored
fingerprint before converting its representation. Preserve saved trajectories,
recorded clocks and original provenance; use an explicit legacy/unspecified value
when missing metadata cannot be recovered. New writes declare cyclic runtime
rates and their time convention. General angular-to-cyclic round trips may need
numerical comparison because float division/multiplication is not bitwise exact;
that does not authorize weakening original hash checks or changing viewer data.

### Consumer migration and documentation

Audit source, tests, examples, scripts, generated-code strings, visualization
readers and tracked/ignored notebooks. Reconcile the previous migration inventory
with a new search; old inventory counts alone are not proof of full coverage.
Include `scripts/create_gc_energy_bound_notebook.py` and
`scripts/run_bm4_projection_comparison.py`. Distinguish supported live consumers
from frozen exports/backups, and record each exclusion with its reason.

During accepted implementation, update the POT contract, HDF5 import guide,
affected model/JAX documentation and confirmed MEMORY entries that currently
call the source angular. Preserve the prior verified record as history and
explain its superseded assumption by a link to this corrective record. Keep
canonical content English and maintain full local Spanish Markdown copies.

## Compatibility

The loader signature, B ownership, spatial conventions and selection policies
stay stable. Public frequencies change back to cycles per runtime time unit;
all supported direct constructors and consumers must migrate together. A new
physical raw import intentionally corrects its temporal phase and normalization.
Historical readers preserve the actual saved equation, not the old claim that
its source interpretation was physically correct. Do not rewrite archived
seconds, publish a viewer, change Git tracking or run production simulations.

## Verification plan

- AC-001/AC-002: build a small real time signal with a known mean, nontrivial
  complex phases and two positive non-Nyquist bins. Obtain coefficients with
  fft/N and frequencies with fftfreq at a non-unit dt in seconds; write a temporary
  HDF5 in the test. Compare reconstruction at original sample times against the
  original signal/independent inverse FFT, and compare time/mixed derivatives
  against analytic derivatives. Demonstrate that the present exp(i*f*t) kernel
  fails the physical-period check before applying the fix. Use asymmetric spatial
  data and nonzero origin; separate spatial spline error from temporal error.
- AC-003/AC-007: exercise all four rescaling flag combinations, multiple selections
  with a nontrivial first reference, mean-only policy, B != 1 and nonzero rho.
  Transform initial states and sample times consistently. Compare dimensional
  GC trajectories with independently integrated DOP853/Radau references and
  reduced-step convergence, rather than only comparing two rescaled copies.
- AC-004: compare legacy radial and cycle_time, custom periods, metadata and
  exact representative fingerprints against the captured working-tree baseline.
  Exercise the local folded radial viewer with a reduced, disposable notebook
  run, preserving saved results and old time arrays.
- AC-005: test unmarked cyclic archives, explicitly angular archives from the
  preceding implementation and new cyclic archives; include unknown tags,
  round trips and checks against double conversion. Verify saved provenance and
  hashes in the original representation before runtime adaptation.
- AC-006: test direct/random construction, subclass contracts and available
  NumPy/JAX backends, then audit every affected live consumer and generator.
  Run minimal notebook setup/import/reconstruction plus a short integration;
  reduce particles, steps, spans and frames in memory or a disposable copy.

Use the tolerances in spec.md: exact source samples, FFT phase rtol=atol=1e-12,
derivatives/maps/backends 1e-10, trajectory discrepancy/L <=1e-8 and independent
reference error/L <=1e-9. Report measured maxima and the reference convergence
floor; a failing threshold requires investigation, not an unexplained relaxation.
Run focused tests first, then strict mypy, the full headless unittest suite, both
smoke examples, build and the SDD checker. Record unsupported backend skips and
exclude long production notebook runs. Structure validation alone is not evidence
for any numerical criterion.
