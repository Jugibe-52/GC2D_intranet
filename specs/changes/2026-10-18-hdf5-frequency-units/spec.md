+++
schema_version = 2
kind = "change-spec"
id = "2026-10-18-hdf5-frequency-units"
revision = 2
status = "verified"
approval = "2026-10-18: User requested $sdd implementa after reviewing spec revision 2 and plan revision 1; cyclic public frequencies and historical viewer preservation accepted."
sources = ["src/potential/load.py", "src/potential/potential.py", "src/potential/_evaluation.py", "src/potential/jax_potential.py", "src/dynamics/gc.py", "src/studies/historical_h5.py", "src/studies/dimensional_h5_midpoint.py", "src/studies/poincare_rho_sweep.py", "src/studies/h5_provenance.py", "src/studies/_potential_snapshot.py", "src/studies/poincare_probe.py", "src/diagnostics/io/persistence.py", "tests/test_h5_rescaling.py", "tests/test_gc2d_h5_potential.py", "tests/test_potential_snapshot.py", "tests/test_solution_persistence.py"]
documents = ["specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/spec.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/plan.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/tasks.md"]
requirements = ["POT-001", "POT-002", "POT-003", "POT-004", "POT-005"]
proposed_requirements = []
open_questions = []

[[criteria]]
id = "AC-001"
description = "Treat source freqs as cycles per source time unit, preserving raw samples and source frequency provenance; establish the producer relationship with an independent FFT fixture."
requirements = ["POT-001", "POT-002", "POT-005"]

[[criteria]]
id = "AC-002"
description = "Reconstruct the source signal with exp(i*2*pi*f*t), spectral factor 2, and first/second/mixed time derivatives; use cycles per runtime time unit in Potential.frequencies for all constructors and backends."
requirements = ["POT-004"]

[[criteria]]
id = "AC-003"
description = "Use T=1/f_ref for physical temporal rescaling, preserve all four coordinate/amplitude maps T/s**2, first-selected reference, mean-only policy and loader signature."
requirements = ["POT-001", "POT-003", "POT-004"]

[[criteria]]
id = "AC-004"
description = "Preserve historical viewer equations, radial/cycle_time amplitudes, custom clocks, saved states, provenance checks and sampled-field fingerprints without relabeling old times as corrected physical seconds."
requirements = ["POT-004", "POT-005"]

[[criteria]]
id = "AC-005"
description = "Decode legacy unmarked cyclic and explicitly angular processed archives according to their stored convention; identify new source/time provenance, reject unknown conventions and prevent double conversion."
requirements = ["POT-005"]

[[criteria]]
id = "AC-006"
description = "Migrate all affected Python, notebook, generator, visualization and documentation consumers, including ignored notebooks; preserve random-field behavior, B in dynamics, orientation, filtering and subclass/backend contracts."
requirements = ["POT-001", "POT-002", "POT-003", "POT-004", "POT-005"]

[[criteria]]
id = "AC-007"
description = "Verify equivalent physical GC trajectories across all rescaling flags using an independent time reference and a quantified solver audit, separately from historical reproduction."
requirements = ["POT-004"]
+++

# HDF5 frequency units: specification

## Problem

The producer supplied by the user computes `np.fft.fftfreq(Nt, d=dt)`, copies the
selected value into `freqs_final`, and writes it unchanged to HDF5 `freqs`.
These are cycles per unit of dt, not angular rates. The script labels dt in
seconds and its spectrum in Hz. For source time in seconds, f is in Hz = s^-1,
and the angular rate is omega=2*pi*f. Dimensional notation s^-1 alone does not
distinguish cycles from radians; the FFT definition establishes that distinction.

Evidence inspected on 2026-10-18: [extractSingleFreq.py](/home/juan/Descargas/extractSingleFreq.py),
SHA-256 `0b6dc6cfc57bc023b4fb9d5939c019b661da8e9c7328ffb722e37b9afdc501fa`.
Relevant operations: dt from successive tsimu values, fftfreq, fft/Nt,
mean restoration in field 0, selection and writing freqs_final. The producer is
external evidence, not a repository dependency to execute or modify. Its imported
load_tsimu implementation and original time series were not supplied: this change
assumes source time is seconds as declared; it must not invent a tau0 conversion.
Reconstructed time is elapsed time from the FFT sample origin, whose absolute
timestamp is not stored by this producer.

The current working tree instead interprets raw f as omega, uses exp(i*f*t),
and derives a physical period 2*pi/f. This is a factor-2*pi error for the FFT
signal in seconds. The earlier verified change tested internal consistency and
historical reproducibility under that incorrect source-unit assumption; its
recorded passes are not independent evidence for the physical input units.

## Scope

Refine existing POT-001 through POT-005. Correct source interpretation, temporal
phase/derivatives/rescaling, runtime frequency representation, metadata and all
affected consumers. Preserve B in dynamics, current spatial orientation without
transposition, square/common-period grid limits, original-index selection,
optional filtering, and the reduced public loader signature.

Preserve the user's explicit decision to keep viewer dynamics and historical
results. Do not modify producer data, saved trajectories, frozen study packages,
production outputs, Git tracking or personal configuration. No full production
notebook execution, remote jobs, downloads, publication or commits. Remain on
main and preserve the existing uncommitted implementation. This is a new
corrective change; retain the previous change's records and observed history.
Current contracts and MEMORY are updated only during accepted implementation.

## Proposed behavior

### Source and public frequency representation

`source_frequencies` records original cyclic f exactly, in Hz when source time
is seconds. Keep source indices/order and raw mean/complex samples. For every
selected positive mode, the physical reconstruction is:

```text
Phi(t,x,y) = mean(x,y) + 2 Re sum C_j(x,y)*exp(i*2*pi*f_j*t)
partial_t^k mode_j = 2 Re [C_j*exp(i*2*pi*f_j*t)*(i*2*pi*f_j)**k]
```

Keep the existing positive-mode contract. Independent FFT tests use nonzero,
non-Nyquist positive bins and a real conjugate pair; repairing the producer's
argmax/sign selection or broadening negative/Nyquist support is outside scope.

On 2026-10-18 the user resolved Q1: keep Hz in `Potential.frequencies`.
The public array therefore contains cycles per runtime time unit: raw f without
temporal scaling, f*T with scaling. The kernel forms 2*pi*f for phase and
derivatives. Use f where it avoids confusing cyclic and angular quantities;
retain established public identifiers. Every constructor and backend uses this
same convention. The angular public-array alternative is not selected.

### Physical changes of variables

Keep `Potential.load(..., indx=(15,), spatial_rescaling=False,
temporal_rescaling=False, sigma=None, interpolation_order=3)`.
Let s=L=n*dx with spatial scaling, otherwise s=1. Let T=1/f_ref with temporal
scaling, otherwise T=1, using the first selected positive mode. Preserve mean-only
loading without temporal scaling and reject a missing temporal reference.

```text
u = t/T
q = (x-origin)/s when spatial scaling is enabled; otherwise q=x
P(u,q) = (T/s**2)*Phi(T*u, origin+s*q)
f_runtime = f*T                 # cyclic public convention
omega_runtime = 2*pi*f_runtime  # evaluation factor, not the stored public array
D_qx^a D_qy^b D_u^c P = (T/s**2)*s**(a+b)*T**c * D_x^a D_y^b D_t^c Phi
```

Use the origin term only with spatial scaling. Multiply mean and all modes by
the same T/s**2. The reference mode has normalized period 1 (cyclic rate 1,
angular rate 2*pi); a multimode reference cycle need not be a common period.
H=P/B remains in GC dynamics. Map rho, initial states, steps and sample times
consistently. Raw amplitudes do not change; only correct physical time scaling
changes the amplitude factor relative to the currently implemented loader.

### Historical numerical equations and physical interpretation

In the existing dimensional study, with f_ref the raw source value:

| Representation in meter coordinates | Amplitude multiplying Phi | Recorded reference clock |
| --- | --- | --- |
| Correct physical cycle Hamiltonian | 1/(f_ref*B) | 1/f_ref seconds |
| Historical cycle_time | 2*pi/(f_ref*B) | 2*pi/f_ref, legacy interpretation |
| Historical radial | 1/(f_ref*B) | same recorded legacy clock |

All three use phase 2*pi*(f_j/f_ref)*u when indexed by their normalized cycle
parameter. Thus historical radial and the corrected physical cycle Hamiltonian
are numerically equal as functions of that parameter and meter coordinates.
This does not make the recorded legacy seconds correct or authorize replacing
those timestamps. In characteristic coordinates multiply each Hamiltonian by
(2*pi/lambda)**2 and map rho as already recorded.

Keep radial's existing extra 1/(2*pi) within the explicit historical study
construction and preserve cycle_time as a distinct legacy equation. Do not
replace its upstream clock by 1/f and leave the old extra factor silently in
place: that would slow the radial trajectories by 2*pi. Likewise, new physical
loads must not inherit the legacy clock. Custom archived periods stay authoritative
for reproduction, even when different from both reference formulas.

### Storage and acceptance

Keep source units, runtime frequency convention and time-normalization convention
separate in provenance. Processed arrays already labeled angular remain angular
input to their reader, even when produced under the former incorrect source
interpretation. Preserve their actual phase; do not multiply them again because
the source dataset has now been identified as Hz. Known unmarked legacy processed
archives remain cyclic. Unknown conventions fail explicitly.

For float64 order-one fixtures require exact raw arrays and source rates;
FFT reconstruction/period checks rtol=atol=1e-12; temporal/mixed derivatives,
variable maps and NumPy/JAX agreement rtol=atol=1e-10. Measure spline approximation
separately. Require transformed GC position discrepancy/L <=1e-8 and independent
DOP853/Radau reference discrepancy/L <=1e-9. Preserve representative historical
fingerprints exactly; no replacement of expected hashes to conceal a mismatch.
For general archive representation round trips use numerical tolerances and
verify stored hashes in their original convention before any explicit conversion.

## Outcome and limitations

Implemented and verified on 2026-10-18. Public frequencies remain cyclic;
physical temporal rescaling uses T=1/f_ref and historical viewer results retain
their legacy clocks and dynamics. Q1 was resolved by the user in revision 2.
See [tasks.md](tasks.md) for measured FFT/trajectory errors, historical hashes,
archive coverage, consumer inventory, the corrected test-only expectation and
explicit GPU skips. No production notebook sweep, publication or commit was made.
The original time-reading routine was not supplied; source seconds follow the
producer's declared units, as documented above.
