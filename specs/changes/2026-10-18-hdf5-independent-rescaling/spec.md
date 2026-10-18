+++
schema_version = 2
kind = "change-spec"
id = "2026-10-18-hdf5-independent-rescaling"
revision = 2
status = "verified"
approval = "2026-10-18: User requested implementation with B in dynamics, angular frequencies, no transposition and preservation of the viewer dynamics and historical results."
sources = ["src/potential/potential.py", "src/potential/load.py", "src/potential/_evaluation.py", "src/potential/jax_potential.py", "src/dynamics/gc.py", "src/dynamics/fc.py", "src/studies/dimensional_h5_midpoint.py", "src/studies/poincare_rho_sweep.py", "tests/test_gc2d_h5_potential.py", "tests/test_potential_snapshot.py"]
documents = ["specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md"]
requirements = ["POT-001", "POT-002", "POT-003", "POT-004", "POT-005"]
proposed_requirements = []
open_questions = []

[[criteria]]
id = "AC-001"
description = "With both rescaling flags false and sigma=None, preserve original coordinate origins, spacing, sample counts, mean and selected complex amplitudes, without resampling or physical normalization."
requirements = ["POT-001", "POT-004", "POT-005"]

[[criteria]]
id = "AC-002"
description = "Remove B, characteristic_length, characteristic_frequency, nx and ny from loading; expose independent Boolean spatial_rescaling and temporal_rescaling, false by default."
requirements = ["POT-001", "POT-003", "POT-004"]

[[criteria]]
id = "AC-003"
description = "Reconstruct physical time with exp(i*omega*t), preserve the spectral factor 2, and verify first, second and mixed time derivatives against an independent harmonic reference."
requirements = ["POT-004"]

[[criteria]]
id = "AC-004"
description = "For all four combinations, transform the potential by T/s**2, coordinates and angular frequencies consistently; verify inverse maps and amplitude-aware chain rules, unit cell and unit reference cycle."
requirements = ["POT-004"]

[[criteria]]
id = "AC-005"
description = "Retain original-index selection, mean-only loading without temporal rescaling, validation, optional per-component smoothing, immutable provenance and subclass preservation."
requirements = ["POT-002", "POT-003", "POT-005"]

[[criteria]]
id = "AC-006"
description = "Preserve NumPy/JAX agreement, random-potential behavior, and explicit frequency semantics across construction, gyroaveraging and persistence."
requirements = ["POT-004", "POT-005"]

[[criteria]]
id = "AC-007"
description = "Demonstrate equivalent guiding-center trajectories in physical and rescaled units with consistent Hamiltonian, rho, initial states and time; preserve historical radial and cycle_time conventions explicitly."
requirements = ["POT-004", "POT-005"]

[[criteria]]
id = "AC-008"
description = "Document the supported API, units, removed arguments, archive interpretation and a readable separation of import and transformations, without leaving active consumers on ambiguous conventions."
requirements = ["POT-001", "POT-003", "POT-004", "POT-005"]
+++

# Independent HDF5 rescaling: specification

## Problem

The former importer coupled measured samples to magnetic, spatial and temporal
normalization. The accepted interface separates loading from equivalent changes
of variables and stores angular frequencies directly throughout evaluation.

## Scope

Refine POT-001 through POT-005 across loading, derivatives, GC dynamics,
provenance, persistence and every active consumer, including local development
notebooks. Preserve the existing matrix orientation without transposition and
the square-grid/common-period restriction. Exclude numerical-method changes,
remote jobs, bucket downloads, production integrations and rewriting saved
trajectories. Work remains on `main`, without commits or publication.

Revision 2 supersedes the earlier cyclic-frequency and transpose proposals.
The user accepted B in dynamics, angular frequencies, readable transformations
and preservation of the radial viewer's dynamics and historical results on
2026-10-18. No physical decision remains open.

## Proposed behavior

`Potential.load(filename, *, indx=(15,), spatial_rescaling=False,
temporal_rescaling=False, sigma=None, interpolation_order=3)` is the supported
interface. Removed B, characteristic length/frequency and nx/ny keywords raise
TypeError. Defaults preserve raw samples, source axes, spacing and resolution.
Field 0 is the real mean with exactly zero source frequency. Select original
variable indices in caller order, with finite positive angular frequencies;
None means all positive modes and () means constant only. No sorting or transpose.

Both flags are Boolean. Filtering uses a finite nonnegative sigma in samples,
independently on the real mean and real/imaginary mode components before scaling;
None and zero leave values unchanged. Keep orders 2 through 5, immutable owned
arrays and inherited JaxPotential/subclass behavior.

Runtime phase is exp(i*omega*t), with spectral factor 2 and derivative factors
(i*omega)**dt for dt=0,1,2. Random potentials have omega=1. Convert known legacy
cyclic frequencies once at decoding; reject unknown conventions. Retain grids,
amplitudes, orientation, historical trajectories and fingerprint semantics.

Let s=L=n*dx if spatial rescaling is enabled, otherwise s=1. Let
T=2*pi/omega_ref if temporal rescaling is enabled, otherwise T=1. The reference
is the first selected mode; reject temporal rescaling for a constant-only field.
Use q=(x-origin)/s only with spatial rescaling, otherwise retain x; use u=t/T.
Transform all amplitudes by T/s**2 and angular rates by T. Thus
P(u,q)=(T/s**2)*Phi(T*u,origin+s*q), with the origin used only for spatial
rescaling. Derivatives acquire (T/s**2)*s**(dx+dy)*T**dt. Recover source
potential/field units using the inverse factors. Do not normalize amplitudes
to [0,1] or clip trajectories/time. A reference cycle need not be a common
period for a multimode field.

GC dynamics receives finite signed nonzero B, default 1 for prepared Hamiltonians,
and uses H=P/B for velocity, Jacobians, energy and temporal momentum derivative.
Map physical rho to rho/s and transform states, steps and sample times consistently.
B is absent from the loader and coordinate transformation. Audit FC consumers
without blindly applying GC scaling to their velocities, rho or eta.

Keep radial's additional amplitude 1/(2*pi) at unchanged forcing period; angular
storage does not cancel it. Preserve the distinct cycle_time convention, old
rho_m=rho_hat*lambda/(2*pi), original source selections and custom archived periods.
Compose these recorded study conventions explicitly from raw loading, without
normalizing and undoing normalization inside the new importer. Preserve legacy
sampling separately where needed to reconstruct the original numerical field.

Metadata records original axes, indices, angular rates, scales, flags, filtering
and amplitude divisor. Version pickle/snapshot/solution frequency semantics.
Migrate Python, examples, tests, live docs and all active notebooks, including
aliases and forwarded keyword dictionaries; explain archival exclusions.

AC-001 through AC-008 retain their identifiers. Require exact raw stored samples;
phase/coordinate rtol=atol=1e-12; derivative maps and NumPy/JAX rtol=atol=1e-10.
Separate interpolation error from coordinate and frequency errors. For short
GC trajectories require max position discrepancy/L <=1e-8 and independently
measured reference discrepancy/L <=1e-9. Solver tolerances are not measured errors.

## Outcome and limitations

Implementation and observed evidence are recorded in [tasks.md](tasks.md).
Historical outputs remain intact. Reduced notebook validation covers affected
load/composition paths, not complete production studies or remote execution.

Historical correction (2026-10-18): the producer evidence subsequently established
that source freqs is cyclic Hz. The numerical results above remain the observed
results of this change; their angular-source assumption is superseded by
[the frequency-unit correction](../2026-10-18-hdf5-frequency-units/spec.md).
