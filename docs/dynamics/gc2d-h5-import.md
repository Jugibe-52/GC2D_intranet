# GC2D measured-potential import

## Public interface

```python
from potential import Potential
from dynamics import GuidingCenterDynamics

potential = Potential.load(
    "data/potential/V1/PHI_2.h5",
    indx=(15,),
    spatial_rescaling=False,
    temporal_rescaling=False,
    sigma=None,
    interpolation_order=3,
    spatial_extension="reflected",
)
dynamics = GuidingCenterDynamics(potential, B=1.5, rho=0.0)
```

`JaxPotential.load` inherits this interface and preserves its class. The loader
accepts no `B`, `characteristic_length`, `characteristic_frequency`, `nx` or `ny`;
removed keywords raise `TypeError`. Sample counts remain properties of `Grid`
and parameters of artificial field construction. `spatial_extension` selects
`"reflected"` by default; use `"periodic"` explicitly for the previous direct
periodic construction. See [spatial extension](spatial-extension.md).

## Source data and orientation

`Rcells` and `Zcells` are finite, increasing, uniformly spaced one-dimensional
axes. `freqs` is one-dimensional and stores cyclic frequencies, in cycles per
source time unit (Hz when time is seconds). `fields` has shape `(len(freqs), len(Zcells), len(Rcells))`.
The existing square-grid/common-period restriction and matrix interpretation
are retained: arrays are **not transposed**. This convention must be considered
when interpreting spatial derivatives. Rectangular sampled fields are rejected.
The base-square width is `L=n*dx`, including the unsampled upper endpoint;
it is not `last-first`. `Grid.period` retains this base width. The physical
period is `potential.spatial_period`: `2*L` for reflection and `L` for direct
periodicity. Spatial rescaling still uses `L`, so the reflected unit square
belongs to a field of period 2.

The constant component is always `fields[0].real`, with `freqs[0] == 0` exactly.
`indx` selects original variable-field indices, preserving order and pairing.
The default is `(15,)`; `None` selects all finite positive-frequency entries;
`()` selects only the constant component. Invalid selected indices or frequencies
fail explicitly. There is no amplitude ranking or one-based index conversion.

With both flags false and `sigma=None`, source samples, cyclic frequencies,
origins, spacing and resolution are preserved. Time and coordinates retain
source units (seconds and meters for `PHI_2.h5`). No normalization or resampling
is performed. Reflected fields use a cubic tensor-product spline with zero
normal derivative on all four sides; the resulting reflected scalar is C2
and its canonical Hamiltonian field is C1. Direct-periodic fields retain
orders 2 through 5 and their previous interpolation behavior. Interpolation
error and floating-point evaluation roundoff are separate from preserving
the original half-open sample arrays.

For reflected input, the missing upper sample edges are initially completed
from the opposite source traces as an explicit data convention. This does not
identify the opposite sides dynamically. After gyroaveraging, upper edges are
independent and retained in read-only `reflected_samples`, whose shape is
`(1 + mode_count, nx + 1, ny + 1)` and whose first field is the mean. The FFT
uses the doubled domain and the same physical gyro-radius. Reconstruction
enforces the normal conditions again; it does not overwrite the averaged
upper edges with the lower edges.

## Cyclic frequency convention

The supplied producer uses `np.fft.fftfreq(Nt, d=dt)` and `fft/Nt`; its labels
declare dt in seconds. Thus source f is in Hz, not radians per second. The
imported time-reading routine was not supplied; no extra time-unit conversion
is inferred. Reconstructed time is elapsed time from the first FFT sample.

All constructors store cyclic rates f; both evaluators form omega=2*pi*f:

```text
Phi(t,x,y) = mean(x,y) + 2 Re sum_j modes_j(x,y) exp(i*2*pi*f_j*t)
partial_t^k mode_j = 2 Re [modes_j exp(i*2*pi*f_j*t) (i*2*pi*f_j)^k]
```

`Potential.frequencies` equals raw source f without temporal rescaling and f*T
with it. The spectral factor 2 is unchanged. A mode has physical period 1/f.
Random potentials store f=1/(2*pi), preserving angular rate 1 and their previous
time dependence. A multimode field need not have a common period.

## Equivalent changes of variables

Set `s=L` when `spatial_rescaling=True`, otherwise `s=1`. Set
`T=1/f_ref` when `temporal_rescaling=True`, otherwise `T=1`.
The reference is the **first selected** positive-frequency mode. Mean-only input
cannot request temporal rescaling because it supplies no time reference.

With spatial rescaling, `q=(x-origin)/s` and the grid becomes the half-open unit
cell. Otherwise runtime coordinates equal source coordinates, including origins.
With temporal rescaling, `u=t/T`; otherwise runtime time equals source time.
All mean and mode amplitudes transform together:

```text
P(u,q) = (T/s**2) Phi(T*u, origin+s*q)
f_runtime_j = f_j*T
partial_qx^a partial_qy^b partial_u^c P
    = (T/s**2)*s**(a+b)*T**c * partial_x^a partial_y^b partial_t^c Phi
```

Use the origin term only when spatial rescaling is enabled. These are changes
of dynamical variables, not amplitude min/max normalization. Recover source
amplitudes by multiplying by `s**2/T`. `electric_field` returns minus the
runtime gradient; recovering V/m requires the inverse amplitude and length
factors. Neither time nor trajectories are clipped to one cycle or one cell.

The magnetic field belongs to `GuidingCenterDynamics`, whose finite signed
nonzero `B` forms `H=P/B`. All velocity, Jacobian, Hamiltonian and temporal
momentum evaluations use that same Hamiltonian, stored as `hamiltonian_potential`. `B=1` supports inputs
already expressed as Hamiltonians. Physical gyro-radius transforms as `rho/s`;
initial positions, output times and steps must use the same variable maps.
This GC equivalence does not automatically prescribe FC velocity/rho/eta scaling.

## Filtering and provenance

`sigma=None` disables filtering; zero is numerically unchanged. A finite
nonnegative width enables a reflected Gaussian filter in source grid samples,
separately for the mean and each real/imaginary mode component, before rescaling.

`GC2DH5Metadata` owns read-only source axes, original indices, cyclic source
rates and attributes. It records `spatial_scale`, `time_scale`, both flags and
`sigma`. `normalization_factor` is the amplitude divisor, `s**2/T` for new
imports. Legacy `characteristic_length` and `characteristic_period` remain for
archive interpretation; they are not loader arguments. Runtime arrays also own
read-only storage. Gyroaveraging preserves class and import provenance; applying
B in dynamics retains the original potential separately from its Hamiltonian.

## Existing studies and archives

Historical studies retain their recorded clock T_old=2*pi/f_ref. In meter
coordinates, cycle_time uses 2*pi*Phi/(f_ref*B) and radial uses Phi/(f_ref*B).
Their runtime cyclic rates are f/f_ref. Radial is numerically equal to the new
physical cycle Hamiltonian (T_phys/B)*Phi with T_phys=1/f_ref; this equivalence
does not make archived timestamps physically correct. Preserve those timestamps
and label the time convention `legacy_cycles`. Keep the historical radial factor
1/(2*pi) within the legacy construction, without changing its upstream clock.

`studies.historical_h5.historical_h5_hamiltonian` composes the historical study
representation from raw samples. For recorded length lambda, reference r and B,
coordinates are 2*pi*(X-X0)/lambda, cyclic rates f/r and amplitudes are divided by
r*lambda**2*B/(2*pi)**2. The legacy parameter `angular_reference` names r and
retains its public spelling; by default r is raw f_ref, not physical omega_ref.
The result is already a Hamiltonian, so downstream GC dynamics uses B=1.
These historical helpers explicitly select `spatial_extension="periodic"`.
Original arithmetic, fingerprints and recorded custom periods are preserved.
Selection uses recorded original indices; old ranks are recovered by provenance
helpers, never reinterpreted as source indices.

`sample_study_grid` separately reproduces an explicitly recorded historical
sampling grid. It is not part of loading. New imports retain the source grid;
plotting can evaluate on a separate mesh without changing the potential.
`load_dimensional_h5_field` starts directly from physical samples and builds its
legacy cycle-time Hamiltonian with T_old and B, without normalizing and undoing the import.
Historical resampling rejects reflected fields rather than silently changing
their boundary construction. `RandomPotentialConfig` uses cubic reflection for
new studies; reproducing an older random study requires an explicit periodic
option and its recorded interpolation order.

New potential pickles, study snapshots and solution sidecars identify
`frequency_convention="cycles"`. Known unmarked legacy arrays remain cyclic;
explicitly angular processed arrays divide by 2*pi once when decoded. Their
actual saved phase is preserved, even when their original source-unit assumption
was wrong. Verify original hashes before conversion, reject unknown conventions
and leave saved trajectories unchanged. Source units and time conventions are
separate: new raw imports record `source_frequency_unit="Hz"` and
`time_convention="physical_seconds"` or `"physical_cycles"`; historical composition
records `"legacy_cycles"`. Missing old metadata defaults to `"legacy_unspecified"`.
The legacy property `characteristic_frequency` remains 2*pi/characteristic_period;
it is not the public cyclic rate or necessarily the raw source frequency.

Frozen study bundles execute their own verified historical source and retain
its cyclic convention; do not convert their data under that old evaluator.

New pickles, worker snapshots and solution sidecars also retain
`spatial_extension` and all closed reflected samples, including independent
gyroaveraged upper edges. Missing extension markers are interpreted as
`"periodic"`. Do not infer reflection from a saved trajectory's appearance or
rebuild an old field using the new default. Reflected dynamics have a distinct
fingerprint, including when both models happen to have identical zero fields;
historical periodic fingerprints retain their original bytes. Minimum-image
trajectory diagnostics use the physical period, while initial geometry and
folded displays keep the original base-square width.

## Validation

`tests/test_h5_frequency_units.py` independently reconstructs FFT time samples.
`tests/test_h5_rescaling.py` checks raw samples, harmonic derivatives,
all four variable maps, amplitude-aware mixed derivatives, gyro-radius maps,
B, JAX agreement, serialization, an analytical shear and independently audited
short trajectories. `tests/test_gc2d_h5_potential.py` retains historical
normalization and interpolation regressions through explicit study composition.
`tests/test_solution_persistence.py`, `tests/test_potential_snapshot.py` and
`tests/test_spatial_extension_study_contracts.py` check independent reflected
endpoint storage, historical fallbacks and model-specific reference identities.
See [the change record](../../specs/changes/2026-10-18-hdf5-frequency-units/tasks.md)
for observed errors, consumer coverage and execution limitations.
