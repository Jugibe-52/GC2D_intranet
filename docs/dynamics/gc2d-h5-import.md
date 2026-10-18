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
)
dynamics = GuidingCenterDynamics(potential, B=1.5, rho=0.0)
```

`JaxPotential.load` inherits this interface and preserves its class. The loader
accepts no `B`, `characteristic_length`, `characteristic_frequency`, `nx` or `ny`;
removed keywords raise `TypeError`. Sample counts remain properties of `Grid`
and parameters of artificial field construction.

## Source data and orientation

`Rcells` and `Zcells` are finite, increasing, uniformly spaced one-dimensional
axes. `freqs` is one-dimensional and stores cyclic frequencies, in cycles per
source time unit (Hz when time is seconds). `fields` has shape `(len(freqs), len(Zcells), len(Rcells))`.
The existing square-grid/common-period restriction and matrix interpretation
are retained: arrays are **not transposed**. This convention must be considered
when interpreting spatial derivatives. Rectangular sampled fields are rejected.
The periodic cell length is `L=n*dx`, including the final interval back to the
first sample; it is not `last-first`.

The constant component is always `fields[0].real`, with `freqs[0] == 0` exactly.
`indx` selects original variable-field indices, preserving order and pairing.
The default is `(15,)`; `None` selects all finite positive-frequency entries;
`()` selects only the constant component. Invalid selected indices or frequencies
fail explicitly. There is no amplitude ranking or one-based index conversion.

With both flags false and `sigma=None`, source samples, cyclic frequencies,
origins, spacing and resolution are preserved. Time and coordinates retain
source units (seconds and meters for `PHI_2.h5`). No normalization or resampling
is performed. Periodic splines interpolate the samples; interpolation error is
separate from sample preservation. Supported spline orders are 2 through 5.

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
momentum evaluations use that same effective Hamiltonian. `B=1` supports inputs
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
Original arithmetic, fingerprints and recorded custom periods are preserved.
Selection uses recorded original indices; old ranks are recovered by provenance
helpers, never reinterpreted as source indices.

`sample_study_grid` separately reproduces an explicitly recorded historical
sampling grid. It is not part of loading. New imports retain the source grid;
plotting can evaluate on a separate mesh without changing the potential.
`load_dimensional_h5_field` starts directly from physical samples and builds its
legacy cycle-time Hamiltonian with T_old and B, without normalizing and undoing the import.

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

## Validation

`tests/test_h5_frequency_units.py` independently reconstructs FFT time samples.
`tests/test_h5_rescaling.py` checks raw samples, harmonic derivatives,
all four variable maps, amplitude-aware mixed derivatives, gyro-radius maps,
B, JAX agreement, serialization, an analytical shear and independently audited
short trajectories. `tests/test_gc2d_h5_potential.py` retains historical
normalization and interpolation regressions through explicit study composition.
See [the change record](../../specs/changes/2026-10-18-hdf5-frequency-units/tasks.md)
for observed errors, consumer coverage and execution limitations.
